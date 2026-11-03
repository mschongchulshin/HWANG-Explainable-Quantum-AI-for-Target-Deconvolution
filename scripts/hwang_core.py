import os

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVC

import qsim_torch as qsim

ACT, INACT, MIN_NEG = 1000.0, 10000.0, 20
K = 2
DEV = "cuda" if torch.cuda.is_available() else "cpu"
S = {}


def init(out, ef):
    gene_meta = pd.read_csv(os.path.join(out, "gene_meta.csv"))
    cmp_meta = pd.read_csv(os.path.join(out, "cmp_meta.csv"))
    cmp_emb = np.load(os.path.join(out, "cmp_emb.npy"))
    gene_emb = np.load(os.path.join(out, "gene_emb.npy"))
    chem = pd.read_csv(os.path.join(out, "chem_meta.csv"))
    groups = pd.read_csv(os.path.join(out, "chem_groups.csv"))
    desc = np.load(os.path.join(out, "chem_desc.npy"))
    fp = np.load(os.path.join(out, "chem_ecfp.npy")).astype(np.float32)

    genes = gene_meta["perturbation"].tolist()
    gpos = {g: i for i, g in enumerate(genes)}
    chem_c = chem["perturbation"].tolist()
    cpos = {c: i for i, c in enumerate(chem_c)}
    clusters = dict(zip(groups["perturbation"], groups["cluster"]))
    cvec = (pd.DataFrame(cmp_emb, index=cmp_meta["perturbation"].to_numpy())
            .groupby(level=0).mean())
    cvi = {c: i for i, c in enumerate(cvec.index.tolist())}

    gt = pd.read_csv(os.path.join(ef, "compound_gene_interactions.csv"))
    gt = gt[gt.gene_symbol.isin(gpos) & gt.treatment.isin(set(cvec.index))
            & gt.treatment.isin(set(chem_c))].copy()
    gt["y"] = np.where(gt.nM_value < ACT, 1, np.where(gt.nM_value > INACT, 0, -1))
    gt = gt[gt.y >= 0].reset_index(drop=True)
    gt["cluster"] = gt.treatment.map(clusters).fillna(-1).astype(int)
    POS = gt[gt.y == 1].groupby("treatment")["gene_symbol"].apply(set).to_dict()
    NEG = gt[gt.y == 0].groupby("treatment")["gene_symbol"].apply(set).to_dict()

    Vc = cvec.to_numpy(dtype=np.float32)
    Vc = Vc / (np.linalg.norm(Vc, axis=1, keepdims=True) + 1e-12)
    Vg = gene_emb / (np.linalg.norm(gene_emb, axis=1, keepdims=True) + 1e-12)
    CG = (Vc @ Vg.T).astype(np.float32)
    PCOS = (Vc @ Vc.T).astype(np.float32)
    _c2v = np.array([cvi[c] for c in chem_c], dtype=int)
    PCOS = PCOS[np.ix_(_c2v, _c2v)]
    _i = fp @ fp.T
    _p = fp.sum(1)
    TAN = (_i / np.maximum(_p[:, None] + _p[None, :] - _i, 1e-9)).astype(np.float32)
    np.fill_diagonal(TAN, 1.0)
    dz = StandardScaler().fit_transform(np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0))

    def _pca(M):
        return PCA(n_components=min(K, *M.shape), random_state=0).fit_transform(M).astype(np.float32)

    S.update(dict(genes=genes, gpos=gpos, NG=len(genes), chem_c=chem_c, cpos=cpos,
                  clusters=clusters, cvi=cvi, gt=gt, POS=POS, NEG=NEG,
                  CG=CG, PCOS=PCOS, TAN=TAN, D_PCA=_pca(dz), C_PCA=_pca(Vc), G_PCA=_pca(Vg)))


def gene_actives(train_gt):
    by = {}
    for g, grp in train_gt[train_gt.y == 1].groupby("gene_symbol"):
        cols, cl = [], []
        for c in grp["treatment"].unique():
            j = S["cpos"].get(c)
            if j is None:
                continue
            cols.append(j); cl.append(S["clusters"].get(c, -2))
        if cols:
            by[g] = (np.asarray(cols), np.asarray(cl))
    return by


def tf_pairs(sim, by, ci_arr, cl_arr, gi_arr):
    NG = S["NG"]
    genes = S["genes"]
    v = np.zeros(len(ci_arr), dtype=np.float32)
    order = np.argsort(gi_arr, kind="stable")
    gi_s = gi_arr[order]
    bounds = np.searchsorted(gi_s, np.arange(NG + 1))
    for g_i in range(NG):
        lo, hi = bounds[g_i], bounds[g_i + 1]
        if lo == hi:
            continue
        e = by.get(genes[g_i])
        if e is None:
            continue
        cols, cl = e
        loc = order[lo:hi]
        sub = sim[np.ix_(ci_arr[loc], cols)].astype(np.float32)
        mask = cl[None, :] != cl_arr[loc][:, None]
        sub = np.where(mask, sub, -np.inf)
        m = sub.max(axis=1)
        m[~np.isfinite(m)] = 0.0
        v[loc] = m
    return v


def featurise(by, cnames, gnames):
    ci = np.array([S["cpos"][c] for c in cnames])
    cj = np.array([S["cvi"][c] for c in cnames])
    gi = np.array([S["gpos"][g] for g in gnames])
    cl = np.array([S["clusters"].get(c, -1) for c in cnames])
    cos = S["CG"][cj, gi]
    return np.hstack([
        tf_pairs(S["TAN"], by, ci, cl, gi)[:, None],
        tf_pairs(S["PCOS"], by, ci, cl, gi)[:, None],
        cos[:, None], np.abs(cos)[:, None],
        S["D_PCA"][ci, :K], S["C_PCA"][cj, :K], S["G_PCA"][gi, :K]]).astype(np.float64)


def prep(tr_gt, test_cmps, te_gt=None):
    by = gene_actives(tr_gt)
    Xtr = featurise(by, tr_gt.treatment.tolist(), tr_gt.gene_symbol.tolist())
    cn = [c for c in test_cmps for _ in S["genes"]]
    gn = S["genes"] * len(test_cmps)
    Xg = featurise(by, cn, gn)
    Xm, ym = None, None
    if te_gt is not None and len(te_gt):
        Xm = featurise(by, te_gt.treatment.tolist(), te_gt.gene_symbol.tolist())
        ym = te_gt.y.to_numpy()
    return Xtr, tr_gt.y.to_numpy(), Xg, len(test_cmps), Xm, ym


def fit_score(name, p, Xtr, ytr, Xg):
    qt = QuantileTransformer(output_distribution="uniform",
                             n_quantiles=min(1000, max(10, len(Xtr))), random_state=0)
    U = qt.fit_transform(Xtr) * p["scale"]
    Ut = torch.as_tensor(U, device=DEV)
    if name != "QK-zz":
        raise ValueError("Only the ZZ kernel is supported")
    n = U.shape[1]
    A = qsim.feature_map(Ut, n, "zz", p["reps"], p["ent"])
    K_tr = qsim.fidelity_kernel(A, A).cpu().numpy()
    m = SVC(C=p["C"], kernel="precomputed", class_weight=p["cw"])
    m.fit(K_tr, ytr)
    out = np.empty(len(Xg))
    for lo in range(0, len(Xg), 20000):
        Ugc = torch.as_tensor(qt.transform(Xg[lo:lo + 20000]) * p["scale"], device=DEV)
        Ag = qsim.feature_map(Ugc, n, "zz", p["reps"], p["ent"])
        Kg = qsim.fidelity_kernel(Ag, A).cpu().numpy()
        out[lo:lo + len(Kg)] = m.decision_function(Kg)
    return out


def bench(score_rows, test_cmps, n_seeds):
    POS, NEG, genes, gpos = S["POS"], S["NEG"], S["genes"], S["gpos"]
    ap_med, auc_med, ap_mean, auc_mean = [], [], [], []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        a, u = [], []
        for i, c in enumerate(test_cmps):
            posg = POS.get(c)
            if not posg:
                continue
            ann = posg | NEG.get(c, set())
            pool = [g for g in genes if g not in ann]
            k = max(MIN_NEG, len(posg))
            if len(pool) < k:
                continue
            negs = rng.choice(pool, size=k, replace=False)
            sel = [gpos[g] for g in list(posg) + list(negs)]
            y = np.r_[np.ones(len(posg)), np.zeros(len(negs))]
            s = score_rows[i, sel].astype(float)
            a.append(average_precision_score(y, s))
            u.append(roc_auc_score(y, s))
        if a:
            ap_med.append(np.median(a)); auc_med.append(np.median(u))
            ap_mean.append(np.mean(a)); auc_mean.append(np.mean(u))
    if not ap_med:
        return 0.0, 0.0, 0.0, 0.0
    return (float(np.mean(ap_med)), float(np.mean(auc_med)),
            float(np.mean(ap_mean)), float(np.mean(auc_mean)))
