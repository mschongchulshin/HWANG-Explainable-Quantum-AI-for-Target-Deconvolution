pass
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold
import hwang_core as hc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True, help='Directory containing prepared RxRx3 feature files')
    ap.add_argument('--annotations', required=True, help='Directory containing compound_gene_interactions.csv')
    ap.add_argument('--output', required=True, help='Directory to write fold predictions and aggregate results')
    ap.add_argument('--fold', type=int, choices=range(10), help='Run one fold; omit for all ten')
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    hc.init(args.data, args.annotations)
    gt = hc.S['gt']
    splits = list(GroupKFold(n_splits=10).split(gt, gt.y.to_numpy(), groups=gt.cluster.to_numpy()))
    params = json.loads((Path(__file__).resolve().parent.parent / 'data' / 'zz_fold_params.json').read_text(encoding='utf-8'))
    predictions = {}
    for f in ([args.fold] if args.fold is not None else range(10)):
        train, test = splits[f]
        tr, te = gt.iloc[train], gt.iloc[test]
        compounds = sorted(set(te.treatment) & set(hc.S['POS']))
        Xtr, ytr, Xg, nc, _, _ = hc.prep(tr, compounds)
        p = dict(params[str(f)])
        p['cw'] = None if p['cw'] == 'none' else p['cw']
        scores = hc.fit_score('QK-zz', p, Xtr, ytr, Xg).reshape(nc, hc.S['NG'])
        np.savez_compressed(out / f'zz_fold{f}.npz', scores=scores, compounds=np.asarray(compounds))
        predictions.update(zip(compounds, scores))
        print(f'fold {f}: {len(compounds)} compounds', flush=True)
    if args.fold is None:
        compounds = sorted(predictions)
        scores = np.vstack([predictions[c] for c in compounds])
        median_ap, median_auc, mean_ap, mean_auc = hc.bench(scores, compounds, 20)
        result = {'n_compounds': len(compounds), 'median_ap': median_ap,
                  'mean_ap': mean_ap, 'median_roc_auc': median_auc, 'mean_roc_auc': mean_auc}
        (out / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
