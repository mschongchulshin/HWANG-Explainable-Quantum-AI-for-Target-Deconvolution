import math

import torch


def zeros_state(N, n, dtype=torch.complex128, device="cpu"):
    psi = torch.zeros((N, 2 ** n), dtype=dtype, device=device)
    psi[:, 0] = 1.0
    return psi


def _zbit(n, q, device):
    idx = torch.arange(2 ** n, device=device)
    return 1.0 - 2.0 * ((idx >> q) & 1).to(torch.float64)


def hadamard_all(psi, n):
    N = psi.shape[0]
    s = 1.0 / math.sqrt(2.0)
    for q in range(n):
        a = psi.reshape(N, 2 ** (n - q - 1), 2, 2 ** q)
        x, y = a[:, :, 0, :], a[:, :, 1, :]
        a = torch.stack(((x + y) * s, (x - y) * s), dim=2)
        psi = a.reshape(N, 2 ** n)
    return psi


def _as_batch(theta, N, device):
    t = torch.as_tensor(theta, dtype=torch.float64, device=device)
    if t.dim() == 0:
        t = t.expand(N)
    return t.reshape(N, 1)


def rz(psi, n, q, theta):
    N = psi.shape[0]
    z = _zbit(n, q, psi.device)
    t = _as_batch(theta, N, psi.device)
    return psi * torch.exp(-0.5j * t * z[None, :]).to(psi.dtype)


def rzz(psi, n, q, r, theta):
    N = psi.shape[0]
    z = _zbit(n, q, psi.device) * _zbit(n, r, psi.device)
    t = _as_batch(theta, N, psi.device)
    return psi * torch.exp(-0.5j * t * z[None, :]).to(psi.dtype)


def normalise(psi):
    nrm = torch.linalg.vector_norm(psi, dim=1, keepdim=True)
    return psi / torch.clamp(nrm, min=1e-12)


def feature_map(X, n, kind="zz", reps=2, entangle="linear"):
    N = X.shape[0]
    psi = zeros_state(N, n, device=X.device)
    X = X.to(torch.float64)
    pairs = [(i, i + 1) for i in range(n - 1)]
    if entangle == "circular" and n > 2:
        pairs = pairs + [(n - 1, 0)]
    elif entangle == "full":
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if kind != "zz":
        raise ValueError(kind)
    for _ in range(reps):
        psi = hadamard_all(psi, n)
        for q in range(n):
            psi = rz(psi, n, q, -2.0 * X[:, q])
        for (q, r) in pairs:
            ang = 2.0 * (math.pi - X[:, q]) * (math.pi - X[:, r])
            psi = rzz(psi, n, q, r, -ang)
    return normalise(psi)


def amplitude_state(X, n):
    N, d = X.shape
    dim = 2 ** n
    A = torch.zeros((N, dim), dtype=torch.complex128, device=X.device)
    m = min(d, dim)
    A[:, :m] = X[:, :m].to(torch.complex128)
    return normalise(A)


def fidelity_kernel(A, B):
    return (A @ B.conj().T).abs() ** 2
