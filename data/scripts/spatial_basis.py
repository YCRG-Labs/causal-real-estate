"""Low-rank thin-plate regression spline (TPRS) basis over lat/lon, for use as a
flexible spatial-confounding control inside the partially-linear DML.

Replaces the 5-column quadratic [lat, lon, lat*lon, lat^2, lon^2], which a
spatial-statistics referee reads as too coarse (it removes geographic variation
only at the metropolitan scale). This is the mgcv s(lat,lon,bs='tp')
construction: project to meters (local equirectangular so the isotropic
thin-plate metric is physical), evaluate the radial kernel eta(r)=r^2 log r on a
knot subsample, eigen-truncate to rank k (Wood 2003), return a fixed,
standardized, full-rank numpy design matrix that column-stacks into the
confounder set exactly like the quadratic did.

Because DML's first stage residualizes both Y and T on the confounders including
this basis, partialling-out with a spatial smooth is the Dupont-Wood-Augustin
'Spatial+' estimator; the only design choice is the basis rank k, swept and
reported as a sensitivity curve (Keller & Szpiro 2020).

References: Wood (2003) JRSS-B 65:95; Kammann & Wand (2003) JRSS-C 52:1;
Dupont, Wood & Augustin (2022) Biometrics 78:1279; Keller & Szpiro (2020)
JRSS-A 183:1121.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

_EARTH_R_M = 6_371_000.0


def _to_meters(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Local equirectangular projection about the data centroid (meters)."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    lat0, lon0 = np.nanmean(lat), np.nanmean(lon)
    x = np.radians(lon - lon0) * np.cos(np.radians(lat0)) * _EARTH_R_M
    y = np.radians(lat - lat0) * _EARTH_R_M
    return np.column_stack([x, y])


def _eta(D: np.ndarray) -> np.ndarray:
    """Thin-plate radial kernel r^2 log r, with eta(0)=0."""
    out = np.zeros_like(D)
    nz = D > 0
    out[nz] = D[nz] ** 2 * np.log(D[nz])
    return out


def thin_plate_basis(lat, lon, k: int = 30, n_knots: int | None = None,
                     seed: int = 0, var_tol: float = 1e-10,
                     return_info: bool = False):
    """Rank-~k TPRS design matrix B(lat,lon) as a fixed standardized array.

    k     target basis dimension (2 linear null-space cols + k-3 radial cols).
          mgcv's default for a 2-D tp smooth is 30; sweep {10,30,50,100}.
    Deterministic given (lat, lon, k, n_knots, seed).
    """
    X = _to_meters(lat, lon)
    n = X.shape[0]

    n_radial = max(k - 3, 1)
    if n_knots is None:
        n_knots = min(n, max(2 * k, 200))
    uniq = np.unique(X, axis=0)
    rng = np.random.default_rng(seed)
    if uniq.shape[0] > n_knots:
        K = uniq[rng.choice(uniq.shape[0], n_knots, replace=False)]
    else:
        K = uniq
    nk = K.shape[0]
    n_radial = min(n_radial, max(nk - 1, 1))

    E_KK = _eta(cdist(K, K))
    w, U = np.linalg.eigh(E_KK)
    order = np.argsort(np.abs(w))[::-1][:n_radial]
    Uk = U[:, order]
    scale = 1.0 / np.sqrt(np.maximum(np.abs(w[order]), 1e-8))

    Phi = (_eta(cdist(X, K)) @ Uk) * scale
    B = np.column_stack([X[:, 0], X[:, 1], Phi])

    sd = B.std(axis=0)
    B = B[:, sd > var_tol]
    B = (B - B.mean(axis=0)) / B.std(axis=0)
    Q, Rm = np.linalg.qr(B, mode="reduced")
    keep = np.abs(np.diag(Rm)) > 1e-7 * np.abs(Rm[0, 0])
    B = B[:, keep]

    if return_info:
        nn = np.sort(cdist(K, K), axis=1)
        info = {"n_knots": int(nk), "rank": int(B.shape[1]),
                "knot_nn_median_m": float(np.median(nn[:, 1])) if nk > 1 else float("nan")}
        return B, info
    return B


def quadratic_basis(lat, lon) -> np.ndarray:
    """The legacy 5-column quadratic, kept for the baseline comparison row."""
    la = np.asarray(lat, float)
    lo = np.asarray(lon, float)
    la = (la - la.mean()) / (la.std() or 1.0)
    lo = (lo - lo.mean()) / (lo.std() or 1.0)
    B = np.column_stack([la, lo, la * lo, la ** 2, lo ** 2])
    return (B - B.mean(0)) / B.std(0)
