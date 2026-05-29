"""
Spatial-honest inference for DML.

Implements:
  - spatial_hac_se: Conley (1999) spatial HAC with triangular Bartlett kernel.
  - salerno_jackknife_hac: Salerno-Wu-McCormick 2026 fold-centered jackknife
    spatial HAC, V_JK = V_off + V_between.
  - buffered_kfold: Emmenegger-style buffered K-fold (training points within
    r_n of any eval point dropped).

References:
  Conley, T. G. (1999) "GMM estimation with cross sectional dependence."
    Journal of Econometrics 92(1):1-45.
  Bester, Conley & Hansen (2011) JoE — Bartlett-kernel sandwich variance.
  Emmenegger, Spohn, Elmer & Buehlmann (2025) "Treatment Effect Estimation
    with Observational Network Data using Machine Learning" arXiv:2206.14591.
  Salerno, Wu, McCormick (2026) "Spatially Robust Inference with Predicted
    and Missing-at-Random Labels" arXiv:2603.11368.
  Cao & Leung (2025) "Neighborhood Stability in DML with Dependent Data"
    arXiv:2511.10995  (cited as the no-cross-fit alternative).
  Lehner (2026) "Data-driven bandwidth selection for Conley HAC"
    arXiv:2603.03997.

The default bandwidth quantile q_h = 0.10 and buffer quantile q_b = 0.05 are
the values explored as workhorse defaults in the simulation section of
Salerno-Wu-McCormick 2026. Sweep both as robustness columns.
"""
from __future__ import annotations
import warnings
from typing import Iterable, List, Tuple

import numpy as np
from scipy.spatial import cKDTree


def _pairwise_dist_quantile(coords: np.ndarray, q: float) -> float:
    """q-th quantile of pairwise Euclidean distances on coords."""
    n = coords.shape[0]
    if n > 4000:
        idx = np.random.default_rng(0).choice(n, 4000, replace=False)
        coords = coords[idx]
    diffs = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt((diffs ** 2).sum(-1))
    iu = np.triu_indices(d.shape[0], k=1)
    return float(np.quantile(d[iu], q))


def spatial_hac_se(scores: np.ndarray,
                   coords: np.ndarray,
                   bandwidth_quantile: float = 0.10,
                   bandwidth: float | None = None) -> float:
    """Conley spatial HAC SE for the mean of `scores`.

    Triangular kernel kappa(u) = max(1 - u, 0). Returns SE of mean(scores)
    under the variance-of-mean scaling Var(theta_hat) = (1/n^2) sum w_ij psi_i psi_j.
    """
    n = scores.shape[0]
    h = bandwidth if bandwidth is not None else _pairwise_dist_quantile(coords, bandwidth_quantile)
    if h <= 0:
        return float(np.std(scores, ddof=1) / np.sqrt(n))
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=h, output_type='ndarray')
    var = float((scores ** 2).sum())
    if pairs.size:
        i, j = pairs[:, 0], pairs[:, 1]
        d = np.linalg.norm(coords[i] - coords[j], axis=1)
        w = np.maximum(1.0 - d / h, 0.0)
        var += 2.0 * float((w * scores[i] * scores[j]).sum())
    var /= n ** 2
    return float(np.sqrt(max(var, 1e-300)))


def salerno_jackknife_hac(scores: np.ndarray,
                          fold_ids: np.ndarray,
                          coords: np.ndarray,
                          bandwidth_quantile: float = 0.10,
                          bandwidth: float | None = None,
                          var_floor: float = 1e-12) -> dict:
    """Salerno-Wu-McCormick (2026) jackknife-HAC: V_JK = V_off + V_between.

    Within-fold centering removes the leading bias from cross-fitted nuisance
    error correlated across observations sharing a fold. V_between captures
    the inter-fold sample-splitting variance that naive DML treats as zero.

    Returns the decomposition so the components can be reported.
    """
    n = scores.shape[0]
    theta_hat = float(scores.mean())

    psi_tilde = np.empty_like(scores, dtype=np.float64)
    fold_means: dict[int, tuple[float, int]] = {}
    K = int(fold_ids.max()) + 1
    for k in range(K):
        mask = fold_ids == k
        if not mask.any():
            continue
        fk = float(scores[mask].mean())
        fold_means[k] = (fk, int(mask.sum()))
        psi_tilde[mask] = scores[mask] - fk

    h = bandwidth if bandwidth is not None else _pairwise_dist_quantile(coords, bandwidth_quantile)
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=h, output_type='ndarray')
    v_within = float((psi_tilde ** 2).sum())
    if pairs.size:
        i, j = pairs[:, 0], pairs[:, 1]
        d = np.linalg.norm(coords[i] - coords[j], axis=1)
        w = np.maximum(1.0 - d / h, 0.0)
        v_within += 2.0 * float((w * psi_tilde[i] * psi_tilde[j]).sum())
    v_within /= n ** 2
    v_diag = float((psi_tilde ** 2).sum()) / n ** 2
    v_off = v_within - v_diag

    v_between = 0.0
    for _, (fk, nk) in fold_means.items():
        v_between += (nk / n) ** 2 * (fk - theta_hat) ** 2
    if K > 1:
        v_between *= K / (K - 1)

    v_jk = v_off + v_between
    v_jk_plus = max(v_jk, var_floor)
    return {
        'se': float(np.sqrt(v_jk_plus)),
        'v_jk': float(v_jk),
        'v_off': float(v_off),
        'v_between': float(v_between),
        'v_diag': float(v_diag),
        'theta_hat': theta_hat,
        'h_n': float(h),
    }


def buffered_kfold(coords: np.ndarray,
                   k: int = 5,
                   buffer_quantile: float = 0.05,
                   buffer_radius: float | None = None,
                   seed: int = 0,
                   min_train_warn: int = 100) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Buffered K-fold under spatial dependence (Emmenegger et al. 2025).

    For each fold the eval set is the standard K-fold eval set; training points
    within `buffer_radius` of any eval point are dropped. Eval points are
    never dropped, so the score average is unbiased.
    """
    n = coords.shape[0]
    r = buffer_radius if buffer_radius is not None else _pairwise_dist_quantile(coords, buffer_quantile)
    tree = cKDTree(coords)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fold_ids = np.empty(n, dtype=int)
    fold_ids[perm] = np.arange(n) % k

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold in range(k):
        eval_idx = np.where(fold_ids == fold)[0]
        cand_train = np.where(fold_ids != fold)[0]
        near = set()
        for lst in tree.query_ball_point(coords[eval_idx], r=r):
            near.update(lst)
        train_idx = np.array([i for i in cand_train if i not in near], dtype=int)
        if train_idx.size < min_train_warn:
            warnings.warn(
                f"buffered fold {fold}: only {train_idx.size} training rows "
                f"(< floor {min_train_warn}); consider smaller buffer_quantile."
            )
        splits.append((train_idx, eval_idx))
    return splits


def bandwidth_sensitivity_sweep(scores: np.ndarray, coords: np.ndarray,
                                fold_ids: np.ndarray | None = None,
                                qs: Iterable[float] = (0.05, 0.10, 0.15, 0.20)) -> dict:
    """Run the recommended sensitivity sweep over bandwidth quantiles.
    Returns the SE at each q plus the max (the value Salerno-Wu-McCormick
    recommend reporting in robustness footnotes)."""
    out = {}
    for q in qs:
        if fold_ids is None:
            out[q] = spatial_hac_se(scores, coords, bandwidth_quantile=q)
        else:
            res = salerno_jackknife_hac(scores, fold_ids, coords,
                                        bandwidth_quantile=q)
            out[q] = res['se']
    out['max'] = float(max(out.values()))
    return out


__all__ = [
    "spatial_hac_se",
    "salerno_jackknife_hac",
    "buffered_kfold",
    "bandwidth_sensitivity_sweep",
]
