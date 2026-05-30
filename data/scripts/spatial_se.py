"""
Spatial-honest inference for DML.

Implements:
  - spatial_hac_se: Conley (1999) spatial HAC with triangular Bartlett kernel.
  - salerno_jackknife_hac: Salerno-Wu-McCormick 2026 fold-centered jackknife
    spatial HAC, V_JK = V_off + V_between.
  - buffered_kfold: Emmenegger-style buffered K-fold (training points within
    r_n of any eval point dropped).
  - ibragimov_muller_self_normalized_ci: Ibragimov-Müller (2010) self-normalized
    cluster-t for the headline inference column. Empirically calibrated as the
    operative SE for our DGP per diag_dk_coverage_sweep.json (coverage 0.73 vs
    0.50 for plain SWM-JK; see Davis-Kahan appendix).
  - salerno_jackknife_t_critical: SWM-JK SE paired with Bester-Conley-Hansen
    2011 fixed-cluster t_{K-1} critical value. Recommended robustness margin.

References:
  Conley, T. G. (1999) "GMM estimation with cross sectional dependence."
    Journal of Econometrics 92(1):1-45.
  Andrews (1991) Econometrica 59:817 — Bartlett-kernel HAC.
  Newey & West (1987) Econometrica 55:703 — original heteroskedasticity- and
    autocorrelation-consistent variance estimator with Bartlett weights.
  Bester, Conley & Hansen (2011) JoE 165:137 — cluster covariance estimators,
    fixed-K asymptotics with t_{K-1} critical values.
  Ibragimov & Müller (2010) JBES 28:453 — self-normalized cluster t-test.
  Sun, Phillips & Jin (2008) Econometrica 76:175 — fixed-b HAR inference.
  Kiefer & Vogelsang (2005) Econometric Theory 21:1130 — HAR alternative
    asymptotics; fixed-b critical values.
  Pötscher & Preinerstorfer (2020) arXiv:1910.00302 — valid fixed-b HAR.
  Cameron, Gelbach & Miller (2008) REStat 90:414 — wild cluster bootstrap.
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

Empirical calibration (Davis-Kahan stress test, n=348, d=768, Matern-2.5,
rho=0.15, 500 reps x 15 spike levels):

  Method                                         Coverage at nominal 0.95
  Classical OLS                                  0.429
  HC1 / White                                    0.438
  SWM-JK (V_off only, fold-centered Bartlett)    0.504
  SWM-JK (V_off + V_between)                     0.490
  SWM-JK bandwidth-sweep max                     0.513
  Conley HAC at Lehner 2026 data-driven BW       0.479
  BCH 2011 cluster cov, t_{K-1=4}                0.539
  WCB (CGM 2008) folds as clusters               0.335
  Block bootstrap                                0.532
  Ibragimov-Müller self-normalized t_{K-1=4}     0.728   <-- recommended
  Fixed-b at b=0.5, t_{1}                        0.940   <-- conservative envelope

See results/diagnostics/dk_coverage_sweep.json for the full rho x method matrix
and the V_JK = V_off + V_between decomposition. Empirical fold-mean SD is 3.45x
the iid theoretical -- direct evidence of fold-shared nuisance error -- but the
V_between correction alone closes only ~3% of the gap because V_off and
V_between have opposite signs of comparable magnitude in this DGP.
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


def ibragimov_muller_self_normalized_ci(scores: np.ndarray,
                                        fold_ids: np.ndarray,
                                        alpha: float = 0.05) -> dict:
    """Self-normalized cluster-t CI (Ibragimov-Müller 2010 JBES 28:453).

    Recipe (K-fixed asymptotics, no spatial bandwidth needed):
      1. Within each fold k, compute bar_psi_k = mean of scores over fold k.
      2. bar_psi = (1/K) Sum_k bar_psi_k.
      3. s^2 = (1 / (K - 1)) Sum_k (bar_psi_k - bar_psi)^2.
      4. CI: bar_psi +/- t_{1-alpha/2, K-1} * sqrt(s^2 / K).

    Conditions (IM 2010 Theorem 1):
      - Within-fold means are approximately independent and Gaussian under
        the null; the spatial dependence within a fold is absorbed into
        bar_psi_k's sampling distribution.
      - K >= 2 and finite. The test is exact in finite samples when the
        within-fold means are iid Gaussian.
      - The variance estimator s^2 / K does NOT need a Bartlett kernel or a
        bandwidth -- it is computed across folds.

    Why this is the recommended headline for our DML pipeline:
      In the Davis-Kahan stress test (n=348, K=5) Salerno V_off+V_between
      reaches 0.49 coverage and the underlying SE is biased downward by
      ~4-5x against the true sampling spread of theta_hat because the
      fold-shared nuisance error correlates with the spatial residual at
      ranges far larger than the Conley bandwidth. The IM cluster-t fully
      absorbs that fold-shared component into bar_psi_k's spread and
      delivers 0.73 coverage without any kernel choice, at the cost of a
      ~1.5x wider CI than Salerno's nominal-z interval (which was anyway
      wrong about its width).

    Returns:
      theta_hat: bar_psi (pooled point estimate)
      se: sqrt(s^2 / K)
      ci_low, ci_high: bar_psi -/+ t_{1-alpha/2, K-1} * se
      t_critical: t_{1-alpha/2, K-1}
      K_eff: number of folds used (excludes empty folds)

    References:
      Ibragimov & Müller 2010 JBES 28(4):453-468.
      Bester-Conley-Hansen 2011 JoE 165:137 -- fixed-K asymptotics view.
    """
    from scipy import stats as _stats
    K = int(fold_ids.max()) + 1
    means = []
    for k in range(K):
        m = fold_ids == k
        if m.sum() < 2:
            continue
        means.append(float(scores[m].mean()))
    means = np.array(means)
    K_eff = len(means)
    if K_eff < 2:
        return {
            'theta_hat': float('nan'), 'se': float('nan'),
            'ci_low': float('nan'), 'ci_high': float('nan'),
            't_critical': float('nan'), 'K_eff': K_eff,
        }
    bar = float(means.mean())
    s2 = float(means.var(ddof=1))
    se = float(np.sqrt(s2 / K_eff))
    tcrit = float(_stats.t.ppf(1.0 - alpha / 2.0, df=K_eff - 1))
    return {
        'theta_hat': bar, 'se': se,
        'ci_low': bar - tcrit * se,
        'ci_high': bar + tcrit * se,
        't_critical': tcrit,
        'K_eff': K_eff,
    }


def salerno_jackknife_t_critical(scores: np.ndarray,
                                 fold_ids: np.ndarray,
                                 coords: np.ndarray,
                                 alpha: float = 0.05,
                                 bandwidth_quantile: float = 0.10,
                                 bandwidth: float | None = None) -> dict:
    """SWM-JK SE paired with the BCH 2011 fixed-K t_{K-1} critical value.

    Reports SWM-JK V_off + V_between as the SE plus a BCH 2011 fixed-cluster
    t-quantile rather than the asymptotic z=1.96. With K=5 folds the critical
    value goes from 1.960 to 2.776, widening the CI by ~42%. In the Davis-
    Kahan stress test this lifts the coverage column from 0.49 to 0.63 --
    still short of nominal but the gap then reflects SE under-estimation
    rather than critical-value misspecification. Use this as the robustness
    margin alongside the IM self-normalized headline.
    """
    from scipy import stats as _stats
    jk = salerno_jackknife_hac(scores, fold_ids, coords,
                                bandwidth_quantile=bandwidth_quantile,
                                bandwidth=bandwidth)
    K = int(fold_ids.max()) + 1
    tcrit = float(_stats.t.ppf(1.0 - alpha / 2.0, df=max(K - 1, 1)))
    theta = jk['theta_hat']
    se = jk['se']
    jk.update({
        't_critical': tcrit,
        'ci_low': theta - tcrit * se,
        'ci_high': theta + tcrit * se,
        'critical_value_source': 'BCH-2011-t_{K-1}',
    })
    return jk


__all__ = [
    "spatial_hac_se",
    "salerno_jackknife_hac",
    "buffered_kfold",
    "bandwidth_sensitivity_sweep",
    "ibragimov_muller_self_normalized_ci",
    "salerno_jackknife_t_critical",
]
