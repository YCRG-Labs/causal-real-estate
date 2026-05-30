"""Davis-Kahan inference laboratory.

Tests eight candidate SE / CI methods on the same DGP that davis_kahan_sim.py
uses for the JBES coverage exhibit, plus the existing four
(classical / HC1 / SWM-JK as currently coded in davis_kahan_sim.py /
SWM-JK as actually formulated in spatial_se.py).

The new methods evaluated:

  (i)    iid t-pivot bootstrap (Lam 2022) -- foil
  (ii)   Conley HAC at Lehner 2026 data-driven bandwidth
         (smallest h s.t. empirical covariogram crosses zero)
  (iii)  Salerno JK HAC with bandwidth sweep, taking the max over
         q in {0.05, 0.10, 0.15, 0.20, 0.25} (the SWM 2026
         "robust report" recipe)
  (iv)   Bester-Conley-Hansen cluster-covariance with K=5 clusters = folds
  (v)    Cameron-Gelbach-Miller wild cluster bootstrap with K=5 = folds
  (vi)   Adam-Müller self-normalized statistic
  (vii)  Politis-Romano stationary block bootstrap, l = sqrt(n)
  (viii) Pötscher-Preinerstorfer fixed-b inference at b = 0.5

Plus a *correct* re-implementation of Salerno-Wu-McCormick 2026 jackknife-HAC
using the V_jk = V_off + V_between decomposition from spatial_se.py applied at
the OLS level (the davis_kahan_sim version computes only V_off).

Runs the full 15-spike grid x 500 reps in parallel.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy import stats
from sklearn.gaussian_process.kernels import Matern

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))   # scripts/
sys.path.insert(0, str(_HERE))          # simulation/

# Reuse the canonical DGP and estimator
from davis_kahan_sim import (
    DGPConfig, make_spatial_dgp, estimate_residual_subspace,
    conley_spatial_hac_se,
)


# ---------------------------------------------------------------------------
# Helpers shared across methods
# ---------------------------------------------------------------------------

def _pairwise_dist(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def _bartlett(d: np.ndarray, h: float) -> np.ndarray:
    return np.maximum(1.0 - d / h, 0.0)


def _hac_var_score(
    X: np.ndarray, e: np.ndarray, coords: np.ndarray, h: float,
    fold_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Bartlett-kernel HAC of S = X * e, returning the (p,p) meat."""
    e2 = e.astype(np.float64).copy()
    if fold_idx is not None:
        for f in np.unique(fold_idx):
            sel = fold_idx == f
            e2[sel] -= e2[sel].mean()
    dist = _pairwise_dist(coords)
    W = _bartlett(dist, h)
    S = X * e2[:, None]
    return S.T @ W @ S


# ---------------------------------------------------------------------------
# Method (1)-(3): existing baselines reproduced here for one-stop coverage
# ---------------------------------------------------------------------------

def se_classical(rs: dict) -> float:
    return float(rs["se_classical"])


def se_hc1(rs: dict) -> float:
    return float(rs["se"])  # rs["se"] is HC1 in davis_kahan_sim.estimate_residual_subspace


def se_conley_hac_swm_centered(
    rs: dict, coords: np.ndarray, fold_idx: np.ndarray, h: float = 0.15,
) -> float:
    """The SE that davis_kahan_sim.py currently uses. Fold-centered residual
    HAC meat, then bread sandwich. NO V_between term. Reported as the
    existing column for comparison."""
    return conley_spatial_hac_se(
        rs["X"], rs["resid"], coords, bandwidth=h,
        kernel="bartlett", coef_idx=1, fold_idx=fold_idx,
    )


# ---------------------------------------------------------------------------
# Method (FIX): SWM 2026 jackknife with V_off + V_between, applied to OLS
# ---------------------------------------------------------------------------

def se_swm_jk_correct(
    rs: dict, coords: np.ndarray, fold_idx: np.ndarray, h: float = 0.15,
) -> tuple[float, float, float]:
    """Salerno-Wu-McCormick 2026 V_jk = V_off + V_between at the OLS level.

    Works on the OLS score s_i = (XtX)^{-1} X_i e_i for the slope coef.
    Equivalent to applying the score-mean SWM formula in spatial_se.py to the
    influence function psi_i = (XtX)^{-1}[1,1] * X_i[1] * e_i and adding the
    fold-mean variance from V_between.

    Returns (se, v_off, v_between).
    """
    X = rs["X"]
    e = rs["resid"].astype(np.float64).copy()
    XtX_inv = rs["XtX_inv"]
    n, p = X.shape
    # Influence function for the slope coefficient (coef_idx=1)
    # psi_i = n * (XtX)^{-1} X_i e_i, slope component.
    # Note theta_hat - theta = (1/n) sum_i psi_i so the variance of the mean
    # of psi equals Var(theta_hat). We follow the score-on-mean SWM exactly.
    psi = n * (XtX_inv @ (X * e[:, None]).T)[1, :]      # (n,)
    K = int(fold_idx.max()) + 1
    theta_hat = float(psi.mean())
    fold_means: dict[int, tuple[float, int]] = {}
    psi_tilde = psi.copy()
    for k in range(K):
        m = fold_idx == k
        if not m.any():
            continue
        fk = float(psi[m].mean())
        fold_means[k] = (fk, int(m.sum()))
        psi_tilde[m] = psi[m] - fk
    # V_off: Bartlett HAC of psi_tilde, off-diagonal only.
    dist = _pairwise_dist(coords)
    W = _bartlett(dist, h)
    # Self-pairs i=j are the diagonal of W (=1); subtract them to get V_off.
    v_within = float(psi_tilde @ W @ psi_tilde) / n ** 2
    v_diag = float(psi_tilde @ psi_tilde) / n ** 2
    v_off = v_within - v_diag
    # V_between
    v_between = 0.0
    for _, (fk, nk) in fold_means.items():
        v_between += (nk / n) ** 2 * (fk - theta_hat) ** 2
    if K > 1:
        v_between *= K / (K - 1)
    v_jk = max(v_off + v_between, 1e-300)
    return float(np.sqrt(v_jk)), float(v_off), float(v_between)


# ---------------------------------------------------------------------------
# Method (ii): Conley HAC at Lehner 2026 data-driven bandwidth
# ---------------------------------------------------------------------------

def lehner_bandwidth(
    e: np.ndarray, coords: np.ndarray, q_grid: tuple = (0.02, 0.04, 0.06, 0.08,
                                                       0.10, 0.12, 0.15, 0.20, 0.25),
    fold_idx: np.ndarray | None = None,
) -> float:
    """Smallest distance quantile h_q at which the empirical covariogram of
    (optionally fold-centered) residuals crosses zero.

    For each h in q_grid we compute mean(e_i e_j over pairs within h) and pick
    the first h at which it drops below 0 (or the gridmax if not).

    This is the Lehner 2026 idea reformulated: rather than minimising a CV
    proxy, find the lag at which spatial dependence has decayed to noise.
    """
    e2 = e.astype(np.float64).copy()
    if fold_idx is not None:
        for f in np.unique(fold_idx):
            sel = fold_idx == f
            e2[sel] -= e2[sel].mean()
    n = coords.shape[0]
    dist = _pairwise_dist(coords)
    iu, ju = np.triu_indices(n, k=1)
    d_pairs = dist[iu, ju]
    prod = e2[iu] * e2[ju]
    qs = np.quantile(d_pairs, list(q_grid))
    h_chosen = qs[-1]                   # fallback to max
    last_q = qs[-1]
    for h in qs:
        mask = d_pairs <= h
        if mask.sum() < 50:
            continue
        cov = float(prod[mask].mean())
        if cov <= 0:
            return float(h)
        last_q = float(h)
    return float(last_q)


def se_conley_lehner(
    rs: dict, coords: np.ndarray, fold_idx: np.ndarray,
) -> tuple[float, float]:
    """Conley HAC with data-driven bandwidth from Lehner 2026 logic."""
    h = lehner_bandwidth(rs["resid"], coords, fold_idx=fold_idx)
    se = conley_spatial_hac_se(
        rs["X"], rs["resid"], coords, bandwidth=h,
        kernel="bartlett", coef_idx=1, fold_idx=fold_idx,
    )
    return float(se), float(h)


# ---------------------------------------------------------------------------
# Method (iii): SWM-correct with bandwidth sweep max
# ---------------------------------------------------------------------------

def se_swm_jk_sweep_max(
    rs: dict, coords: np.ndarray, fold_idx: np.ndarray,
    q_grid: tuple = (0.05, 0.10, 0.15, 0.20, 0.25),
) -> tuple[float, float]:
    """SWM-correct (V_off + V_between) maximised over the bandwidth quantile grid."""
    n = coords.shape[0]
    dist = _pairwise_dist(coords)
    iu, ju = np.triu_indices(n, k=1)
    d_pairs = dist[iu, ju]
    hs = np.quantile(d_pairs, list(q_grid))
    ses = []
    for h in hs:
        s, _, _ = se_swm_jk_correct(rs, coords, fold_idx, h=float(h))
        ses.append(s)
    return float(max(ses)), float(hs[int(np.argmax(ses))])


# ---------------------------------------------------------------------------
# Method (iv): Bester-Conley-Hansen 2011 cluster covariance, K = folds
# ---------------------------------------------------------------------------

def se_bch_cluster(rs: dict, fold_idx: np.ndarray) -> tuple[float, float]:
    """Cluster-covariance SE treating each fold as a cluster.

    V = (XtX)^{-1} [Sum_g X_g' e_g e_g' X_g] (XtX)^{-1} with the standard
    G/(G-1) * (n-1)/(n-p) finite-sample correction (Cameron-Miller 2015).
    Returns (se, t-quantile-at-df=G-1) for Bester-Conley-Hansen t-asymptotics.
    """
    X = rs["X"]; e = rs["resid"]; XtX_inv = rs["XtX_inv"]
    n, p = X.shape
    K = int(fold_idx.max()) + 1
    meat = np.zeros((p, p))
    for k in range(K):
        m = fold_idx == k
        if not m.any():
            continue
        Xk = X[m]; ek = e[m]
        s_g = (Xk * ek[:, None]).sum(axis=0)         # (p,)
        meat += np.outer(s_g, s_g)
    corr = (K / max(K - 1, 1)) * ((n - 1) / max(n - p, 1))
    V = XtX_inv @ meat @ XtX_inv * corr
    se = float(np.sqrt(max(V[1, 1], 1e-300)))
    return se, float(stats.t.ppf(0.975, df=max(K - 1, 1)))


# ---------------------------------------------------------------------------
# Method (v): Cameron-Gelbach-Miller 2008 wild cluster bootstrap
# ---------------------------------------------------------------------------

def se_wild_cluster_bootstrap(
    rs: dict, fold_idx: np.ndarray, rng: np.random.Generator, B: int = 199,
) -> tuple[float, float, float, float]:
    """Wild cluster bootstrap (Rademacher) at K = folds clusters.

    Imposes H_0 : beta_1 = beta_hat (no null imposition; standard percentile-t
    via WCB-CRVE is non-trivial -- we use the conservative WCB-SE form for the
    SE column, plus a percentile-CI from the bootstrap distribution of theta).
    Returns (se, ci_lo, ci_hi, t_quant)."""
    X = rs["X"]; e = rs["resid"]; XtX_inv = rs["XtX_inv"]
    n, p = X.shape
    K = int(fold_idx.max()) + 1
    theta_hat = float(rs["theta"])
    boots = np.empty(B)
    for b in range(B):
        # Rademacher weights at the cluster level
        w_cluster = rng.choice([-1.0, 1.0], size=K)
        w = w_cluster[fold_idx.astype(int)]
        e_star = e * w
        # WCB-CRVE: theta_star = theta_hat + (XtX)^{-1} X' e_star
        delta = XtX_inv @ X.T @ e_star
        boots[b] = theta_hat + float(delta[1])
    se = float(np.std(boots, ddof=1))
    ci_lo = float(np.quantile(boots, 0.025))
    ci_hi = float(np.quantile(boots, 0.975))
    return se, ci_lo, ci_hi, 1.96


# ---------------------------------------------------------------------------
# Method (vi): Adam-Müller 2021 self-normalized statistic
# ---------------------------------------------------------------------------

def se_self_normalized(
    rs: dict, fold_idx: np.ndarray,
) -> tuple[float, float, float]:
    """Self-normalized inference following the Adam-Müller / Müller 2014 family.

    Split residual scores by fold (K groups). Construct
        bar_psi_k = mean of psi over fold k,
        bar_psi   = mean over all folds.
    The test statistic t = sqrt(K) * (bar_psi - 0) / sqrt(1/K-1 sum (bar_psi_k - bar_psi)^2)
    follows t_{K-1} under H_0 in the K-fixed asymptotics of Ibragimov-Müller 2010.

    Adapted here: theta_hat is the OLS estimate; we recompute it within each
    fold and pool with the cluster-t formula. Returns (se_implied, ci_lo, ci_hi).
    """
    X = rs["X"]; Y_res = rs["Y_res"]
    K = int(fold_idx.max()) + 1
    theta_k = []
    for k in range(K):
        m = fold_idx == k
        if m.sum() < 3:
            continue
        Xk = X[m]; Yk = Y_res[m]
        try:
            bk = np.linalg.solve(Xk.T @ Xk, Xk.T @ Yk)
        except np.linalg.LinAlgError:
            continue
        theta_k.append(float(bk[1]))
    theta_k = np.array(theta_k)
    G = len(theta_k)
    if G < 2:
        return float("nan"), float("nan"), float("nan")
    bar = float(theta_k.mean())
    s2 = float(theta_k.var(ddof=1))
    se_im = float(np.sqrt(s2 / G))
    tq = float(stats.t.ppf(0.975, df=G - 1))
    return se_im, bar - tq * se_im, bar + tq * se_im


# ---------------------------------------------------------------------------
# Method (vii): stationary block bootstrap (Politis-Romano 1994), spatial block
# ---------------------------------------------------------------------------

def se_block_bootstrap(
    rs: dict, coords: np.ndarray, rng: np.random.Generator, B: int = 199,
    block_radius: float | None = None,
) -> tuple[float, float, float]:
    """Spatial block bootstrap: resample contiguous spatial blocks of radius r.

    For each replicate, sample n centers uniformly at random, then take the
    nearest-neighbor union under r = block_radius (or 1/sqrt(n) by default).

    Reweight residuals: e_star = e[idx]; X_star = X[idx]; theta_star =
    (X_star'X_star)^{-1} X_star' y_star where y_star = X_star b_hat + e_star.
    """
    X = rs["X"]; e = rs["resid"]; XtX_inv = rs["XtX_inv"]
    n = X.shape[0]
    theta_hat = float(rs["theta"])
    if block_radius is None:
        block_radius = 1.0 / np.sqrt(n)
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    # Precompute neighbor list once
    neighbors = tree.query_ball_point(coords, r=block_radius)
    nbr_arrays = [np.asarray(nb, dtype=np.int64) for nb in neighbors]
    boots = np.empty(B)
    for b in range(B):
        centers = rng.integers(0, n, size=n)
        # vectorised draw: for each center pick one neighbor uniformly
        idx_list = np.empty(n, dtype=np.int64)
        for k, c in enumerate(centers):
            arr = nbr_arrays[c]
            if arr.size == 0:
                idx_list[k] = c
            else:
                idx_list[k] = arr[rng.integers(0, arr.size)]
        idx = idx_list
        Xs = X[idx]; es = e[idx]
        try:
            delta = np.linalg.solve(Xs.T @ Xs, Xs.T @ es)
        except np.linalg.LinAlgError:
            boots[b] = np.nan
            continue
        boots[b] = theta_hat + float(delta[1])
    se = float(np.nanstd(boots, ddof=1))
    ci_lo = float(np.nanquantile(boots, 0.025))
    ci_hi = float(np.nanquantile(boots, 0.975))
    return se, ci_lo, ci_hi


# ---------------------------------------------------------------------------
# Method (viii): Pötscher-Preinerstorfer 2020 fixed-b inference
# ---------------------------------------------------------------------------

def se_fixed_b(
    rs: dict, coords: np.ndarray, fold_idx: np.ndarray, b_frac: float = 0.5,
) -> tuple[float, float]:
    """Fixed-b spatial HAC: bandwidth h = b * D_max with b = b_frac.

    Under Kiefer-Vogelsang 2005 / Pötscher-Preinerstorfer 2020, the t-statistic
    at fixed-b has a non-standard limiting distribution; we compute the SE at
    the b-rescaled bandwidth and report the SE with the Kiefer-Vogelsang
    asymptotic-correction critical value approximated by t-distribution with
    df = floor(1/b) - 1 (a workable approximation; see KV 2005 Table 1).
    """
    n = coords.shape[0]
    dist = _pairwise_dist(coords)
    iu, ju = np.triu_indices(n, k=1)
    D_max = float(np.quantile(dist[iu, ju], 0.95))
    h = b_frac * D_max
    se, _, _ = se_swm_jk_correct(rs, coords, fold_idx, h=h)
    df_approx = max(int(1.0 / b_frac) - 1, 1)
    return float(se), float(stats.t.ppf(0.975, df=df_approx))


# ---------------------------------------------------------------------------
# One replicate runner
# ---------------------------------------------------------------------------

def _one_rep(
    rep_seed: int, spike_strength: float, cfg: DGPConfig,
    coords_fixed: np.ndarray, v_dir_fixed: np.ndarray,
    hac_bw: float, run_block_boot: bool, run_wcb: bool,
    diag_correlation: bool,
) -> dict:
    rng = np.random.default_rng(rep_seed)
    draw = make_spatial_dgp(
        cfg.n, cfg.d, spike_strength, cfg.tau, cfg, rng,
        coords=coords_fixed, v_dir=v_dir_fixed,
    )
    T, Y, coords = draw["T"], draw["Y"], draw["coords"]
    rs = estimate_residual_subspace(
        T, Y, coords, length_scale=cfg.length_scale, nu=cfg.nu, ridge=1e-3,
        n_folds=5, rng=rng,
    )
    fold_idx = np.arange(cfg.n) % 5
    theta = float(rs["theta"])

    # ---- 1. Classical
    se_cl = se_classical(rs)
    # ---- 2. HC1
    se_h1 = se_hc1(rs)
    # ---- 3. SWM-JK as currently coded (V_off only, no V_between)
    se_swm_v_off_only = se_conley_hac_swm_centered(rs, coords, fold_idx, h=hac_bw)
    # ---- FIX: SWM-JK correct (V_off + V_between)
    se_swm_fix, v_off, v_between = se_swm_jk_correct(rs, coords, fold_idx, h=hac_bw)
    # ---- iii: SWM-JK sweep max
    se_swm_sweep, h_sweep = se_swm_jk_sweep_max(rs, coords, fold_idx)
    # ---- ii: Conley at Lehner bandwidth
    se_lehner, h_lehner = se_conley_lehner(rs, coords, fold_idx)
    # ---- iv: BCH cluster covariance
    se_bch, t_bch = se_bch_cluster(rs, fold_idx)
    # ---- vi: Adam-Müller self-normalized
    se_sn, ci_sn_lo, ci_sn_hi = se_self_normalized(rs, fold_idx)
    # ---- viii: fixed-b
    se_fb, t_fb = se_fixed_b(rs, coords, fold_idx, b_frac=0.5)

    out = {
        "spike_strength": spike_strength,
        "rep_seed": rep_seed,
        "theta_rs": theta,
        "tau_true": cfg.tau,
        "se_classical": se_cl,
        "se_hc1": se_h1,
        "se_swm_v_off_only": se_swm_v_off_only,
        "se_swm_fix": se_swm_fix,
        "v_off": v_off,
        "v_between": v_between,
        "se_swm_sweep": se_swm_sweep,
        "h_sweep": h_sweep,
        "se_conley_lehner": se_lehner,
        "h_lehner": h_lehner,
        "se_bch": se_bch,
        "t_bch": t_bch,
        "se_self_normalized": se_sn,
        "ci_sn_lo": ci_sn_lo,
        "ci_sn_hi": ci_sn_hi,
        "se_fixed_b": se_fb,
        "t_fixed_b": t_fb,
    }

    # ---- v: wild cluster bootstrap
    if run_wcb:
        se_wcb, ci_wcb_lo, ci_wcb_hi, _ = se_wild_cluster_bootstrap(
            rs, fold_idx, rng, B=99,
        )
        out["se_wcb"] = se_wcb
        out["ci_wcb_lo"] = ci_wcb_lo
        out["ci_wcb_hi"] = ci_wcb_hi
    else:
        out["se_wcb"] = np.nan
        out["ci_wcb_lo"] = np.nan
        out["ci_wcb_hi"] = np.nan

    # ---- vii: block bootstrap
    if run_block_boot:
        se_bb, ci_bb_lo, ci_bb_hi = se_block_bootstrap(rs, coords, rng, B=99)
        out["se_block_boot"] = se_bb
        out["ci_bb_lo"] = ci_bb_lo
        out["ci_bb_hi"] = ci_bb_hi
    else:
        out["se_block_boot"] = np.nan
        out["ci_bb_lo"] = np.nan
        out["ci_bb_hi"] = np.nan

    # ---- DGP diagnostics
    if diag_correlation:
        # empirical residual covariogram by distance bin
        e = rs["resid"]
        n = cfg.n
        dist = _pairwise_dist(coords)
        iu, ju = np.triu_indices(n, k=1)
        d_pairs = dist[iu, ju]
        prod = e[iu] * e[ju]
        # 6 distance bins on quantile grid
        edges = np.quantile(d_pairs, [0.0, 0.05, 0.10, 0.20, 0.40, 0.70, 1.0])
        bin_means = []
        for k in range(6):
            m = (d_pairs >= edges[k]) & (d_pairs < edges[k + 1])
            if m.any():
                bin_means.append(float(prod[m].mean()))
            else:
                bin_means.append(np.nan)
        for k, v in enumerate(bin_means):
            out[f"cov_bin_{k}"] = v
            out[f"cov_bin_edge_{k}"] = float(edges[k + 1])
        # within-fold vs across-fold residual product correlation
        same_fold = fold_idx[iu] == fold_idx[ju]
        within = float(prod[same_fold].mean()) if same_fold.any() else np.nan
        across = float(prod[~same_fold].mean()) if (~same_fold).any() else np.nan
        out["resid_corr_within_fold"] = within
        out["resid_corr_across_fold"] = across
        # fold means (cross-fold mean residual SE)
        fold_means = np.array([e[fold_idx == k].mean() for k in range(5)])
        out["cross_fold_mean_sd"] = float(fold_means.std(ddof=1))
        out["cross_fold_mean_max_abs"] = float(np.max(np.abs(fold_means)))

    return out


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------

def run_sweep(
    n_reps: int, spike_grid: np.ndarray, cfg: DGPConfig, out_dir: Path,
    n_jobs: int, hac_bw: float, seed: int,
    run_block_boot: bool, run_wcb: bool, diag_correlation: bool,
    rho_label: str | None = None,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    fixed_rng = np.random.default_rng(cfg.coord_seed)
    coords_fixed = fixed_rng.uniform(size=(cfg.n, 2))
    # sparse loading
    idx = fixed_rng.choice(cfg.d, size=cfg.sparse_k, replace=False)
    v = np.zeros(cfg.d)
    v[idx] = fixed_rng.standard_normal(cfg.sparse_k)
    v_dir_fixed = v / (np.linalg.norm(v) + 1e-12)

    rng_master = np.random.default_rng(seed)
    print(f"[lab] sweep: {len(spike_grid)} spikes x {n_reps} reps "
          f"= {len(spike_grid) * n_reps} jobs (rho={rho_label})", flush=True)
    t0 = time.time()
    all_rows: list[dict] = []
    with parallel_config(backend="loky", inner_max_num_threads=1):
        with Parallel(n_jobs=n_jobs, verbose=0) as parallel:
            for gi, lam in enumerate(spike_grid, start=1):
                seeds = rng_master.integers(0, 2**31 - 1, size=n_reps).tolist()
                ct0 = time.time()
                rows = parallel(
                    delayed(_one_rep)(
                        int(s), float(lam), cfg, coords_fixed, v_dir_fixed,
                        hac_bw, run_block_boot, run_wcb, diag_correlation,
                    )
                    for s in seeds
                )
                all_rows.extend(rows)
                print(f"  [{gi}/{len(spike_grid)}] lambda={lam:.3f} "
                      f"R={n_reps} dt={time.time() - ct0:.1f}s", flush=True)
    df = pd.DataFrame(all_rows)
    if rho_label is not None:
        df["rho"] = rho_label
        suffix = f"_rho{rho_label}"
    else:
        suffix = ""
    df.to_csv(out_dir / f"dk_lab_raw{suffix}.csv", index=False)
    print(f"[lab] wrote {out_dir / f'dk_lab_raw{suffix}.csv'} "
          f"total={time.time() - t0:.0f}s", flush=True)
    return df


def aggregate_coverage(df: pd.DataFrame, tau_true: float) -> pd.DataFrame:
    rows = []
    for lam, sub in df.groupby("spike_strength"):
        row = {"spike_strength": lam, "n_reps_ok": len(sub)}
        # Symmetric Z=1.96 intervals
        for nm in ["classical", "hc1", "swm_v_off_only", "swm_fix",
                   "swm_sweep", "conley_lehner", "fixed_b"]:
            se = sub[f"se_{nm}"]
            lo = sub["theta_rs"] - 1.96 * se
            hi = sub["theta_rs"] + 1.96 * se
            row[f"cov_{nm}"] = float(((lo <= tau_true) & (tau_true <= hi)).mean())
            row[f"mean_se_{nm}"] = float(se.mean())
        # BCH: uses t with df=K-1=4
        tcrit_bch = float(stats.t.ppf(0.975, df=4))
        lo = sub["theta_rs"] - tcrit_bch * sub["se_bch"]
        hi = sub["theta_rs"] + tcrit_bch * sub["se_bch"]
        row["cov_bch_cluster"] = float(((lo <= tau_true) & (tau_true <= hi)).mean())
        row["mean_se_bch_cluster"] = float(sub["se_bch"].mean())
        # Self-normalized: stored ci_sn_lo/hi
        row["cov_self_normalized"] = float(
            ((sub["ci_sn_lo"] <= tau_true) & (tau_true <= sub["ci_sn_hi"])).mean()
        )
        row["mean_se_self_normalized"] = float(sub["se_self_normalized"].mean())
        # WCB
        if sub["se_wcb"].notna().any():
            row["cov_wcb"] = float(
                ((sub["ci_wcb_lo"] <= tau_true) & (tau_true <= sub["ci_wcb_hi"])).mean()
            )
            row["mean_se_wcb"] = float(sub["se_wcb"].mean())
        # Block bootstrap
        if sub["se_block_boot"].notna().any():
            row["cov_block_boot"] = float(
                ((sub["ci_bb_lo"] <= tau_true) & (tau_true <= sub["ci_bb_hi"])).mean()
            )
            row["mean_se_block_boot"] = float(sub["se_block_boot"].mean())
        # Mean v_off vs v_between
        row["mean_v_off"] = float(sub["v_off"].mean())
        row["mean_v_between"] = float(sub["v_between"].mean())
        row["mean_ratio_v_off_to_total"] = float(
            (sub["v_off"] / (sub["v_off"] + sub["v_between"] + 1e-30)).mean()
        )
        # Diagnostics
        for k in ["cov_bin_0", "cov_bin_1", "cov_bin_2", "cov_bin_3",
                  "cov_bin_4", "cov_bin_5", "resid_corr_within_fold",
                  "resid_corr_across_fold", "cross_fold_mean_sd",
                  "cross_fold_mean_max_abs"]:
            if k in sub.columns:
                row[f"mean_{k}"] = float(sub[k].mean())
        # Bias / SD of theta
        row["bias_theta"] = float((sub["theta_rs"] - tau_true).mean())
        row["sd_theta"] = float(sub["theta_rs"].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("spike_strength").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_reps", type=int, default=500)
    ap.add_argument("--n_spike", type=int, default=15)
    ap.add_argument("--spike_min", type=float, default=0.5)
    ap.add_argument("--spike_max", type=float, default=4.0)
    ap.add_argument("--length_scale", type=float, default=0.15)
    ap.add_argument("--nu", type=float, default=2.5)
    ap.add_argument("--n", type=int, default=348)
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--tau", type=float, default=0.10)
    ap.add_argument("--sigma_nu", type=float, default=0.30)
    ap.add_argument("--sparse_k", type=int, default=50)
    ap.add_argument("--hac_bw", type=float, default=0.15)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=20260529)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("results/simulation/davis_kahan_diag"))
    ap.add_argument("--no_block_boot", action="store_true")
    ap.add_argument("--no_wcb", action="store_true")
    ap.add_argument("--no_diag", action="store_true",
                    help="Skip per-rep correlation diagnostics")
    ap.add_argument("--rho_sweep", action="store_true",
                    help="Run rho in {0.05,0.10,0.15,0.25,0.40,0.60} at "
                         "a smaller spike grid")
    args = ap.parse_args()

    cfg = DGPConfig(
        n=args.n, d=args.d, tau=args.tau, length_scale=args.length_scale,
        nu=args.nu, sigma_nu=args.sigma_nu, sparse_k=args.sparse_k,
    )
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    spike_grid = np.linspace(args.spike_min, args.spike_max, args.n_spike)
    with open(out_dir / "config.json", "w") as f:
        json.dump({**asdict(cfg), "spike_grid": spike_grid.tolist(),
                   "n_reps": args.n_reps, "hac_bw": args.hac_bw,
                   "seed": args.seed}, f, indent=2)

    if args.rho_sweep:
        rho_list = [0.05, 0.10, 0.15, 0.25, 0.40, 0.60]
        # Use a smaller spike grid for the rho sweep
        spike_sub = np.linspace(args.spike_min, args.spike_max, 5)
        for rho in rho_list:
            cfg_rho = DGPConfig(
                n=args.n, d=args.d, tau=args.tau, length_scale=rho,
                nu=args.nu, sigma_nu=args.sigma_nu, sparse_k=args.sparse_k,
            )
            df = run_sweep(
                args.n_reps, spike_sub, cfg_rho, out_dir,
                n_jobs=args.n_jobs, hac_bw=rho,
                seed=args.seed,
                run_block_boot=not args.no_block_boot,
                run_wcb=not args.no_wcb,
                diag_correlation=not args.no_diag,
                rho_label=f"{rho:.2f}",
            )
            agg = aggregate_coverage(df, tau_true=cfg.tau)
            agg["rho"] = rho
            agg.to_csv(out_dir / f"dk_lab_summary_rho{rho:.2f}.csv", index=False)
        return

    df = run_sweep(
        args.n_reps, spike_grid, cfg, out_dir,
        n_jobs=args.n_jobs, hac_bw=args.hac_bw,
        seed=args.seed,
        run_block_boot=not args.no_block_boot,
        run_wcb=not args.no_wcb,
        diag_correlation=not args.no_diag,
    )
    agg = aggregate_coverage(df, tau_true=cfg.tau)
    agg.to_csv(out_dir / "dk_lab_summary.csv", index=False)
    print("Coverage summary:")
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)
    keep = ["spike_strength", "n_reps_ok", "cov_classical", "cov_hc1",
            "cov_swm_v_off_only", "cov_swm_fix", "cov_swm_sweep",
            "cov_conley_lehner", "cov_bch_cluster", "cov_wcb",
            "cov_self_normalized", "cov_block_boot", "cov_fixed_b",
            "mean_ratio_v_off_to_total"]
    print(agg[[c for c in keep if c in agg.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
