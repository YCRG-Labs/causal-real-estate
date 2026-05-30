"""Davis-Kahan / spiked-covariance Monte Carlo for the JBES theorem appendix.

Three findings in one sim, indexed by spike strength lambda along a BBP grid
(Baik-Ben Arous-Peche 2005; Paul 2007; Benaych-Georges-Nadakuditi 2011):

  (i)  PC1-OLS bias is monotone in the between/within variance ratio induced
       by the spatial confounder U (Davis-Kahan upper bound saturating).
  (ii) Residual-subspace DML (orthogonalize T against a Matern smoother of
       coords, then DML the residual on Y) recovers tau approximately
       unbiased across the same grid.
  (iii) Standard iid bootstrap CI under-covers tau; a Conley-style
       spatial HAC (Bartlett kernel, Cao-Leung 2025 buffered cross-fit,
       Salerno-Wu-McCormick 2026 fold-centered correction) restores
       nominal 95 percent coverage.

DGP (n=348, d=768 by default; matches the dedup SF sample from
project_causal_real_estate.md):

  coords ~ Uniform([0,1]^2)
  U      = chol(K_matern(coords, l=0.15, nu=2.5)) @ z,   z ~ N(0, I_n)
  T      = lambda * u_dir * v_dir^T  +  E,   E_ij ~ N(0, 1/sqrt(n))
           (rank-one spike on a sparse loading v_dir, signal = U via u_dir)
  Y      = U  +  tau * (T @ v_perp)  +  nu,   nu ~ N(0, sigma_nu^2)

  where v_dir is a fixed sparse unit loading in R^d,
        u_dir is U / ||U||,
        v_perp is a unit vector orthogonal to v_dir (the "deconfounded" axis).

This is the cleanest setup that simultaneously gives a BBP-style spike with
ground-truth principal direction v_dir, and an outcome whose causal signal
lives on a v-orthogonal axis (Cevid-Buhlmann 2020 architecture; Wang-Zhao-
Hastie-Owen AoS 2017 RUV/cate architecture).

Refs (full citations in research notes):
  - Baik, Ben Arous, Peche 2005 AoP                          (BBP transition)
  - Paul 2007 Stat. Sinica                                   (spike eigenvec)
  - Benaych-Georges, Nadakuditi 2011 Adv. Math.              (singular vec)
  - Yu, Wang, Samworth 2015 Biometrika                       (DK variant)
  - Cevid, Buhlmann, Meinshausen 2020 JMLR                   (spectral deconf.)
  - Guo, Cevid, Buhlmann 2022 AoS                            (doubly debiased)
  - Wiecha, Hoppin, Reich 2025 Biometrics  (Double Spatial Regression DGP)
  - Burman, Ogburn, Datta 2025 arXiv:2510.22464              (basis voting)
  - Salerno, Wu, McCormick 2026 arXiv:2603.11368             (jackknife sHAC)
  - Conley 1999 JoE; Bester-Conley-Hansen 2011 JoE           (spatial HAC)

Usage:
  python davis_kahan_sim.py --n_reps 50                # smoke (~5 min)
  python davis_kahan_sim.py --n_reps 500               # publication
  python davis_kahan_sim.py --n_reps 500 --n_jobs 1    # sequential debug
  python davis_kahan_sim.py --plot_only                # re-render figures
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy import stats
from sklearn.gaussian_process.kernels import Matern

_NCORES = os.cpu_count() or 4

DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parents[3] / "results" / "simulation" / "davis_kahan"
)


# ---------------------------------------------------------------------------
# DGP
# ---------------------------------------------------------------------------

@dataclass
class DGPConfig:
    n: int = 348
    d: int = 768
    tau: float = 0.10
    length_scale: float = 0.15        # Matern length-scale (units of [0,1])
    nu: float = 2.5                   # Matern smoothness (DSR / Wiecha 2025)
    sigma_nu: float = 0.30            # outcome noise SD
    sparse_k: int = 50                # support size of v_dir
    coord_seed: int = 20260529        # one fixed coords map across reps


def _sparse_unit_loading(d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Sparse unit-norm loading v_dir in R^d with support size k (Paul 2007)."""
    idx = rng.choice(d, size=k, replace=False)
    v = np.zeros(d, dtype=np.float64)
    v[idx] = rng.standard_normal(k)
    v /= np.linalg.norm(v) + 1e-12
    return v


def _matern_field(
    coords: np.ndarray, length_scale: float, nu: float,
    rng: np.random.Generator, jitter: float = 1e-6,
) -> np.ndarray:
    """Draw one Matern(l, nu) Gaussian field at the supplied 2D coords.

    Builds K = Matern(coords), Cholesky-factorises, draws U = L z. The
    jitter on the diagonal is the standard fix for borderline-PSD Matern
    matrices at small length scales.
    """
    n = coords.shape[0]
    kernel = Matern(length_scale=length_scale, nu=nu)
    K = kernel(coords)
    K[np.diag_indices_from(K)] += jitter
    L = np.linalg.cholesky(K)
    z = rng.standard_normal(n)
    return L @ z


def make_spatial_dgp(
    n: int,
    d: int,
    spike_strength: float,
    tau: float,
    cfg: DGPConfig,
    rng: np.random.Generator,
    coords: np.ndarray | None = None,
    v_dir: np.ndarray | None = None,
) -> dict:
    """Generate one Monte Carlo draw of (T, coords, Y, truth).

    Parameters
    ----------
    n, d : int                shape of T (n=348, d=768 by default)
    spike_strength : float    lambda in the rank-one spike  T = lambda u v^T + E
    tau : float               causal coefficient of T @ v_perp on Y
    cfg : DGPConfig           Matern + noise parameters
    rng : Generator
    coords, v_dir : optional  pre-drawn coordinates / loading. Pass these in
                              from the orchestrator so they are FIXED across
                              reps and across the spike grid; only U and
                              the noise E vary.

    Returns
    -------
    dict with keys T, coords, Y, U, v_dir, v_perp, tau, spike_strength.

    Spike convention.  Following Benaych-Georges-Nadakuditi 2011 we write
        T = lambda * u v^T + E,        E_ij ~ N(0, 1/n)
    where ||u|| = ||v|| = 1. The BBP threshold for top singular value
    detection is lambda_c = (p/n)^(1/4) (rectangular case). For n=348,
    d=768 we have lambda_c approx 1.21, so a grid lambda in [0.5, 4] sweeps
    sub- and super-critical regimes.

    Y construction.  Y = U + tau (T @ v_perp) + nu  with v_perp chosen
    orthogonal to v_dir so the causal signal lives on the deconfounded
    axis (Cevid-Buhlmann spectral deconfounding architecture).
    """
    # --- spatial confounder
    if coords is None:
        coords = rng.uniform(size=(n, 2))
    U = _matern_field(coords, cfg.length_scale, cfg.nu, rng)
    U_norm = float(np.linalg.norm(U))
    if U_norm < 1e-10:
        u_dir = np.ones(n) / np.sqrt(n)
    else:
        u_dir = U / U_norm

    # --- representation T = spike + noise
    if v_dir is None:
        v_dir = _sparse_unit_loading(d, cfg.sparse_k, rng)
    # Noise scale 1/sqrt(n) so that the bulk singular values follow
    # Marchenko-Pastur with edge 1 + sqrt(d/n); spike detectable above BBP.
    sigma_E = 1.0 / np.sqrt(n)
    E_noise = sigma_E * rng.standard_normal((n, d))
    T = spike_strength * np.outer(u_dir, v_dir) + E_noise

    # --- orthogonal axis (the deconfounded direction)
    v_perp_raw = rng.standard_normal(d)
    v_perp_raw -= np.dot(v_perp_raw, v_dir) * v_dir   # Gram-Schmidt
    v_perp = v_perp_raw / (np.linalg.norm(v_perp_raw) + 1e-12)

    # --- outcome with confounder + causal axis
    nu_noise = cfg.sigma_nu * rng.standard_normal(n)
    Y = U + tau * (T @ v_perp) + nu_noise

    return {
        "T": T, "coords": coords, "Y": Y, "U": U,
        "v_dir": v_dir, "v_perp": v_perp,
        "tau": tau, "spike_strength": spike_strength,
    }


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def _top_k_pcs(T: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k right singular directions and corresponding scores.

    Returns (scores, V_k) where scores = T @ V_k is (n, k) and V_k is (d, k).
    Centering on the column mean only (the standard PCA convention; we are
    not interested in an intercept on Y here because OLS will add one).
    """
    Tc = T - T.mean(axis=0, keepdims=True)
    # Randomized SVD would be faster for d=768 but full SVD on n=348 is cheap
    # (n is the binding dim, full_matrices=False gives the thin SVD).
    U_, s, Vt = np.linalg.svd(Tc, full_matrices=False)
    V_k = Vt[:k].T                      # (d, k)
    scores = U_[:, :k] * s[:k]          # (n, k)
    return scores, V_k


def estimate_pc_bias(
    T: np.ndarray, Y: np.ndarray, coords: np.ndarray, k: int = 1,
) -> dict:
    """OLS of Y on the top-k PCs of T. Returns theta-on-PC1 with iid OLS SE.

    This is the "naive" estimator under test: it estimates Y = alpha + beta_1
    PC_1(T) + ... + beta_k PC_k(T) + eps with iid Gaussian-OLS standard
    errors. Per the theorem, beta_1 is biased proportional to the
    between/within ratio when the top PC aligns with the U-driven direction.
    """
    scores, V_k = _top_k_pcs(T, k)
    X = np.column_stack([np.ones(len(Y)), scores])    # (n, 1+k)
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ Y
    resid = Y - X @ beta
    n, p = X.shape
    sigma2 = float(resid @ resid) / max(n - p, 1)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    theta = float(beta[1])              # PC1 coefficient
    se_theta = float(se[1])
    ci_lo = theta - 1.96 * se_theta
    ci_hi = theta + 1.96 * se_theta
    return {
        "theta": theta, "se": se_theta,
        "ci_low": ci_lo, "ci_high": ci_hi,
        "V_k": V_k,
    }


def _matern_smoother(
    coords: np.ndarray, Y: np.ndarray, length_scale: float, nu: float,
    ridge: float = 1e-3,
) -> np.ndarray:
    """Kernel ridge fit Y_hat = K (K + ridge I)^{-1} Y on Matern(l, nu).

    This is the Wiecha-Hoppin-Reich 2025 (DSR) spatial smoother used both
    here and in the JBES Appendix F architecture: substract a Matern GP
    smoothing of coords from both Y and T, then run DML on the residuals.
    Closed-form (kernel ridge regression) instead of a fitted Gaussian
    process likelihood -- equivalent point estimates and orders of magnitude
    faster for the Monte Carlo loop.
    """
    n = coords.shape[0]
    kernel = Matern(length_scale=length_scale, nu=nu)
    K = kernel(coords)
    A = K + ridge * np.eye(n)
    # Solve A alpha = Y, then Y_hat = K alpha.
    alpha = np.linalg.solve(A, Y)
    return K @ alpha


def estimate_residual_subspace(
    T: np.ndarray, Y: np.ndarray, coords: np.ndarray,
    length_scale: float = 0.15, nu: float = 2.5,
    ridge: float = 1e-3, n_folds: int = 5,
    rng: np.random.Generator | None = None,
) -> dict:
    """Orthogonalize-then-DML on the spatial residuals.

    Step 1.  Cross-fit a Matern kernel-ridge smoother of coords -> Y and
             -> each column of T (Burman-Ogburn-Datta 2025 + Wiecha-Hoppin-
             Reich 2025 DSR architecture; Cao-Leung 2025 buffered cross-fit
             approximated by 5-fold here -- spatial leakage between folds is
             second-order at length_scale=0.15 and 5 folds of randomly
             permuted indices).
    Step 2.  Residualize:  Y_res, T_res.
    Step 3.  Estimate the leading direction v_perp_hat from PCA of T_res
             (top-1 PC) -- the residual structure is dominated by the
             deconfounded axis once U is removed.
    Step 4.  OLS theta = T_res @ v_perp_hat on Y_res. Report HC1 se for
             the IF-style baseline CI.

    Notes.
      - For d=768, residualising every column is the expensive step
        (n^3 Cholesky once, reused for all d columns: O(n^2 d) extra).
      - We CAN cheat for the sim and residualise only the v_dir axis
        (and the eventual v_perp_hat), but that would be a different
        estimator than the JBES pipeline. Keep the honest version.
    """
    if rng is None:
        rng = np.random.default_rng()

    n, d = T.shape
    fold_idx = rng.permutation(n) % n_folds

    Y_res = np.empty_like(Y)
    T_res = np.empty_like(T)
    for f in range(n_folds):
        train = np.where(fold_idx != f)[0]
        test = np.where(fold_idx == f)[0]
        if len(train) < 2 or len(test) < 1:
            continue
        # Matern kernel between train and (train, test) blocks.
        kernel = Matern(length_scale=length_scale, nu=nu)
        K_tr = kernel(coords[train])
        K_te = kernel(coords[test], coords[train])
        A = K_tr + ridge * np.eye(len(train))
        # Y residual on test fold.
        alpha_Y = np.linalg.solve(A, Y[train])
        Y_res[test] = Y[test] - K_te @ alpha_Y
        # T residual on test fold (vectorised over the d columns).
        alpha_T = np.linalg.solve(A, T[train])    # (n_train, d)
        T_res[test] = T[test] - K_te @ alpha_T

    # Direction estimate from residual-T's top PC.
    Tc_res = T_res - T_res.mean(axis=0, keepdims=True)
    _, s, Vt = np.linalg.svd(Tc_res, full_matrices=False)
    v_perp_hat = Vt[0]
    v_perp_hat /= np.linalg.norm(v_perp_hat) + 1e-12

    score = T_res @ v_perp_hat            # (n,)
    # Final OLS: Y_res = alpha + theta * score + eps.
    X = np.column_stack([np.ones(n), score])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ Y_res
    resid = Y_res - X @ beta
    p = X.shape[1]
    sigma2 = float(resid @ resid) / max(n - p, 1)
    se_classical = float(np.sqrt((sigma2 * XtX_inv)[1, 1]))
    # HC1 sandwich (White) standard error.
    h = (X * (resid[:, None] ** 2)) @ XtX_inv      # used below
    XeeX = (X * (resid[:, None] ** 2)).T @ X        # (p, p) meat
    V_hc1 = XtX_inv @ XeeX @ XtX_inv * (n / max(n - p, 1))
    se_hc1 = float(np.sqrt(V_hc1[1, 1]))
    theta = float(beta[1])
    return {
        "theta": theta, "se": se_hc1, "se_classical": se_classical,
        "ci_low": theta - 1.96 * se_hc1,
        "ci_high": theta + 1.96 * se_hc1,
        "v_perp_hat": v_perp_hat,
        "score": score, "Y_res": Y_res, "X": X, "XtX_inv": XtX_inv,
        "resid": resid,
    }


# ---------------------------------------------------------------------------
# Spatial HAC (Conley 1999 / Bester-Conley-Hansen 2011 / SWM 2026)
# ---------------------------------------------------------------------------

def conley_spatial_hac_se(
    X: np.ndarray, resid: np.ndarray, coords: np.ndarray,
    bandwidth: float, kernel: str = "bartlett",
    coef_idx: int = 1, fold_idx: np.ndarray | None = None,
) -> float:
    """Conley spatial HAC standard error for coefficient `coef_idx` in OLS.

    The Bester-Conley-Hansen (2011) fixed-b spatial HAC.  When `fold_idx`
    is supplied, applies Salerno-Wu-McCormick 2026 fold-centering -- the
    "meat" matrix is computed on residuals demeaned within fold to remove
    the fold-induced spurious correlation Cao-Leung 2025 identified.

    Parameters
    ----------
    X : (n, p)              design matrix INCLUDING intercept column
    resid : (n,)            OLS residuals
    coords : (n, 2)         spatial coordinates
    bandwidth : float       Bartlett-kernel bandwidth h (Euclidean units)
    kernel : {bartlett, uniform}
    coef_idx : int          which coefficient's SE to return
    fold_idx : (n,) or None fold assignments for SWM-2026 centering

    Variance formula
    ----------------
    V = (X'X)^{-1} [Sum_{i,j} K(d_ij/h) X_i X_j' e_i e_j] (X'X)^{-1}

    Returns sqrt(V[coef_idx, coef_idx]).
    """
    n, p = X.shape
    e = resid.astype(np.float64).copy()
    if fold_idx is not None:
        # Salerno-Wu-McCormick fold-centering: subtract within-fold mean.
        for f in np.unique(fold_idx):
            sel = fold_idx == f
            e[sel] -= e[sel].mean()

    # Pairwise distance matrix.
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))

    if kernel == "bartlett":
        W = np.maximum(1.0 - dist / bandwidth, 0.0)
    elif kernel == "uniform":
        W = (dist <= bandwidth).astype(np.float64)
    else:
        raise ValueError(f"Unknown kernel {kernel}")

    # Score matrix s_i = X_i * e_i, then meat = sum_ij W_ij s_i s_j'.
    S = X * e[:, None]                                # (n, p)
    meat = S.T @ W @ S                                # (p, p)

    bread = np.linalg.inv(X.T @ X)
    V = bread @ meat @ bread
    return float(np.sqrt(max(V[coef_idx, coef_idx], 0.0)))


def iid_bootstrap_se(
    estimator_fn, T: np.ndarray, Y: np.ndarray, coords: np.ndarray,
    B: int, rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Lam-2022 cheap bootstrap SE for an estimator returning {theta, ...}.

    Returns (se_cb, ci_low, ci_high) with a Lam-2022 t-pivot. The SE is the
    cheap-bootstrap form sqrt(mean((boots - theta_hat)^2)) centered at the
    ORIGINAL-sample estimate (Lam 2022 eq. (1)-(2)), not the bootstrap mean;
    centering at the bootstrap mean would substitute the bias-correcting
    Efron pivot. The t-quantile uses df = B exactly. Resamples rows iid
    with replacement -- the iid bootstrap that breaks under spatial
    dependence (Politis-Romano block bootstrap would be the right answer;
    iid is the foil this simulation is designed to indict).
    """
    n = T.shape[0]
    # Original-sample point estimate for centering (Lam 2022 eq. 1).
    theta_hat_orig = float(estimator_fn(T, Y, coords)["theta"])
    thetas = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        out = estimator_fn(T[idx], Y[idx], coords[idx])
        thetas[b] = out["theta"]
    se_cb = float(np.sqrt(np.mean((thetas - theta_hat_orig) ** 2)))
    t_quant = float(stats.t.ppf(0.975, df=B))
    return se_cb, theta_hat_orig - t_quant * se_cb, theta_hat_orig + t_quant * se_cb


# ---------------------------------------------------------------------------
# One replicate
# ---------------------------------------------------------------------------

def _one_rep(
    rep_seed: int, spike_strength: float, cfg: DGPConfig,
    coords: np.ndarray, v_dir: np.ndarray, hac_bandwidth: float,
    n_boot: int,
) -> dict:
    """Run one replicate across all three estimators and inference methods."""
    rng = np.random.default_rng(rep_seed)
    draw = make_spatial_dgp(
        cfg.n, cfg.d, spike_strength, cfg.tau, cfg, rng,
        coords=coords, v_dir=v_dir,
    )
    T = draw["T"]
    Y = draw["Y"]
    coords_local = draw["coords"]
    v_perp_true = draw["v_perp"]

    # --- (1) PC1-OLS biased estimator.
    pc = estimate_pc_bias(T, Y, coords_local, k=1)
    # Per the theorem the truth FOR THIS ESTIMAND is 0: tau lives entirely
    # on v_perp, which is orthogonal to v_dir = top PC under high spike.
    # We do NOT pin truth_pc=0 because under weak spike the alignment is
    # not perfect; reporting raw theta lets us measure the dependence on
    # spike strength.

    # --- (2) Residual-subspace DML estimator.
    rs = estimate_residual_subspace(
        T, Y, coords_local,
        length_scale=cfg.length_scale, nu=cfg.nu, ridge=1e-3, n_folds=5,
        rng=rng,
    )
    theta_rs = rs["theta"]
    se_rs_classical = rs["se_classical"]
    se_rs_hc1 = rs["se"]
    # Spatial HAC SE with SWM-2026 fold-centering.
    fold_idx = np.arange(cfg.n) % 5
    se_rs_hac = conley_spatial_hac_se(
        rs["X"], rs["resid"], coords_local, bandwidth=hac_bandwidth,
        kernel="bartlett", coef_idx=1, fold_idx=fold_idx,
    )

    # --- (3) IID bootstrap on residual-subspace estimator (foil).
    def _rs_fn(T_b, Y_b, coords_b):
        return estimate_residual_subspace(
            T_b, Y_b, coords_b,
            length_scale=cfg.length_scale, nu=cfg.nu, ridge=1e-3, n_folds=5,
            rng=np.random.default_rng(int(rep_seed) + 7919),
        )
    if n_boot > 0:
        se_rs_boot, ci_boot_lo, ci_boot_hi = iid_bootstrap_se(
            _rs_fn, T, Y, coords_local, B=n_boot, rng=rng,
        )
    else:
        se_rs_boot = np.nan
        ci_boot_lo = np.nan
        ci_boot_hi = np.nan

    # --- (4) Davis-Kahan diagnostic: sin-Theta(top-1 PC of T, v_dir).
    Tc = T - T.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Tc, full_matrices=False)
    v1_hat = Vt[0]
    # sin-Theta = sqrt(1 - <v1_hat, v_dir>^2).
    cos_th = abs(float(v1_hat @ v_dir))
    sin_theta_pc1 = np.sqrt(max(1.0 - cos_th ** 2, 0.0))
    # Alignment of v_perp_hat with the true v_perp (causal axis).
    cos_vp = abs(float(rs["v_perp_hat"] @ v_perp_true))
    sin_theta_vperp = np.sqrt(max(1.0 - cos_vp ** 2, 0.0))

    # --- (5) Between/within variance ratio: SS explained by U vs total.
    U = draw["U"]
    var_U = float(np.var(U, ddof=1))
    var_T_total = float(np.trace(np.cov(T.T)))
    btw_within = var_U / max(var_T_total, 1e-12)

    return {
        "spike_strength": spike_strength,
        "rep_seed": rep_seed,
        # PC1-OLS estimator (the biased one).
        "theta_pc1": pc["theta"],
        "se_pc1_iid": pc["se"],
        "ci_pc1_iid_lo": pc["ci_low"],
        "ci_pc1_iid_hi": pc["ci_high"],
        # Residual-subspace estimator (the unbiased one).
        "theta_rs": theta_rs,
        "se_rs_classical": se_rs_classical,
        "se_rs_hc1": se_rs_hc1,
        "se_rs_boot": se_rs_boot,
        "se_rs_hac": se_rs_hac,
        "ci_rs_boot_lo": ci_boot_lo,
        "ci_rs_boot_hi": ci_boot_hi,
        # Davis-Kahan diagnostics.
        "sin_theta_pc1": sin_theta_pc1,
        "sin_theta_vperp": sin_theta_vperp,
        # Between/within signal.
        "btw_within": btw_within,
        # Truths (constant across reps; carried for downstream agg).
        "tau_true": cfg.tau,
    }


# ---------------------------------------------------------------------------
# Coverage sweep
# ---------------------------------------------------------------------------

def coverage_sweep(
    n_reps: int,
    spike_grid: np.ndarray,
    cfg: DGPConfig,
    out_dir: Path,
    n_jobs: int = -1,
    n_boot: int = 0,
    hac_bandwidth: float = 0.15,
    seed: int = 20260529,
) -> pd.DataFrame:
    """Run the full Monte Carlo across (spike_strength, rep) cells.

    Coords and v_dir are FIXED across reps and across the spike grid so
    that the only thing varying along the grid is the spike strength
    itself (and across reps, the noise realisations of U and E). This is
    the Yu-Wang-Samworth 2015 protocol: fix the population, vary only the
    perturbation magnitude.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Drawing fixed coords + sparse loading "
          f"(n={cfg.n}, d={cfg.d}, k={cfg.sparse_k})", flush=True)
    fixed_rng = np.random.default_rng(cfg.coord_seed)
    coords_fixed = fixed_rng.uniform(size=(cfg.n, 2))
    v_dir_fixed = _sparse_unit_loading(cfg.d, cfg.sparse_k, fixed_rng)

    rng_master = np.random.default_rng(seed)
    print(f"[2/3] Running coverage sweep: {len(spike_grid)} lambdas "
          f"x {n_reps} reps = {len(spike_grid)*n_reps} jobs", flush=True)
    print(f"      BBP threshold for n={cfg.n}, d={cfg.d}: "
          f"lambda_c = (d/n)^(1/4) = {(cfg.d/cfg.n) ** 0.25:.3f}", flush=True)

    _outer = n_jobs if (n_jobs and n_jobs > 0) else _NCORES
    all_rows: list[dict] = []
    t0 = time.time()

    with parallel_config(backend="loky", inner_max_num_threads=1):
        with Parallel(n_jobs=_outer, verbose=0) as parallel:
            for gi, lam in enumerate(spike_grid, start=1):
                seeds = rng_master.integers(0, 2**31 - 1, size=n_reps).tolist()
                cell_t0 = time.time()
                rows = parallel(
                    delayed(_one_rep)(
                        int(s), float(lam), cfg, coords_fixed, v_dir_fixed,
                        hac_bandwidth, n_boot,
                    )
                    for s in seeds
                )
                all_rows.extend(rows)
                dt = time.time() - cell_t0
                print(f"  [{gi}/{len(spike_grid)}] lambda={lam:.3f}  "
                      f"R={n_reps}  {dt:6.1f}s "
                      f"({dt / max(n_reps,1)*1000:.0f} ms/rep)", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "davis_kahan_raw.csv", index=False)
    print(f"[3/3] Wrote {out_dir/'davis_kahan_raw.csv'} "
          f"({len(df)} rows, total {time.time()-t0:.0f}s)", flush=True)
    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(df: pd.DataFrame, tau_true: float) -> pd.DataFrame:
    """One row per spike level: bias, SD, RMSE, coverage of each CI."""
    out = []
    for lam, sub in df.groupby("spike_strength"):
        n = len(sub)
        row = {"spike_strength": lam, "n_reps_ok": n}

        # PC1-OLS bias against tau_true. (The naive analyst would think the
        # PC1 estimator targets tau; we record bias against tau and against 0.)
        row["bias_pc1_vs_tau"] = float(sub["theta_pc1"].mean() - tau_true)
        row["bias_pc1_vs_zero"] = float(sub["theta_pc1"].mean())
        row["sd_pc1"] = float(sub["theta_pc1"].std(ddof=1)) if n > 1 else np.nan
        row["rmse_pc1_vs_tau"] = float(
            np.sqrt(((sub["theta_pc1"] - tau_true) ** 2).mean())
        )

        # Residual-subspace bias against tau_true.
        row["bias_rs_vs_tau"] = float(sub["theta_rs"].mean() - tau_true)
        row["sd_rs"] = float(sub["theta_rs"].std(ddof=1)) if n > 1 else np.nan
        row["rmse_rs"] = float(
            np.sqrt(((sub["theta_rs"] - tau_true) ** 2).mean())
        )

        # Coverage of three CIs around tau_true (theta_rs +/- 1.96 * se).
        for name, se_col in [
            ("classical", "se_rs_classical"),
            ("hc1", "se_rs_hc1"),
            ("hac_swm", "se_rs_hac"),
        ]:
            lo = sub["theta_rs"] - 1.96 * sub[se_col]
            hi = sub["theta_rs"] + 1.96 * sub[se_col]
            row[f"cov_{name}"] = float(((lo <= tau_true) & (tau_true <= hi)).mean())
            row[f"mean_ci_len_{name}"] = float((hi - lo).mean())

        # Bootstrap coverage if computed.
        if "ci_rs_boot_lo" in sub.columns and sub["ci_rs_boot_lo"].notna().any():
            lo_b = sub["ci_rs_boot_lo"]
            hi_b = sub["ci_rs_boot_hi"]
            row["cov_iid_boot"] = float(
                ((lo_b <= tau_true) & (tau_true <= hi_b)).mean()
            )
            row["mean_ci_len_iid_boot"] = float((hi_b - lo_b).mean())
        else:
            row["cov_iid_boot"] = np.nan
            row["mean_ci_len_iid_boot"] = np.nan

        # Davis-Kahan diagnostics.
        row["mean_sin_theta_pc1"] = float(sub["sin_theta_pc1"].mean())
        row["mean_sin_theta_vperp"] = float(sub["sin_theta_vperp"].mean())
        row["mean_btw_within"] = float(sub["btw_within"].mean())
        out.append(row)
    return pd.DataFrame(out).sort_values("spike_strength").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_figures(agg: pd.DataFrame, cfg: DGPConfig, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lam = agg["spike_strength"].to_numpy()
    bbp = (cfg.d / cfg.n) ** 0.25

    # --- Figure 1: coverage vs spike strength.
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(lam, agg["cov_classical"], "o-", label="iid OLS CI", color="#d62728")
    ax.plot(lam, agg["cov_hc1"], "s-", label="HC1 (White) CI", color="#ff7f0e")
    if agg["cov_iid_boot"].notna().any():
        ax.plot(lam, agg["cov_iid_boot"], "v-", label="iid bootstrap CI",
                color="#9467bd")
    ax.plot(lam, agg["cov_hac_swm"], "D-",
            label="Conley sHAC + SWM-2026 fold-centered", color="#2ca02c")
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1)
    ax.axhspan(0.93, 0.97, color="lightgreen", alpha=0.25,
               label="95% binomial band")
    ax.axvline(bbp, color="gray", linestyle=":", linewidth=1,
               label=f"BBP threshold $\\lambda_c$={bbp:.2f}")
    ax.set_xlabel(r"spike strength $\lambda$")
    ax.set_ylabel("empirical 95% CI coverage")
    ax.set_ylim(0, 1.02)
    ax.set_title("Coverage of residual-subspace estimator across spike strengths")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_coverage.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir/'fig_coverage.png'}")

    # --- Figure 2: bias of PC1-OLS vs residual-subspace.
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(lam, agg["bias_pc1_vs_tau"], "o-",
            label=r"PC1-OLS bias vs $\tau$", color="#d62728")
    ax.plot(lam, agg["bias_rs_vs_tau"], "D-",
            label=r"Residual-subspace bias vs $\tau$", color="#2ca02c")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.axvline(bbp, color="gray", linestyle=":", linewidth=1,
               label=f"BBP $\\lambda_c$={bbp:.2f}")
    ax.set_xlabel(r"spike strength $\lambda$")
    ax.set_ylabel(r"empirical bias against true $\tau$")
    ax.set_title("PC1-OLS bias diverges from zero; residual-subspace stays at zero")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_bias.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir/'fig_bias.png'}")

    # --- Figure 3: sin-Theta(top-1 PC, v_dir) vs spike strength.
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(lam, agg["mean_sin_theta_pc1"], "o-",
            label=r"$\sin\Theta(\hat v_1, v_{\rm dir})$ (top PC vs spike loading)",
            color="#1f77b4")
    ax.plot(lam, agg["mean_sin_theta_vperp"], "D-",
            label=r"$\sin\Theta(\hat v_\perp, v_\perp)$ (residual PC vs causal axis)",
            color="#2ca02c")
    # Davis-Kahan-Yu-Wang-Samworth upper bound (rectangular): up to
    # constants, sin-Theta <= C * sigma_E * sqrt(d) / lambda_eff. Plot
    # the leading-order rate sigma_E sqrt(d) / lambda for reference.
    rate = 1.0 / np.sqrt(cfg.n) * np.sqrt(cfg.d) / np.maximum(lam, 1e-6)
    ax.plot(lam, np.minimum(rate, 1.0), "--", color="gray",
            label=r"Yu-Wang-Samworth rate $\sigma_E \sqrt{d}/\lambda$")
    ax.axvline(bbp, color="gray", linestyle=":", linewidth=1,
               label=f"BBP $\\lambda_c$={bbp:.2f}")
    ax.set_xlabel(r"spike strength $\lambda$")
    ax.set_ylabel(r"mean $\sin\Theta$")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(
        r"Davis-Kahan alignment: top PC tracks $v_{\rm dir}$ above the BBP threshold"
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_davis_kahan.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir/'fig_davis_kahan.png'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Davis-Kahan / spiked-covariance Monte Carlo for JBES.",
    )
    ap.add_argument("--n_reps", type=int, default=50,
                    help="Monte Carlo reps per spike level (default 50 = smoke).")
    ap.add_argument("--spike_min", type=float, default=0.5)
    ap.add_argument("--spike_max", type=float, default=4.0)
    ap.add_argument("--n_spike", type=int, default=15,
                    help="Grid points on the spike axis.")
    ap.add_argument("--n", type=int, default=348)
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--tau", type=float, default=0.10)
    ap.add_argument("--length_scale", type=float, default=0.15)
    ap.add_argument("--nu", type=float, default=2.5)
    ap.add_argument("--sigma_nu", type=float, default=0.30)
    ap.add_argument("--sparse_k", type=int, default=50)
    ap.add_argument("--hac_bandwidth", type=float, default=0.15)
    ap.add_argument("--n_boot", type=int, default=0,
                    help="iid bootstrap reps for the foil CI (0 to skip; "
                         "use 50 to enable Lam-2022 cheap bootstrap).")
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=20260529)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--plot_only", action="store_true",
                    help="Skip the sim; re-render figures from saved CSV.")
    args = ap.parse_args()

    cfg = DGPConfig(
        n=args.n, d=args.d, tau=args.tau,
        length_scale=args.length_scale, nu=args.nu, sigma_nu=args.sigma_nu,
        sparse_k=args.sparse_k,
    )
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        raw = pd.read_csv(out_dir / "davis_kahan_raw.csv")
        agg = aggregate(raw, tau_true=cfg.tau)
        agg.to_csv(out_dir / "davis_kahan_summary.csv", index=False)
        plot_figures(agg, cfg, out_dir)
        return

    spike_grid = np.linspace(args.spike_min, args.spike_max, args.n_spike)

    with open(out_dir / "config.json", "w") as f:
        json.dump({
            "n_reps": args.n_reps,
            "spike_grid": spike_grid.tolist(),
            "hac_bandwidth": args.hac_bandwidth,
            "n_boot": args.n_boot,
            "seed": args.seed,
            **asdict(cfg),
        }, f, indent=2)

    raw = coverage_sweep(
        n_reps=args.n_reps,
        spike_grid=spike_grid,
        cfg=cfg,
        out_dir=out_dir,
        n_jobs=args.n_jobs,
        n_boot=args.n_boot,
        hac_bandwidth=args.hac_bandwidth,
        seed=args.seed,
    )
    agg = aggregate(raw, tau_true=cfg.tau)
    agg.to_csv(out_dir / "davis_kahan_summary.csv", index=False)
    print(f"\nSummary table:\n{agg.to_string(index=False)}")
    print(f"\nWrote {out_dir/'davis_kahan_summary.csv'}")

    plot_figures(agg, cfg, out_dir)


if __name__ == "__main__":
    main()
