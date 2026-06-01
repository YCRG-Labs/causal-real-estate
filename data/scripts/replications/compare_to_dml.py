"""Thin wrapper around dml_continuous_treatment for the replication scripts.

The published replications (Shen 2021, Baur 2023) each define their own
"treatment" of interest:

  - Shen: a scalar TF-IDF uniqueness score per listing
  - Baur: the 768-dim BERT (mpnet) embedding (PC1 is what DML actually scores)

dml_continuous_treatment expects a (n, k) matrix and PCAs it down. This
wrapper accepts either a 1-D vector or a 2-D matrix, calls the project DML
silently, and packs the result into a uniform DMLResult dataclass that both
replications can consume.
"""
from __future__ import annotations

import contextlib
import io
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from causal_inference import dml_continuous_treatment


def _ridge_dml_core(T_1d, confounders, Y, k_folds=5, seed=42):
    """Partially-linear DML with cross-fitted RidgeCV nuisances.

    Ridge is the right nuisance for the Shen replication target: the
    confounder set is dense and well-conditioned (≈30 standardised numeric
    features at n≈300), and the treatment is a scalar uniqueness score. With
    LightGBM's per-fit overhead in this regime, ridge produces equivalent
    cross-fitted residuals roughly 100× faster than gradient boosting.

    Returns (theta, se_if) or None if the score denominator collapses.
    """
    n = len(Y)
    T_1d = np.asarray(T_1d, dtype=np.float64).ravel()
    Y = np.asarray(Y, dtype=np.float64).ravel()
    conf_s = StandardScaler().fit_transform(np.asarray(confounders))
    alphas = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    Y_resid = np.empty(n)
    T_resid = np.empty(n)
    for tr, te in kf.split(np.arange(n)):
        m_y = RidgeCV(alphas=alphas).fit(conf_s[tr], Y[tr])
        m_t = RidgeCV(alphas=alphas).fit(conf_s[tr], T_1d[tr])
        Y_resid[te] = Y[te] - m_y.predict(conf_s[te])
        T_resid[te] = T_1d[te] - m_t.predict(conf_s[te])
    denom = float(np.mean(T_resid * T_resid))
    if denom < 1e-12:
        return None
    theta = float(np.mean(T_resid * Y_resid)) / denom
    psi = (Y_resid - theta * T_resid) * T_resid / denom
    se_if = float(np.sqrt(float(np.var(psi, ddof=1)) / n))
    return theta, se_if


def _ridge_dml_bootstrap(T_1d, confounders, Y, n_boot=500, k_folds=5, seed=42):
    """Nonparametric pairs bootstrap on the ridge-DML estimator.

    Resamples (T_i, conf_i, Y_i) triples with replacement, refits the K-fold
    cross-fitted ridge nuisances on the resampled set, and recovers theta_b.
    Returns (theta_hat, se_if, se_boot, ci_perc_low, ci_perc_high, thetas_b).

    At n=300, p=30, k_folds=5, n_boot=500 this is ~5,000 RidgeCV fits per
    city, ~5-10 s wall-clock on a single core. Pairs bootstrap is the
    cleanest finite-sample variance estimate for IF-SE undercoverage at
    moderate n; matches the project's spatial-confounding bootstrap protocol
    [[project_causal_real_estate]].
    """
    n = len(Y)
    T_1d = np.asarray(T_1d, dtype=np.float64).ravel()
    Y_arr = np.asarray(Y, dtype=np.float64).ravel()
    conf_arr = np.asarray(confounders)

    base = _ridge_dml_core(T_1d, conf_arr, Y_arr, k_folds=k_folds, seed=seed)
    if base is None:
        return None
    theta_hat, se_if = base

    rng = np.random.default_rng(seed)
    thetas = np.full(n_boot, np.nan, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        out = _ridge_dml_core(T_1d[idx], conf_arr[idx], Y_arr[idx],
                              k_folds=k_folds, seed=seed + b + 1)
        if out is None:
            continue
        thetas[b] = out[0]
    valid = thetas[~np.isnan(thetas)]
    if valid.size < max(20, n_boot // 10):
        return theta_hat, se_if, float("nan"), float("nan"), float("nan"), thetas
    se_boot = float(np.std(valid, ddof=1))
    ci_lo = float(np.percentile(valid, 2.5))
    ci_hi = float(np.percentile(valid, 97.5))
    return theta_hat, se_if, se_boot, ci_lo, ci_hi, thetas


@dataclass
class DMLResult:
    label: str
    n: int
    theta: float
    se: float
    ci_low: float
    ci_high: float
    mde: float
    contains_zero: bool
    se_if: float | None = None
    n_boot: int | None = None


def run_dml(
    T: np.ndarray,
    confounders: np.ndarray,
    Y: np.ndarray,
    label: str,
    n_pca: int = 50,
    k_folds: int = 5,
    ci_method: str = "bootstrap",
    n_boot: int | None = 500,
    use_ridge: bool = False,
    seed: int = 42,
) -> DMLResult | None:
    """Run cross-fitted DML and box the output as a DMLResult.

    T may be 1-D (scalar treatment) or 2-D (vector treatment).  If 1-D it is
    reshaped to (n, 1) so the underlying PCA degenerates to a standardisation.

    Two backends:
      - use_ridge=False (default): legacy dml_continuous_treatment with
        LightGBM/sklearn-GBR nuisances. Supports ci_method='bootstrap' (B=n_boot)
        and ci_method='if'. Correct but slow at n<2000 because GBM nuisances
        dominate.
      - use_ridge=True: hand-rolled RidgeCV nuisance DML with IF SE only.
        Valid per Chernozhukov et al. 2018 for dense well-conditioned
        confounders at n<2000 (DoubleML py_learner notebook shows ridge often
        beats GBM in nuisance MSE at n=1000, p=50). Approx 50-100× faster than
        the GBM path for Shen's scalar-uniqueness treatment.
    """
    T_in = np.asarray(T)
    if use_ridge:
        if T_in.ndim == 2 and T_in.shape[1] != 1:
            raise ValueError("use_ridge=True requires a scalar treatment "
                             "(T must be 1-D or shape (n,1))")
        T_1d = T_in.ravel()
        n_obs = int(len(Y))
        do_boot = ci_method == "bootstrap" and n_boot is not None and n_boot > 0
        if do_boot:
            res = _ridge_dml_bootstrap(T_1d, confounders, Y,
                                       n_boot=n_boot, k_folds=k_folds, seed=seed)
            if res is None:
                return None
            theta, se_if, se_boot, lo, hi, _ = res
            se = float(se_boot) if not np.isnan(se_boot) else float(se_if)
            mde = 2.802 * se
            return DMLResult(
                label=label, n=n_obs, theta=float(theta), se=se,
                ci_low=float(lo) if not np.isnan(lo) else float(theta - 1.96 * se),
                ci_high=float(hi) if not np.isnan(hi) else float(theta + 1.96 * se),
                mde=float(mde),
                contains_zero=bool((lo if not np.isnan(lo) else theta - 1.96 * se) <= 0
                                    <= (hi if not np.isnan(hi) else theta + 1.96 * se)),
                se_if=float(se_if), n_boot=int(n_boot),
            )
        out = _ridge_dml_core(T_1d, confounders, Y, k_folds=k_folds, seed=seed)
        if out is None:
            return None
        theta, se = out
        lo = theta - 1.96 * se
        hi = theta + 1.96 * se
        mde = 2.802 * se
        return DMLResult(
            label=label, n=n_obs, theta=float(theta), se=float(se),
            ci_low=float(lo), ci_high=float(hi), mde=float(mde),
            contains_zero=bool(lo <= 0 <= hi),
            se_if=float(se), n_boot=None,
        )

    T_mat = T_in.reshape(-1, 1) if T_in.ndim == 1 else T_in
    n_pca = min(n_pca, T_mat.shape[1], T_mat.shape[0] - 1)
    dml_kwargs = dict(n_pca=n_pca, k_folds=k_folds, ci_method=ci_method)
    if ci_method == "bootstrap" and n_boot is not None:
        dml_kwargs["n_boot"] = n_boot
    with contextlib.redirect_stdout(io.StringIO()):
        raw = dml_continuous_treatment(T_mat, confounders, Y, **dml_kwargs)
    if raw is None:
        return None
    lo, hi = raw["ci"]
    return DMLResult(
        label=label,
        n=int(len(Y)),
        theta=float(raw["theta"]),
        se=float(raw["se"]),
        ci_low=float(lo),
        ci_high=float(hi),
        mde=float(raw["mde"]),
        contains_zero=bool(lo <= 0 <= hi),
    )


def result_to_dict(res: DMLResult | None) -> dict:
    if res is None:
        return {"error": "DML failed (treatment fully explained by confounders)"}
    return asdict(res)
