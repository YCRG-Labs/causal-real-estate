"""Monte Carlo coverage simulation for DML IF-SE vs bootstrap CIs.

Per the methodology research, the canonical setup is the CCDDHNR2018
partially-linear-model DGP (Chernozhukov et al. 2018 EJ) with AR(1)
confounder correlation, packaged as `make_plr_CCDDHNR2018` in DoubleML.
Truth is analytic: theta_0 = alpha. No empirical truth calibration
required (the n_pop = 10,000 calibration step in the legacy
run_simulation.py was the source of the hang).

Four estimators reported, ranked by validity:

  1. Naive ML plug-in: GBM regression of Y on [T, X], report coefficient
     on T. NOT orthogonalised; expected to have severe bias and low
     coverage. Sanity-check floor.
  2. DML + IF-SE: 5-fold cross-fit ridge nuisances, theta_hat from the
     orthogonal score, asymptotic IF standard error. The headline
     production estimator; the hypothesis is that it under-covers at
     moderate n.
  3. DML + multiplier bootstrap: same fit, then a B-iteration
     Rademacher-multiplier bootstrap on the influence functions
     (Chernozhukov et al. 2018 §4.3). Cheap because it does not refit
     nuisances; conditions on eta_hat.
  4. DML + pairs bootstrap: B-iteration nonparametric resampling of
     (Y_i, T_i, X_i) triples with full refit of the DML pipeline on
     each resample. Captures nuisance estimation uncertainty that the
     multiplier bootstrap conditions away. Novel claim: this is what
     restores coverage at moderate n when IF-SE under-covers.

Reports per cell: bias, RMSE, avg_SE, sd_theta_hat, 95% coverage,
mean CI width. Output CSV plus printed coverage table.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import (
    LogisticRegression, LogisticRegressionCV, Ridge, RidgeCV, LinearRegression,
)
from sklearn.isotonic import IsotonicRegression
try:
    from venn_abers import VennAbers
    _HAS_VENN_ABERS = True
except ImportError:
    _HAS_VENN_ABERS = False

K_FOLDS_CAL = 2
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import warnings as _warnings
_warnings.filterwarnings("ignore", category=RuntimeWarning,
                          module=r"venn_abers")

REPO = Path(__file__).resolve().parents[3]
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
FIXED_ALPHA = 10.0


def make_plr_weak_overlap(n: int, dim_x: int = 20, alpha: float = 0.5,
                          rho: float = 0.7,
                          target_R2_DX: float = 0.90,
                          nonlinear_g: bool = True,
                          rng: np.random.Generator = None):
    """Weak-overlap PLR DGP. Targets R²(D|X) ≈ 0.90 to put the
    estimator in the ill-conditioned regime documented in Saco (2025,
    arXiv 2512.07083) Table A.5 and Bach/Schacht (2024, arXiv 2409.04874)
    DGP4. In this regime the orthogonal score's denominator E[T_resid²]
    shrinks toward zero and small biases in the nuisance estimate
    amplify into large coverage errors. Nonlinear g_0 makes ridge a
    misspecified outcome learner.

    Truth: theta_0 = alpha (analytic).
    """
    rng = rng or np.random.default_rng()
    idx = np.arange(dim_x)
    Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
    L = np.linalg.cholesky(Sigma)
    X = rng.standard_normal((n, dim_x)) @ L.T
    v = rng.standard_normal(n)
    zeta = rng.standard_normal(n)
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))
    c = float(np.sqrt(target_R2_DX / (1 - target_R2_DX) / 1.06))
    m0 = c * (X[:, 0] + 0.5 * sig(X[:, 2]))
    D = m0 + 1.0 * v
    if nonlinear_g:
        g0 = (sig(X[:, 0]) + 0.5 * np.sin(X[:, 2])
              + 0.3 * X[:, 0] * X[:, 4]
              + 0.5 * (X[:, 1] ** 2 - 1))
    else:
        g0 = sig(X[:, 0]) + 0.25 * X[:, 2]
    Y = alpha * D + g0 + 1.0 * zeta
    return Y, D, X


def make_irm_dgp4(n: int, dim_x: int = 30, theta: float = 0.5,
                   rng: np.random.Generator = None, **_):
    """Ballinari & Bearth (2024) Table 2 DGP 4 — verbatim port.

    arXiv:2409.04874v2 §5.1 p.10. Difficult outcome (Friedman 1991
    surface) with difficult propensity (Beta-CDF of the min of two
    covariates). They report DML-AIPW coverage with lasso nuisances at
    N=2000 equal to 0.282; with ridge on the same surface coverage is
    expected to land in roughly [0.30, 0.55]. Paired-bootstrap CIs
    materially restore coverage (their Table 3).

        X_i        ~ Uniform(0, 1)^30
        b(X)       = sin(π X_1 X_2) + 2(X_3 − 0.5)^2 + X_4 + 0.5 X_5
        p(X)       = 0.1 + 0.6 · Beta_{2,4}.cdf(min(X_1, X_2))
        D_i        ~ Bernoulli(p(X_i))
        tau_i      = 2 · theta · (X_1 + X_2) / 2 = theta · (X_1 + X_2)
        Y_i        = b(X_i) + (D_i − 0.5) · tau_i + ε_i, ε ~ N(0, 1)

    Heterogeneity is scaled by theta so that E[tau] = theta · E[X_1+X_2]
    = theta. At theta=0.5 the heterogeneity matches the published spec
    exactly (tau = (X_1+X_2)/2); at theta=0 the effect vanishes and the
    size of the test can be evaluated. `dim_x` is hard-coded to 30 per
    the published spec; `rho` (legacy AR(1) hook) is absorbed by **_.
    """
    from scipy.stats import beta as _beta_dist
    rng = rng or np.random.default_rng()
    X = rng.uniform(0.0, 1.0, size=(n, dim_x))
    b = (np.sin(np.pi * X[:, 0] * X[:, 1])
         + 2.0 * (X[:, 2] - 0.5) ** 2
         + X[:, 3] + 0.5 * X[:, 4])
    p = 0.1 + 0.6 * _beta_dist.cdf(np.minimum(X[:, 0], X[:, 1]), 2.0, 4.0)
    D = (rng.uniform(size=n) < p).astype(np.float64)
    tau = theta * (X[:, 0] + X[:, 1])
    Y = b + (D - 0.5) * tau + rng.standard_normal(n)
    return Y, D, X


_HGBM_REG_KW = dict(max_depth=3, max_iter=100, learning_rate=0.1,
                     early_stopping=True, validation_fraction=0.1,
                     tol=1e-4, n_iter_no_change=5, max_bins=64,
                     random_state=0)
_HGBM_CLF_KW = dict(max_depth=3, max_iter=100, learning_rate=0.1,
                     early_stopping=True, validation_fraction=0.1,
                     tol=1e-4, n_iter_no_change=5, max_bins=32,
                     categorical_features=[], random_state=0)


def _fit_outcome(Xs_tr, Y_tr, learner, alpha):
    if learner == "ridge":
        return Ridge(alpha=alpha, solver="cholesky").fit(Xs_tr, Y_tr)
    if learner == "hgbm":
        with threadpool_limits(limits=1, user_api="openmp"):
            return HistGradientBoostingRegressor(**_HGBM_REG_KW).fit(Xs_tr, Y_tr)
    raise ValueError(f"unknown outcome learner: {learner}")


def _fit_propensity(Xs_tr, D_tr, learner, C):
    if learner == "logreg":
        return LogisticRegression(C=C, solver="lbfgs", max_iter=200,
                                   n_jobs=1).fit(Xs_tr, D_tr)
    if learner == "hgbm":
        with threadpool_limits(limits=1, user_api="openmp"):
            return HistGradientBoostingClassifier(**_HGBM_CLF_KW).fit(Xs_tr, D_tr)
    raise ValueError(f"unknown propensity learner: {learner}")


def _tune_aipw_once(Xs, Y, D, learner_outcome="ridge",
                    learner_propensity="logreg"):
    """One-shot full-sample tuning of the AIPW nuisances. Returns the
    selected (alpha_0, alpha_1, C_lr) for the outcome ridges and the
    propensity logistic regression. When learner_outcome="hgbm" or
    learner_propensity="hgbm" the corresponding alpha/C is unused (HGBM
    self-tunes via early stopping) and the returned value is a no-op
    placeholder so downstream signatures stay stable.

    Per DoubleML's `tune_on_folds = FALSE` convention (the default in
    their R interface; see https://docs.doubleml.org/stable/guide/
    learners.html), one set of hyperparameters is chosen on the full
    sample and then held fixed across all cross-fitting folds AND across
    all bootstrap resamples. This is the natural extension of the DML
    fit-time convention to pairs-bootstrap inference: the bootstrap
    measures the resampling distribution of theta_hat given fixed
    nuisance specifications, not given a re-tuned spec on each resample.

    Microbenchmarks at the (n, p=30) sizes used here show RidgeCV vs
    Ridge(solver='cholesky') is ~7x slower per fit and LogisticReg-
    ressionCV vs LogisticRegression(C=C*) is ~9x slower; moving the
    tuning outside the per-fold and per-bootstrap loops gives an ~8x
    speedup on the AIPW path.
    """
    D_int = D.astype(int)
    mask0 = D_int == 0
    mask1 = D_int == 1
    if learner_outcome == "ridge":
        alpha0 = float(RidgeCV(alphas=RIDGE_ALPHAS).fit(
            Xs[mask0], Y[mask0]).alpha_) if mask0.sum() >= 5 else 1.0
        alpha1 = float(RidgeCV(alphas=RIDGE_ALPHAS).fit(
            Xs[mask1], Y[mask1]).alpha_) if mask1.sum() >= 5 else 1.0
    else:
        alpha0 = alpha1 = 0.0
    if learner_propensity == "logreg":
        try:
            lr_cv = LogisticRegressionCV(
                Cs=5, cv=3, max_iter=2000, n_jobs=1,
            ).fit(Xs, D_int)
            C_lr = float(lr_cv.C_[0])
        except Exception:
            C_lr = 1.0
    else:
        C_lr = 0.0
    return alpha0, alpha1, C_lr


def _aipw_cross_fit(Y, D, X, k_folds=5, seed=42, Xs=None,
                    alpha0=None, alpha1=None, C_lr=None,
                    learner_outcome="ridge", learner_propensity="logreg"):
    """Cross-fit AIPW nuisances: g0(X) = E[Y|X,D=0], g1(X) = E[Y|X,D=1],
    p(X) = P(D=1|X). When `(Xs, alpha0, alpha1, C_lr)` are supplied uses
    fixed hyperparameters with `Ridge(solver='cholesky')` and
    `LogisticRegression(C=C_lr)` — the fast path, ~8x faster than per-
    fold CV at (n=2000, p=30). Returns the AIPW score:

        psi_i = g1_i - g0_i + D_i (Y_i - g1_i) / p_i
                              - (1 - D_i)(Y_i - g0_i)/(1 - p_i)

    plus the per-fold predictions for diagnostics. The legacy CV-per-
    fold path is preserved when the frozen params are None (callers
    that don't pass them get the original behaviour).
    """
    n = len(Y)
    if Xs is None:
        Xs = StandardScaler().fit_transform(X)
    if alpha0 is None or alpha1 is None or C_lr is None:
        alpha0_fb, alpha1_fb, C_lr_fb = _tune_aipw_once(
            Xs, Y, D,
            learner_outcome=learner_outcome,
            learner_propensity=learner_propensity,
        )
        alpha0 = alpha0 if alpha0 is not None else alpha0_fb
        alpha1 = alpha1 if alpha1 is not None else alpha1_fb
        C_lr   = C_lr   if C_lr   is not None else C_lr_fb
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    g0_hat = np.empty(n); g1_hat = np.empty(n); p_hat = np.empty(n)
    for tr, te in kf.split(np.arange(n)):
        D_tr = D[tr].astype(int)
        try:
            m_p = _fit_propensity(Xs[tr], D_tr,
                                   learner=learner_propensity, C=C_lr)
            p_te = m_p.predict_proba(Xs[te])[:, 1]
        except Exception:
            p_te = np.full(len(te), float(D_tr.mean()))
        p_te = np.clip(p_te, 1e-3, 1 - 1e-3)
        mask0 = D_tr == 0
        mask1 = D_tr == 1
        if mask0.sum() >= 5:
            g0_te = _fit_outcome(Xs[tr][mask0], Y[tr][mask0],
                                  learner=learner_outcome, alpha=alpha0
                                  ).predict(Xs[te])
        else:
            g0_te = np.full(len(te), float(Y[tr][mask0].mean()) if mask0.sum() else 0.0)
        if mask1.sum() >= 5:
            g1_te = _fit_outcome(Xs[tr][mask1], Y[tr][mask1],
                                  learner=learner_outcome, alpha=alpha1
                                  ).predict(Xs[te])
        else:
            g1_te = np.full(len(te), float(Y[tr][mask1].mean()) if mask1.sum() else 0.0)
        g0_hat[te] = g0_te
        g1_hat[te] = g1_te
        p_hat[te] = p_te
    psi = (g1_hat - g0_hat
           + D * (Y - g1_hat) / p_hat
           - (1.0 - D) * (Y - g0_hat) / (1.0 - p_hat))
    return psi, g0_hat, g1_hat, p_hat


def aipw_if(Y, D, X, k_folds=5, seed=42, Xs=None,
            alpha0=None, alpha1=None, C_lr=None,
            learner_outcome="ridge", learner_propensity="logreg"):
    psi, _, _, _ = _aipw_cross_fit(
        Y, D, X, k_folds=k_folds, seed=seed,
        Xs=Xs, alpha0=alpha0, alpha1=alpha1, C_lr=C_lr,
        learner_outcome=learner_outcome,
        learner_propensity=learner_propensity,
    )
    theta = float(np.mean(psi))
    se = float(np.sqrt(np.var(psi, ddof=1) / len(Y)))
    return theta, se, psi


def aipw_mult_boot(theta_hat, psi, n_boot=500, seed=42):
    n = len(psi)
    rng = np.random.default_rng(seed)
    eps = rng.choice([-1.0, 1.0], size=(n_boot, n))
    thetas = theta_hat + (eps * (psi - theta_hat)).mean(axis=1)
    se = float(np.std(thetas, ddof=1))
    lo, hi = float(np.percentile(thetas, 2.5)), float(np.percentile(thetas, 97.5))
    return theta_hat, se, lo, hi


def aipw_pairs_boot(Y, D, X, n_boot=100, k_folds=5, seed=42,
                    learner_outcome="ridge", learner_propensity="logreg"):
    """Pairs bootstrap for AIPW with frozen hyperparameters.

    Tunes (alpha_0, alpha_1, C_lr) once on the full sample (DoubleML's
    tune_on_folds=FALSE convention), then resamples (Y_i, D_i, X_i)
    rows and re-fits the cross-fit AIPW at the FROZEN params for each
    of `n_boot` resamples. Row resampling preserves column moments so
    we standardize X once and reuse the row-sliced Xs[idx] downstream.

    The point estimate is computed under the same frozen params so the
    bootstrap distribution is for the estimator the user is actually
    deploying — not a different (CV-per-fold) estimator.
    """
    n = len(Y)
    Xs_full = StandardScaler().fit_transform(X)
    alpha0, alpha1, C_lr = _tune_aipw_once(
        Xs_full, Y, D,
        learner_outcome=learner_outcome,
        learner_propensity=learner_propensity,
    )
    theta_hat, _, _ = aipw_if(
        Y, D, X, k_folds=k_folds, seed=seed,
        Xs=Xs_full, alpha0=alpha0, alpha1=alpha1, C_lr=C_lr,
        learner_outcome=learner_outcome,
        learner_propensity=learner_propensity,
    )
    if np.isnan(theta_hat):
        return theta_hat, float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    thetas = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            psi_b, _, _, _ = _aipw_cross_fit(
                Y[idx], D[idx], X[idx], k_folds=k_folds,
                seed=seed + b + 1, Xs=Xs_full[idx],
                alpha0=alpha0, alpha1=alpha1, C_lr=C_lr,
                learner_outcome=learner_outcome,
                learner_propensity=learner_propensity,
            )
            thetas[b] = float(np.mean(psi_b))
        except Exception:
            continue
    valid = thetas[~np.isnan(thetas)]
    if valid.size < max(5, n_boot // 4):
        return theta_hat, float("nan"), float("nan"), float("nan")
    se = float(np.std(valid, ddof=1))
    lo, hi = float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))
    return theta_hat, se, lo, hi



def _fit_calibrator(raw_ps, D, method="isotonic"):
    """B&B (2024) calibrator factory. Returns a callable f(raw) -> calibrated.

    Implements isotonic / Platt / beta scaling per Sections 3.2-3.4. Isotonic
    is the default (parameter-free, asymptotically licensed per Theorem 1,
    coverage on DGP4/Lasso 0.937 per Table 2). Platt is NOT asymptotically
    licensed (Theorem 1 footnote) so we keep it for sensitivity only.
    """
    raw_ps = np.clip(np.asarray(raw_ps, dtype=float), 1e-6, 1 - 1e-6)
    D = np.asarray(D, dtype=int)
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-3, y_max=1 - 1e-3)
        iso.fit(raw_ps, D)
        return lambda p: iso.transform(np.clip(p, 1e-6, 1 - 1e-6))
    if method == "venn_abers":
        if not _HAS_VENN_ABERS:
            raise ImportError(
                "pip install venn-abers (required for --calibration venn_abers)"
            )
        va = VennAbers()
        p_cal_2col = np.column_stack([1.0 - raw_ps, raw_ps])
        va.fit(p_cal_2col, D)
        def _apply(p):
            p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
            p_2col = np.column_stack([1.0 - p, p])
            p_prime, _ = va.predict_proba(p_2col)
            return np.clip(p_prime[:, 1], 1e-3, 1 - 1e-3)
        return _apply
    if method == "platt":
        z = np.log(raw_ps / (1.0 - raw_ps)).reshape(-1, 1)
        try:
            lr = LogisticRegression(C=1e6, solver="lbfgs",
                                     max_iter=500).fit(z, D)
        except Exception:
            return lambda p: np.clip(p, 1e-3, 1 - 1e-3)
        def _apply(p):
            p = np.clip(p, 1e-6, 1 - 1e-6)
            zz = np.log(p / (1.0 - p)).reshape(-1, 1)
            return np.clip(lr.predict_proba(zz)[:, 1], 1e-3, 1 - 1e-3)
        return _apply
    if method == "beta":
        Z = np.column_stack([np.log(raw_ps), -np.log(1.0 - raw_ps)])
        try:
            lr = LogisticRegression(C=1e6, solver="lbfgs",
                                     max_iter=500).fit(Z, D)
        except Exception:
            return lambda p: np.clip(p, 1e-3, 1 - 1e-3)
        def _apply(p):
            p = np.clip(p, 1e-6, 1 - 1e-6)
            ZZ = np.column_stack([np.log(p), -np.log(1.0 - p)])
            return np.clip(lr.predict_proba(ZZ)[:, 1], 1e-3, 1 - 1e-3)
        return _apply
    raise ValueError(f"unknown calibration method: {method}")


def _aipw_calibrated_cross_fit(Y, D, X, k_folds=5, seed=42, Xs=None,
                                alpha0=None, alpha1=None, C_lr=None,
                                calibration="isotonic", j_subfolds=2,
                                learner_outcome="ridge",
                                learner_propensity="logreg"):
    """Cross-fit AIPW with propensity-score calibration per Ballinari &
    Bearth (2024) Algorithm 1. Within each outer fold k the held-out
    indices `te` are partitioned into J=`j_subfolds` inner sub-folds;
    for each j a calibrator f is fit on (raw_ps, D) over the other J-1
    sub-folds and applied to the held-out sub-fold. Outcome ridges are
    not calibrated. Falls back to raw PS when a sub-fold has only one
    class. Returns the AIPW score under the calibrated PS plus raw PS
    for diagnostics.
    """
    n = len(Y)
    if Xs is None:
        Xs = StandardScaler().fit_transform(X)
    if alpha0 is None or alpha1 is None or C_lr is None:
        alpha0_fb, alpha1_fb, C_lr_fb = _tune_aipw_once(
            Xs, Y, D,
            learner_outcome=learner_outcome,
            learner_propensity=learner_propensity,
        )
        alpha0 = alpha0 if alpha0 is not None else alpha0_fb
        alpha1 = alpha1 if alpha1 is not None else alpha1_fb
        C_lr   = C_lr   if C_lr   is not None else C_lr_fb
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    g0_hat = np.empty(n); g1_hat = np.empty(n)
    p_hat = np.empty(n); p_hat_raw = np.empty(n)
    rng = np.random.default_rng(seed)
    for tr, te in kf.split(np.arange(n)):
        D_tr = D[tr].astype(int)
        try:
            m_p = _fit_propensity(Xs[tr], D_tr,
                                   learner=learner_propensity, C=C_lr)
            p_te_raw = m_p.predict_proba(Xs[te])[:, 1]
        except Exception:
            p_te_raw = np.full(len(te), float(D_tr.mean()))
        p_te_raw = np.clip(p_te_raw, 1e-3, 1 - 1e-3)
        mask0 = D_tr == 0
        mask1 = D_tr == 1
        if mask0.sum() >= 5:
            g0_te = _fit_outcome(Xs[tr][mask0], Y[tr][mask0],
                                  learner=learner_outcome, alpha=alpha0
                                  ).predict(Xs[te])
        else:
            g0_te = np.full(len(te), float(Y[tr][mask0].mean()) if mask0.sum() else 0.0)
        if mask1.sum() >= 5:
            g1_te = _fit_outcome(Xs[tr][mask1], Y[tr][mask1],
                                  learner=learner_outcome, alpha=alpha1
                                  ).predict(Xs[te])
        else:
            g1_te = np.full(len(te), float(Y[tr][mask1].mean()) if mask1.sum() else 0.0)
        n_te = len(te)
        D_te = D[te].astype(int)
        perm = rng.permutation(n_te)
        sub = np.array_split(perm, j_subfolds)
        p_te_cal = np.empty(n_te)
        for j in range(j_subfolds):
            idx_E = sub[j]
            idx_C = np.concatenate([sub[i] for i in range(j_subfolds) if i != j])
            D_C = D_te[idx_C]
            if D_C.sum() < 2 or (len(D_C) - D_C.sum()) < 2:
                p_te_cal[idx_E] = p_te_raw[idx_E]
                continue
            try:
                f_cal = _fit_calibrator(p_te_raw[idx_C], D_C,
                                         method=calibration)
                p_te_cal[idx_E] = f_cal(p_te_raw[idx_E])
            except Exception:
                p_te_cal[idx_E] = p_te_raw[idx_E]
        p_te_cal = np.clip(p_te_cal, 1e-3, 1 - 1e-3)
        g0_hat[te] = g0_te
        g1_hat[te] = g1_te
        p_hat[te] = p_te_cal
        p_hat_raw[te] = p_te_raw
    psi = (g1_hat - g0_hat
           + D * (Y - g1_hat) / p_hat
           - (1.0 - D) * (Y - g0_hat) / (1.0 - p_hat))
    return psi, g0_hat, g1_hat, p_hat, p_hat_raw


def aipw_cal_if(Y, D, X, k_folds=5, seed=42, Xs=None,
                alpha0=None, alpha1=None, C_lr=None,
                calibration="isotonic",
                learner_outcome="ridge", learner_propensity="logreg"):
    psi, _, _, _, _ = _aipw_calibrated_cross_fit(
        Y, D, X, k_folds=k_folds, seed=seed, Xs=Xs,
        alpha0=alpha0, alpha1=alpha1, C_lr=C_lr, calibration=calibration,
        learner_outcome=learner_outcome,
        learner_propensity=learner_propensity,
    )
    theta = float(np.mean(psi))
    se = float(np.sqrt(np.var(psi, ddof=1) / len(Y)))
    return theta, se, psi


def aipw_cal_pairs_boot(Y, D, X, n_boot=50, k_folds=5, seed=42,
                         calibration="isotonic",
                         learner_outcome="ridge",
                         learner_propensity="logreg"):
    """Pairs bootstrap for the calibrated AIPW. Frozen hyperparameters
    and standardization (B&B + DoubleML tune_on_folds=FALSE convention);
    calibrator is refit per resample inside _aipw_calibrated_cross_fit.
    """
    n = len(Y)
    Xs_full = StandardScaler().fit_transform(X)
    alpha0, alpha1, C_lr = _tune_aipw_once(
        Xs_full, Y, D,
        learner_outcome=learner_outcome,
        learner_propensity=learner_propensity,
    )
    theta_hat, _, _ = aipw_cal_if(
        Y, D, X, k_folds=k_folds, seed=seed, Xs=Xs_full,
        alpha0=alpha0, alpha1=alpha1, C_lr=C_lr, calibration=calibration,
        learner_outcome=learner_outcome,
        learner_propensity=learner_propensity,
    )
    if np.isnan(theta_hat):
        return theta_hat, float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    thetas = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            psi_b, _, _, _, _ = _aipw_calibrated_cross_fit(
                Y[idx], D[idx], X[idx], k_folds=k_folds,
                seed=seed + b + 1, Xs=Xs_full[idx],
                alpha0=alpha0, alpha1=alpha1, C_lr=C_lr,
                calibration=calibration,
                learner_outcome=learner_outcome,
                learner_propensity=learner_propensity,
            )
            thetas[b] = float(np.mean(psi_b))
        except Exception:
            continue
    valid = thetas[~np.isnan(thetas)]
    if valid.size < max(5, n_boot // 4):
        return theta_hat, float("nan"), float("nan"), float("nan")
    se = float(np.std(valid, ddof=1))
    lo, hi = float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))
    return theta_hat, se, lo, hi


def make_plr_ccddhnr_2018(n: int, dim_x: int = 20, alpha: float = 0.5,
                          rho: float = 0.7, s1: float = 1.0, s2: float = 1.0,
                          a0: float = 1.0, a1: float = 0.25,
                          b0: float = 1.0, b1: float = 0.25,
                          rng: np.random.Generator = None):
    """Canonical PLR DGP from Chernozhukov et al. 2018 EJ Figure 1.

    X ~ N(0, Sigma) with AR(1) correlation rho.
    D = m_0(X) + s_1 * v,   v ~ N(0,1)
    Y = alpha * D + g_0(X) + s_2 * zeta,  zeta ~ N(0,1)
    m_0(x) = a_0 x_1 + a_1 sigmoid(x_3)
    g_0(x) = b_0 sigmoid(x_1) + b_1 x_3

    Truth: theta_0 = alpha (analytic).
    """
    rng = rng or np.random.default_rng()
    idx = np.arange(dim_x)
    Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
    L = np.linalg.cholesky(Sigma)
    X = rng.standard_normal((n, dim_x)) @ L.T
    v = rng.standard_normal(n)
    zeta = rng.standard_normal(n)
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))
    m0 = a0 * X[:, 0] + a1 * sig(X[:, 2])
    g0 = b0 * sig(X[:, 0]) + b1 * X[:, 2]
    D = m0 + s1 * v
    Y = alpha * D + g0 + s2 * zeta
    return Y, D, X


def naive_ml_plugin(Y, D, X, seed=42):
    """GBM regression of Y on [D, X]; coefficient on D is the naive estimate.

    GBM does not have an analytic coef on D so we use a two-step: GBM
    predicts E[Y | D=0, X], then OLS slope on residual Y - E_hat against D.
    """
    XD0 = np.column_stack([np.zeros_like(D), X])
    XD = np.column_stack([D, X])
    g = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=seed)
    g.fit(XD0, Y)
    Y_resid = Y - g.predict(XD0)
    cov = float(np.cov(D, Y_resid, ddof=1)[0, 1])
    var = float(np.var(D, ddof=1))
    theta = cov / max(var, 1e-12)
    psi = (Y_resid - theta * D) * D / max(var, 1e-12)
    se = float(np.sqrt(np.var(psi, ddof=1) / len(Y)))
    return theta, se


def _pick_alpha(Xs, y, alphas=RIDGE_ALPHAS):
    """Single full-data RidgeCV pick for the ridge nuisance alpha.

    ~2ms at n=2000, p=20. The picked alpha is reused for cross-fit
    folds AND every pairs-bootstrap iteration, so the inner Cholesky
    loop stays fast while the nuisance regularisation matches what
    RidgeCV would have chosen per-fold. Fixes the bias-from-over-
    regularisation problem the v1 fast path produced on the CCDDHNR
    DGP at n=2000 (alpha=10 hardcoded → theta bias +0.067 → coverage
    0.33 in the pairs bootstrap).
    """
    return float(RidgeCV(alphas=alphas).fit(Xs, y).alpha_)


def _ridge_cross_fit(Y, D, X, k_folds=5, seed=42, alpha_y=None, alpha_d=None,
                     use_ridge_cv=True, return_alphas=False):
    """Cross-fit ridge nuisances via per-fold RidgeCV.

    Reverted from the cho_solve fast path (which produced biased theta
    estimates: bias +0.067 vs the reference RidgeCV pipeline at n=2000
    on the CCDDHNR DGP; coverage tanked to 0.33). RidgeCV-per-fold
    matches the published DML simulations and is what the legacy
    pipeline used. Speedup comes from outer-loop joblib parallelism,
    not from skipping the per-fold CV.
    """
    n, p = X.shape
    Xs = StandardScaler().fit_transform(X)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    Y_res = np.empty(n); D_res = np.empty(n)
    picked_alpha_y = []
    picked_alpha_d = []
    for tr, te in kf.split(np.arange(n)):
        m_y = RidgeCV(alphas=RIDGE_ALPHAS).fit(Xs[tr], Y[tr])
        m_d = RidgeCV(alphas=RIDGE_ALPHAS).fit(Xs[tr], D[tr])
        Y_res[te] = Y[te] - m_y.predict(Xs[te])
        D_res[te] = D[te] - m_d.predict(Xs[te])
        picked_alpha_y.append(float(m_y.alpha_))
        picked_alpha_d.append(float(m_d.alpha_))
    if return_alphas:
        return Y_res, D_res, float(np.median(picked_alpha_y)), float(np.median(picked_alpha_d))
    return Y_res, D_res


def dml_if(Y, D, X, k_folds=5, seed=42, return_alphas=False):
    out = _ridge_cross_fit(Y, D, X, k_folds=k_folds, seed=seed,
                            return_alphas=True)
    Y_res, D_res, alpha_y, alpha_d = out
    denom = float(np.mean(D_res ** 2))
    if denom < 1e-12:
        if return_alphas:
            return float("nan"), float("nan"), None, None, alpha_y, alpha_d
        return float("nan"), float("nan"), None, None
    theta = float(np.mean(D_res * Y_res) / denom)
    psi = (Y_res - theta * D_res) * D_res / denom
    se = float(np.sqrt(np.var(psi, ddof=1) / len(Y)))
    if return_alphas:
        return theta, se, Y_res, D_res, alpha_y, alpha_d
    return theta, se, Y_res, D_res


def dml_multiplier_boot(theta_hat, Y_res, D_res, n_boot=500, seed=42):
    """Rademacher multiplier bootstrap on the IF scores (CCDDHNR §4.3).

    Cheap: no nuisance refit. Conditions on eta_hat. Returns bootstrap SE
    and 95% percentile CI.
    """
    n = len(Y_res)
    denom = float(np.mean(D_res ** 2))
    psi = (Y_res - theta_hat * D_res) * D_res / max(denom, 1e-12)
    rng = np.random.default_rng(seed)
    eps = rng.choice([-1.0, 1.0], size=(n_boot, n))
    thetas = theta_hat + (eps * psi).mean(axis=1)
    se = float(np.std(thetas, ddof=1))
    lo, hi = float(np.percentile(thetas, 2.5)), float(np.percentile(thetas, 97.5))
    return theta_hat, se, lo, hi


def dml_pairs_boot(Y, D, X, n_boot=200, k_folds=5, seed=42,
                    alpha_y=None, alpha_d=None):
    """Nonparametric pairs bootstrap with full DML refit per resample.

    Resample (Y_i, D_i, X_i) triples with replacement, refit full
    RidgeCV-per-fold cross-fit, recover theta_b. Captures nuisance
    estimation uncertainty by re-CV'ing alpha per bootstrap fold (the
    only way to match the original-sample theta_hat in bias).
    """
    n = len(Y)
    base = dml_if(Y, D, X, k_folds=k_folds, seed=seed)
    theta_hat = base[0]
    if np.isnan(theta_hat):
        return theta_hat, float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    thetas = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        Y_res, D_res = _ridge_cross_fit(
            Y[idx], D[idx], X[idx], k_folds=k_folds, seed=seed + b + 1,
        )
        denom = float(np.mean(D_res ** 2))
        if denom < 1e-12:
            continue
        thetas[b] = float(np.mean(D_res * Y_res) / denom)
    valid = thetas[~np.isnan(thetas)]
    if valid.size < max(5, n_boot // 4):
        return theta_hat, float("nan"), float("nan"), float("nan")
    se = float(np.std(valid, ddof=1))
    lo = float(np.percentile(valid, 2.5))
    hi = float(np.percentile(valid, 97.5))
    return theta_hat, se, lo, hi


def dml_fixed_nuisance_boot(Y_res, D_res, theta_hat, n_boot=500, seed=42):
    """Fixed-nuisance bootstrap (Tang & Westling 2024 arXiv 2404.03064).

    Holds the cross-fit nuisance estimates at their original-sample
    values; resamples (Y_res_i, D_res_i) pairs with replacement and
    recomputes theta_b from the orthogonal score. Avoids cross-fit
    re-estimation on duplicated bootstrap rows (their footnote warns
    that empirical bootstrap with cross-validated nuisance produces
    biased CIs because duplicates appear in train and test folds).
    Asymptotically valid for the partially-linear score per their
    Theorem 1; computationally ~10x cheaper than full-refit pairs
    bootstrap because the ridge fits are not redone.
    """
    n = len(Y_res)
    rng = np.random.default_rng(seed)
    thetas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        Yr, Dr = Y_res[idx], D_res[idx]
        denom = float(np.mean(Dr ** 2))
        thetas[b] = float(np.mean(Dr * Yr) / denom) if denom > 1e-12 else np.nan
    valid = thetas[~np.isnan(thetas)]
    if valid.size < max(5, n_boot // 4):
        return theta_hat, float("nan"), float("nan"), float("nan")
    se = float(np.std(valid, ddof=1))
    lo = float(np.percentile(valid, 2.5))
    hi = float(np.percentile(valid, 97.5))
    return theta_hat, se, lo, hi


@dataclass
class RepResult:
    n: int
    theta_true: float
    rep: int
    estimator: str
    theta_hat: float
    se: float
    ci_low: float
    ci_high: float
    covers: bool


def _run_cell_worker(args_pickle, n, theta, q):
    """Cell worker for the multiprocessing spawn dispatcher. Runs one full
    (n, theta) cell sequentially in its own Python interpreter, sending
    the resulting rows back via a Queue. Used only when
    `--parallel_cells > 1`; the sequential / loky / threading paths
    use the in-process dispatch in `main()`.
    """
    import pickle as _pickle
    import time as _time
    args = _pickle.loads(args_pickle)
    rows = []
    seeds = [(rep, args["seed"] + 1009 * rep + 41 * int(n) + 13 * int(theta * 100))
             for rep in range(args["n_reps"])]
    t0 = _time.time()
    for rep, seed in seeds:
        rows.extend(run_one_rep(
            n, theta, rep, seed,
            args["n_boot_mult"], args["n_boot_pairs"],
            args["n_boot_fixed"], args["k_folds"],
            dgp=args["dgp"],
            calibration=args["calibration"],
            learner_outcome=args["learner_outcome"],
            learner_propensity=args["learner_propensity"],
        ))
    q.put((n, theta, _time.time() - t0, rows))


def run_one_rep(n, theta_true, rep, seed, n_boot_mult, n_boot_pairs,
                n_boot_fixed, k_folds, dgp="ccddhnr",
                calibration="isotonic",
                learner_outcome="ridge", learner_propensity="logreg"):
    rng = np.random.default_rng(seed)
    if dgp == "ccddhnr":
        Y, D, X = make_plr_ccddhnr_2018(n=n, alpha=theta_true, rng=rng)
    elif dgp == "weak_overlap":
        Y, D, X = make_plr_weak_overlap(n=n, alpha=theta_true, rng=rng)
    elif dgp == "irm_dgp4":
        Y, D, X = make_irm_dgp4(n=n, theta=theta_true, rng=rng)
        rows = []
        theta_if, se_if, psi = aipw_if(
            Y, D, X, k_folds=k_folds, seed=seed,
            learner_outcome=learner_outcome,
            learner_propensity=learner_propensity,
        )
        lo_if, hi_if = theta_if - 1.96 * se_if, theta_if + 1.96 * se_if
        rows.append(RepResult(n, theta_true, rep, "dml_if", theta_if, se_if,
                              lo_if, hi_if, lo_if <= theta_true <= hi_if))
        if n_boot_mult > 0:
            _, se_mb, lo_mb, hi_mb = aipw_mult_boot(theta_if, psi,
                                                     n_boot=n_boot_mult,
                                                     seed=seed + 7919)
            rows.append(RepResult(n, theta_true, rep, "dml_mult_boot",
                                  theta_if, se_mb, lo_mb, hi_mb,
                                  lo_mb <= theta_true <= hi_mb))
        if n_boot_pairs > 0:
            _, se_pb, lo_pb, hi_pb = aipw_pairs_boot(
                Y, D, X, n_boot=n_boot_pairs, k_folds=k_folds,
                seed=seed + 31337,
                learner_outcome=learner_outcome,
                learner_propensity=learner_propensity,
            )
            rows.append(RepResult(n, theta_true, rep, "dml_pairs_boot",
                                  theta_if, se_pb, lo_pb, hi_pb,
                                  lo_pb <= theta_true <= hi_pb))
        theta_cal, se_cal, psi_cal = aipw_cal_if(
            Y, D, X, k_folds=K_FOLDS_CAL, seed=seed, calibration=calibration,
            learner_outcome=learner_outcome,
            learner_propensity=learner_propensity,
        )
        lo_cal, hi_cal = theta_cal - 1.96 * se_cal, theta_cal + 1.96 * se_cal
        rows.append(RepResult(n, theta_true, rep, "dml_cal_if",
                              theta_cal, se_cal, lo_cal, hi_cal,
                              lo_cal <= theta_true <= hi_cal))
        if n_boot_mult > 0:
            _, se_cmb, lo_cmb, hi_cmb = aipw_mult_boot(
                theta_cal, psi_cal, n_boot=n_boot_mult, seed=seed + 7919,
            )
            rows.append(RepResult(n, theta_true, rep, "dml_cal_mult_boot",
                                  theta_cal, se_cmb, lo_cmb, hi_cmb,
                                  lo_cmb <= theta_true <= hi_cmb))
        if n_boot_pairs > 0:
            _, se_cpb, lo_cpb, hi_cpb = aipw_cal_pairs_boot(
                Y, D, X, n_boot=n_boot_pairs, k_folds=K_FOLDS_CAL,
                seed=seed + 31337, calibration=calibration,
                learner_outcome=learner_outcome,
                learner_propensity=learner_propensity,
            )
            rows.append(RepResult(n, theta_true, rep, "dml_cal_pairs_boot",
                                  theta_cal, se_cpb, lo_cpb, hi_cpb,
                                  lo_cpb <= theta_true <= hi_cpb))
        return rows
    else:
        raise ValueError(f"unknown dgp: {dgp}")
    rows = []
    th_n, se_n = naive_ml_plugin(Y, D, X, seed=seed)
    lo_n, hi_n = th_n - 1.96 * se_n, th_n + 1.96 * se_n
    rows.append(RepResult(n, theta_true, rep, "naive_ml", th_n, se_n,
                          lo_n, hi_n, lo_n <= theta_true <= hi_n))

    theta_if, se_if, Y_res, D_res = dml_if(Y, D, X, k_folds=k_folds, seed=seed)
    lo_if, hi_if = theta_if - 1.96 * se_if, theta_if + 1.96 * se_if
    rows.append(RepResult(n, theta_true, rep, "dml_if", theta_if, se_if,
                          lo_if, hi_if, lo_if <= theta_true <= hi_if))

    if n_boot_mult > 0:
        _, se_mb, lo_mb, hi_mb = dml_multiplier_boot(theta_if, Y_res, D_res,
                                                      n_boot=n_boot_mult,
                                                      seed=seed + 7919)
        rows.append(RepResult(n, theta_true, rep, "dml_mult_boot", theta_if,
                              se_mb, lo_mb, hi_mb, lo_mb <= theta_true <= hi_mb))

    if n_boot_fixed > 0:
        _, se_fn, lo_fn, hi_fn = dml_fixed_nuisance_boot(Y_res, D_res, theta_if,
                                                          n_boot=n_boot_fixed,
                                                          seed=seed + 50021)
        rows.append(RepResult(n, theta_true, rep, "dml_fixed_nuis_boot",
                              theta_if, se_fn, lo_fn, hi_fn,
                              lo_fn <= theta_true <= hi_fn))

    if n_boot_pairs > 0:
        _, se_pb, lo_pb, hi_pb = dml_pairs_boot(Y, D, X, n_boot=n_boot_pairs,
                                                 k_folds=k_folds, seed=seed + 31337)
        rows.append(RepResult(n, theta_true, rep, "dml_pairs_boot", theta_if,
                              se_pb, lo_pb, hi_pb, lo_pb <= theta_true <= hi_pb))
    return rows


def summarize(df):
    """Per-cell summary: bias, RMSE, avg SE, sd(theta_hat), coverage,
    mean CI width. One row per (n, theta_true, estimator) cell."""
    out = []
    for (n, t, est), g in df.groupby(["n", "theta_true", "estimator"]):
        th = g["theta_hat"].values
        se = g["se"].values
        cov = g["covers"].mean()
        width = (g["ci_high"] - g["ci_low"]).mean()
        out.append({
            "n": int(n), "theta_true": float(t), "estimator": est,
            "n_reps": int(len(g)),
            "bias": float(np.mean(th - t)),
            "rmse": float(np.sqrt(np.mean((th - t) ** 2))),
            "avg_se": float(np.mean(se)),
            "sd_theta_hat": float(np.std(th, ddof=1)),
            "coverage_95": float(cov),
            "mean_ci_width": float(width),
        })
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_grid", type=int, nargs="*",
                    default=[500, 1000, 2000])
    ap.add_argument("--theta_grid", type=float, nargs="*",
                    default=[0.0, 0.5])
    ap.add_argument("--n_reps", type=int, default=200)
    ap.add_argument("--n_boot_mult", type=int, default=500)
    ap.add_argument("--n_boot_pairs", type=int, default=100,
                    help="pairs-bootstrap iterations; expensive (full DML "
                         "refit each); set 0 to skip")
    ap.add_argument("--n_boot_fixed", type=int, default=500,
                    help="fixed-nuisance bootstrap iterations (Tang & "
                         "Westling 2024 arXiv 2404.03064); cheap, set 0 to skip")
    ap.add_argument("--joblib_n_jobs", type=int, default=-1,
                    help="joblib outer-loop parallelism; -1 uses all cores "
                         "(safe now that RidgeCV's inner CV deadlock is "
                         "avoided via the fixed-alpha Cholesky path)")
    ap.add_argument("--chunk", type=int, default=8,
                    help="reps per joblib task (avoids per-rep dispatch "
                         "overhead; tasks should be >100ms each)")
    ap.add_argument("--use_ridge_cv_for_point", action="store_true",
                    help="use RidgeCV (slow) only for the headline dml_if "
                         "point estimate; bootstrap loops always use fixed-α")
    ap.add_argument("--dgp", choices=["ccddhnr", "weak_overlap", "irm_dgp4"],
                    default="ccddhnr",
                    help="DGP family. weak_overlap is a PLR with R²(D|X)≈0.90; "
                         "irm_dgp4 mirrors Ballinari & Bearth 2024 (arXiv "
                         "2409.04874) DGP4 — binary treatment + extreme PS + "
                         "nonlinear outcome → published IF-SE coverage 0.28 "
                         "with lasso, 0.81 with RF.")
    ap.add_argument("--k_folds", type=int, default=5)
    ap.add_argument("--parallel_cells", type=int, default=1,
                    help="If > 1, dispatch each (n, theta) cell to its "
                         "own spawn-method multiprocessing.Process. Each "
                         "child gets a fresh Python interpreter + fresh "
                         "libgomp, bypassing the loky / OpenMP deadlocks "
                         "that hang shared-process backends in cgroup "
                         "containers (Brev/shadeform). ~4x speedup with "
                         "4-6 cells on a 32-vCPU box. Inside each child, "
                         "reps run sequentially.")
    ap.add_argument("--backend", choices=["loky", "threading", "sequential"],
                    default="loky",
                    help="joblib backend. Default 'loky' (process pool). "
                         "Use 'threading' when loky deadlocks at worker "
                         "spawn (Brev-style containers with fork+OpenMP "
                         "issues); RidgeCV/LogisticRegressionCV release "
                         "GIL during inner BLAS, so threading is "
                         "~30% slower but never hangs. Use 'sequential' "
                         "for single-threaded debugging.")
    ap.add_argument("--learner_outcome",
                    choices=["ridge", "hgbm"], default="ridge",
                    help="Outcome regression learner for the IRM/AIPW path. "
                         "'ridge' (default) is linear, fast, and matches "
                         "our applied §6 BERT-PC1 ridge nuisance. 'hgbm' is "
                         "sklearn HistGradientBoostingRegressor — flexible "
                         "enough to capture the Friedman 1991 surface in "
                         "DGP4; ~2-3x slower than ridge.")
    ap.add_argument("--learner_propensity",
                    choices=["logreg", "hgbm"], default="logreg",
                    help="Propensity classifier for the IRM/AIPW path. "
                         "'logreg' (default) is L2-regularized logistic "
                         "regression and is already well-calibrated. "
                         "'hgbm' is sklearn HistGradientBoostingClassifier "
                         "— more flexible but typically miscalibrated, so "
                         "pairs well with --calibration venn_abers.")
    ap.add_argument("--calibration",
                    choices=["venn_abers", "isotonic", "platt", "beta"],
                    default="venn_abers",
                    help="Propensity-score calibration method appended "
                         "to the IRM panel per Ballinari & Bearth (2024) "
                         "Algorithm 1. Venn-Abers (default) is B&B's "
                         "headline recommendation and is asymptotically "
                         "licensed (Theorem 1); isotonic is parameter-"
                         "free but unstable on small inner sub-folds; "
                         "Platt is NOT asymptotically licensed (see B&B "
                         "Theorem 1 footnote). The calibrated path uses "
                         "K_FOLDS_CAL=2 outer folds (B&B's default) "
                         "regardless of --k_folds.")
    ap.add_argument("--seed", type=int, default=20260601)
    ap.add_argument("--out_dir", type=Path,
                    default=REPO / "results" / "simulation_jbes2026")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.dgp}" if args.dgp != "ccddhnr" else ""
    out_csv = args.out_dir / f"coverage_raw{suffix}.csv"
    summary_csv = args.out_dir / f"coverage_summary{suffix}.csv"

    print(f"DGP={args.dgp}  Grid: n in {args.n_grid}, theta in {args.theta_grid}, "
          f"reps={args.n_reps}, mult_boot={args.n_boot_mult}, "
          f"pairs_boot={args.n_boot_pairs}, fixed_nuis_boot={args.n_boot_fixed}, "
          f"joblib_n_jobs={args.joblib_n_jobs}, chunk={args.chunk}, "
          f"backend={args.backend}, calibration={args.calibration}, "
          f"learner_outcome={args.learner_outcome}, "
          f"learner_propensity={args.learner_propensity}")
    from joblib import Parallel, delayed

    def _rep_chunk(seeds_with_reps, n, theta):
        out = []
        for rep, seed in seeds_with_reps:
            out.extend(run_one_rep(n, theta, rep, seed,
                                    args.n_boot_mult, args.n_boot_pairs,
                                    args.n_boot_fixed, args.k_folds,
                                    dgp=args.dgp,
                                    calibration=args.calibration,
                                    learner_outcome=args.learner_outcome,
                                    learner_propensity=args.learner_propensity))
        return out

    t0 = time.time()
    all_rows = []
    cells = [(n, theta) for n in args.n_grid for theta in args.theta_grid]

    if args.parallel_cells > 1:
        import multiprocessing as _mp
        import pickle as _pickle
        ctx = _mp.get_context("spawn")
        q = ctx.Queue()
        args_pickle = _pickle.dumps(vars(args))
        print(f"  launching {len(cells)} cells as spawn processes "
              f"(parallel_cells={args.parallel_cells})", flush=True)
        procs = []
        for (n, theta) in cells:
            p = ctx.Process(target=_run_cell_worker,
                             args=(args_pickle, n, theta, q))
            p.start()
            procs.append(p)
        for _ in cells:
            n_done, theta_done, dt, cell_rows = q.get()
            all_rows.extend(cell_rows)
            print(f"  cell n={n_done} theta={theta_done} done in {dt:.1f}s "
                  f"({len(cell_rows)} rep-estimator rows)", flush=True)
        for p in procs:
            p.join()
    else:
        for n, theta in cells:
            cell_t = time.time()
            seeds = [(rep, args.seed + 1009 * rep + 41 * int(n) + 13 * int(theta * 100))
                     for rep in range(args.n_reps)]
            chunks = [seeds[i:i + args.chunk]
                      for i in range(0, len(seeds), args.chunk)]
            print(f"  [n={n}, theta={theta}] launching {len(chunks)} chunks "
                  f"of {args.chunk} reps each across n_jobs={args.joblib_n_jobs} "
                  f"(backend={args.backend})", flush=True)
            if args.backend == "sequential":
                chunk_outputs = [_rep_chunk(c, n, theta) for c in chunks]
            else:
                chunk_outputs = Parallel(n_jobs=args.joblib_n_jobs,
                                         backend=args.backend, verbose=5)(
                    delayed(_rep_chunk)(c, n, theta) for c in chunks
                )
            cell_rows = [r for chunk in chunk_outputs for r in chunk]
            all_rows.extend(cell_rows)
            print(f"  cell n={n} theta={theta} done in {time.time()-cell_t:.1f}s "
                  f"({len(cell_rows)} rep-estimator rows)")

    df = pd.DataFrame([asdict(r) for r in all_rows])
    df.to_csv(out_csv, index=False)
    summary = summarize(df)
    summary.to_csv(summary_csv, index=False)

    print(f"\n=== Coverage summary table (total runtime {time.time()-t0:.0f}s) ===")
    print(summary.to_string(index=False,
                            float_format=lambda x: f"{x:.4f}"))
    print(f"\nRaw -> {out_csv}")
    print(f"Summary -> {summary_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
