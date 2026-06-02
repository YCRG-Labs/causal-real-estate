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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


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


def _ridge_cross_fit(Y, D, X, k_folds=5, seed=42):
    n = len(Y)
    Xs = StandardScaler().fit_transform(X)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    Y_res = np.empty(n); D_res = np.empty(n)
    for tr, te in kf.split(np.arange(n)):
        m_y = RidgeCV(alphas=RIDGE_ALPHAS).fit(Xs[tr], Y[tr])
        m_d = RidgeCV(alphas=RIDGE_ALPHAS).fit(Xs[tr], D[tr])
        Y_res[te] = Y[te] - m_y.predict(Xs[te])
        D_res[te] = D[te] - m_d.predict(Xs[te])
    return Y_res, D_res


def dml_if(Y, D, X, k_folds=5, seed=42):
    Y_res, D_res = _ridge_cross_fit(Y, D, X, k_folds=k_folds, seed=seed)
    denom = float(np.mean(D_res ** 2))
    if denom < 1e-12:
        return float("nan"), float("nan"), None, None
    theta = float(np.mean(D_res * Y_res) / denom)
    psi = (Y_res - theta * D_res) * D_res / denom
    se = float(np.sqrt(np.var(psi, ddof=1) / len(Y)))
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


def dml_pairs_boot(Y, D, X, n_boot=200, k_folds=5, seed=42):
    """Nonparametric pairs bootstrap with full DML refit per resample.

    Resample (Y_i, D_i, X_i) triples with replacement, refit cross-fit
    ridge nuisances, recover theta_b. Captures nuisance uncertainty.
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
        out = dml_if(Y[idx], D[idx], X[idx], k_folds=k_folds, seed=seed + b + 1)
        if not np.isnan(out[0]):
            thetas[b] = out[0]
    valid = thetas[~np.isnan(thetas)]
    if valid.size < max(20, n_boot // 5):
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


def run_one_rep(n, theta_true, rep, seed, n_boot_mult, n_boot_pairs, k_folds):
    rng = np.random.default_rng(seed)
    Y, D, X = make_plr_ccddhnr_2018(n=n, alpha=theta_true, rng=rng)
    rows = []
    th_n, se_n = naive_ml_plugin(Y, D, X, seed=seed)
    lo_n, hi_n = th_n - 1.96 * se_n, th_n + 1.96 * se_n
    rows.append(RepResult(n, theta_true, rep, "naive_ml", th_n, se_n,
                          lo_n, hi_n, lo_n <= theta_true <= hi_n))

    theta_if, se_if, Y_res, D_res = dml_if(Y, D, X, k_folds=k_folds, seed=seed)
    lo_if, hi_if = theta_if - 1.96 * se_if, theta_if + 1.96 * se_if
    rows.append(RepResult(n, theta_true, rep, "dml_if", theta_if, se_if,
                          lo_if, hi_if, lo_if <= theta_true <= hi_if))

    _, se_mb, lo_mb, hi_mb = dml_multiplier_boot(theta_if, Y_res, D_res,
                                                  n_boot=n_boot_mult,
                                                  seed=seed + 7919)
    rows.append(RepResult(n, theta_true, rep, "dml_mult_boot", theta_if,
                          se_mb, lo_mb, hi_mb, lo_mb <= theta_true <= hi_mb))

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
                         "refit each), default 100, paper-grade 200-500")
    ap.add_argument("--k_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260601)
    ap.add_argument("--out_dir", type=Path,
                    default=REPO / "results" / "simulation_jbes2026")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "coverage_raw.csv"
    summary_csv = args.out_dir / "coverage_summary.csv"

    print(f"Grid: n in {args.n_grid}, theta in {args.theta_grid}, "
          f"reps={args.n_reps}, mult_boot={args.n_boot_mult}, "
          f"pairs_boot={args.n_boot_pairs}")
    t0 = time.time()
    all_rows = []
    for n in args.n_grid:
        for theta in args.theta_grid:
            cell_t = time.time()
            cell_rows = []
            for rep in range(args.n_reps):
                seed = args.seed + 1009 * rep + 41 * int(n) + 13 * int(theta * 100)
                rows = run_one_rep(n, theta, rep, seed,
                                    args.n_boot_mult, args.n_boot_pairs,
                                    args.k_folds)
                cell_rows.extend(rows)
                if (rep + 1) % max(1, args.n_reps // 10) == 0:
                    elapsed = time.time() - cell_t
                    print(f"  [n={n}, theta={theta}] rep {rep+1}/{args.n_reps} "
                          f"({elapsed:.1f}s elapsed)")
            all_rows.extend(cell_rows)
            print(f"  cell n={n} theta={theta} done in {time.time()-cell_t:.1f}s")

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
