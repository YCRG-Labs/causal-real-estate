"""City-moderator meta-regression of per-city DML estimates.

Per the methodology research agent's recipe (Borenstein 2009; Viechtbauer
2010 metafor; Knapp-Hartung 2003; Higgins-Thompson 2004; Veroniki et al.
2016; van Lissa et al. 2023 BRMA):

  - Mixed-effects meta-regression with REML τ² estimation
  - Knapp-Hartung small-sample SE adjustment (mandatory at k=12)
  - Three specifications:
      (a) Intercept-only RE: baseline τ² + I² (heterogeneity quantification)
      (b) Theory-driven m=3: log_population, pct_renter, pop_density_per_sqmi
          (pre-registered by Borenstein "k ≥ 10 per moderator" rule)
      (c) Univariate scan of all 8 moderators with BH FDR correction
  - Report: β̂_j, KH-adjusted SE, t-statistic (df=k-p), 95% CI from t
           QM omnibus F-test, residual τ², R²_meta, residual I²

The treatment-effect inputs come from the Shen 12-city bootstrap CIs
(shen_12city_table.csv), with sampling variance v_i = (bootstrap CI half-
width / 1.96)². The headline reading the JBES audience expects is whether
any city-level covariate explains a non-trivial share of the I² ≈ 95%
between-city heterogeneity documented in §6.1.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[3]


def _knha_meta_reml(y: np.ndarray, v: np.ndarray, X: np.ndarray | None,
                    max_iter: int = 200, tol: float = 1e-7):
    """REML τ² + Knapp-Hartung adjusted inference for a mixed-effects
    meta-regression.

    Inputs:
      y : (k,) array of study/city effect estimates
      v : (k,) array of within-study sampling variances (bootstrap SE²)
      X : (k, p) design matrix INCLUDING the intercept column, or None
          (in which case it is intercept-only)

    Returns dict with β, KH-SE, t-stats, CIs, df, residual τ², I², QM, and
    R²_meta computed against the intercept-only τ̂² baseline.
    """
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    k = len(y)
    if X is None:
        X = np.ones((k, 1))
    else:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
    p = X.shape[1]

    # REML iteration on τ². Closed-form-ish update from Viechtbauer 2005.
    tau2 = max(0.0, float(np.var(y, ddof=1) - np.mean(v)))
    for _ in range(max_iter):
        w = 1.0 / (v + tau2)
        W = np.diag(w)
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ y
        beta = np.linalg.solve(XtWX, XtWy)
        resid = y - X @ beta
        # REML stationary-point update for tau2 (Viechtbauer 2005 Eq. 8).
        P = W - W @ X @ np.linalg.solve(XtWX, X.T @ W)
        num = float(resid @ (P @ P) @ resid - np.trace(P))
        den = float(np.trace(P @ P))
        if den <= 0:
            break
        delta = num / den
        new_tau2 = max(0.0, tau2 + delta)
        if abs(new_tau2 - tau2) < tol:
            tau2 = new_tau2
            break
        tau2 = new_tau2

    # Final fit with refined tau2.
    w = 1.0 / (v + tau2)
    W = np.diag(w)
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y
    beta = np.linalg.solve(XtWX, XtWy)
    resid = y - X @ beta

    # Knapp-Hartung quadratic-form variance scaling.
    q_kh = float(resid @ (w * resid)) / max(k - p, 1)
    cov_kh = q_kh * np.linalg.inv(XtWX)
    se_kh = np.sqrt(np.maximum(np.diag(cov_kh), 0))
    df = max(k - p, 1)
    t_stats = beta / np.where(se_kh > 0, se_kh, np.inf)
    ci_half = stats.t.ppf(0.975, df) * se_kh
    ci_low = beta - ci_half
    ci_high = beta + ci_half
    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), df))

    # Q_E residual heterogeneity, I² residual.
    Q_E = float(resid @ (w * resid))
    df_Q = k - p
    I2_resid = max(0.0, (Q_E - df_Q) / max(Q_E, 1e-12)) * 100.0

    # Omnibus F-test for moderators (against intercept-only).
    if p > 1 and X[:, 0].std() == 0:  # column 0 is intercept
        non_intercept = list(range(1, p))
        if non_intercept:
            beta_sub = beta[non_intercept]
            cov_sub = cov_kh[np.ix_(non_intercept, non_intercept)]
            try:
                F = float(beta_sub @ np.linalg.solve(cov_sub, beta_sub) / len(non_intercept))
                F_p = 1 - stats.f.cdf(F, len(non_intercept), df)
            except np.linalg.LinAlgError:
                F = float("nan"); F_p = float("nan")
        else:
            F = float("nan"); F_p = float("nan")
    else:
        F = float("nan"); F_p = float("nan")

    return {
        "beta": beta, "se_kh": se_kh, "t": t_stats, "df": df,
        "ci_low": ci_low, "ci_high": ci_high, "p": p_vals,
        "tau2": float(tau2), "Q_E": Q_E, "I2_resid_pct": float(I2_resid),
        "F_omnibus": F, "F_omnibus_p": F_p,
        "cov_kh": cov_kh,
    }


def bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg q-values for a vector of p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = np.minimum.accumulate((ranked * n / (np.arange(n) + 1))[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shen",
                    default=str(REPO / "results" / "replications" / "shen_12city_table.csv"))
    ap.add_argument("--acs",
                    default=str(REPO / "results" / "replications" / "acs_moderators_12city.csv"))
    ap.add_argument("--out",
                    default=str(REPO / "results" / "replications" / "meta_regression_12.csv"))
    args = ap.parse_args()

    shen = pd.read_csv(args.shen).set_index("city")
    acs = pd.read_csv(args.acs).set_index("city")
    df = shen.join(acs, how="inner")

    # Effect estimates and within-study sampling variances. The Shen
    # bootstrap CI half-widths give us robust SEs that don't inherit the
    # IF-SE quasi-undercoverage documented in §3-4.
    theta = df["dml_theta"].values
    ci_half = (df["dml_ci_high"].values - df["dml_ci_low"].values) / 2.0
    se_boot = ci_half / 1.96
    v = se_boot ** 2

    moderators_full = [
        "log_population", "median_hh_income", "pct_renter", "pct_college",
        "diversity_gini_simpson", "median_home_value", "median_year_built",
        "pop_density_per_sqmi",
    ]
    moderators_full = [m for m in moderators_full if m in df.columns]
    Z_full = df[moderators_full].apply(
        lambda c: (c - c.mean()) / (c.std() if c.std() > 0 else 1.0)
    )

    print(f"\n=== Inputs: k={len(df)} cities, {len(moderators_full)} candidate moderators ===")
    print(df[["dml_theta", "dml_ci_low", "dml_ci_high"]].assign(se_boot=se_boot).to_string())

    # (a) Intercept-only random-effects baseline.
    print("\n=== (a) Intercept-only random-effects baseline ===")
    base = _knha_meta_reml(theta, v, None)
    tau2_0 = base["tau2"]
    print(f"  pooled θ̂_RE = {base['beta'][0]:+.4f}  KH-SE = {base['se_kh'][0]:.4f}  "
          f"t({base['df']}) = {base['t'][0]:+.2f}  p = {base['p'][0]:.3f}")
    print(f"  95% CI = [{base['ci_low'][0]:+.4f}, {base['ci_high'][0]:+.4f}]")
    print(f"  τ̂² = {tau2_0:.5f}  Q_E = {base['Q_E']:.2f}  "
          f"residual I² = {base['I2_resid_pct']:.1f}%")

    # (b) Theory-driven multivariable m=3.
    primary_mods = ["log_population", "pct_renter", "pop_density_per_sqmi"]
    primary_mods = [m for m in primary_mods if m in moderators_full]
    if primary_mods:
        print(f"\n=== (b) Theory-driven multivariable (m={len(primary_mods)}) ===")
        X_pri = np.column_stack([np.ones(len(df)), Z_full[primary_mods].values])
        pri = _knha_meta_reml(theta, v, X_pri)
        names = ["intercept"] + primary_mods
        R2_meta = max(0.0, (tau2_0 - pri["tau2"])) / max(tau2_0, 1e-12)
        for i, n in enumerate(names):
            print(f"  {n:<22} β = {pri['beta'][i]:+.4f}  "
                  f"KH-SE = {pri['se_kh'][i]:.4f}  "
                  f"t({pri['df']}) = {pri['t'][i]:+.2f}  "
                  f"p = {pri['p'][i]:.3f}  "
                  f"95% CI [{pri['ci_low'][i]:+.4f}, {pri['ci_high'][i]:+.4f}]")
        print(f"  Q_E_residual = {pri['Q_E']:.2f}  τ̂²_residual = {pri['tau2']:.5f}  "
              f"residual I² = {pri['I2_resid_pct']:.1f}%  R²_meta = {R2_meta*100:.1f}%")
        if not math.isnan(pri["F_omnibus"]):
            print(f"  Omnibus F({len(primary_mods)}, {pri['df']}) = "
                  f"{pri['F_omnibus']:.2f}  p = {pri['F_omnibus_p']:.3f}")

    # (c) Univariate scan with BH q-values.
    print("\n=== (c) Univariate scan of all candidate moderators (BH FDR) ===")
    uni_rows = []
    for mod in moderators_full:
        X_uni = np.column_stack([np.ones(len(df)), Z_full[mod].values])
        res = _knha_meta_reml(theta, v, X_uni)
        R2 = max(0.0, (tau2_0 - res["tau2"])) / max(tau2_0, 1e-12)
        uni_rows.append({
            "moderator": mod,
            "beta": res["beta"][1], "kh_se": res["se_kh"][1],
            "t": res["t"][1], "p": res["p"][1],
            "ci_low": res["ci_low"][1], "ci_high": res["ci_high"][1],
            "tau2_resid": res["tau2"], "R2_meta": R2,
            "I2_resid_pct": res["I2_resid_pct"],
        })
    uni_df = pd.DataFrame(uni_rows)
    uni_df["bh_q"] = bh_fdr(uni_df["p"].values)
    uni_df = uni_df.sort_values("p")
    print(uni_df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    uni_df.to_csv(out_path, index=False)
    print(f"\nCSV -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
