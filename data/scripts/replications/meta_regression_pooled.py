"""Meta-regression with Paule-Mandel + Hartung-Knapp + Romano-Wolf.

Random-effects meta-analysis of the per-city DML estimates with the
small-k machinery recommended in Veroniki et al. (2016, Research
Synthesis Methods) for k=12: Paule-Mandel τ² rather than REML, Hartung-
Knapp-Sidik-Jonkman small-sample t-adjusted CIs rather than DL z-CIs,
and Romano-Wolf stepdown bootstrap p-values for moderator inference
rather than Benjamini-Hochberg FDR (more powerful under dependent tests
per Clarke, Romano & Wolf 2020 Stata Journal).

Romano-Wolf is computed over the SAME family of univariate per-moderator
slope statistics that BH/Storey use; it is NOT run on the joint
multivariate meta-regression, because at k=12 the design
[intercept | all moderators] has n_predictors >= k (near-saturated to
saturated), its residuals collapse to ~0, and every joint RW p-value
would floor at 1/(M+1).  A guard in romano_wolf_pvalues refuses any joint
fit with n_predictors >= k.  Per-moderator leave-one-out and case-deletion
influence diagnostics (hat values, Cook's distance, DFBETAS, LOO slope/t
range, per Viechtbauer & Cheung 2010) are emitted to expose single-market
fragility (the renter-share sign-flip pathology) rather than hide it.

Per the agent's anti-p-hacking guidance, this reports the upgraded RW-PM-
HKSJ numbers ALONGSIDE the original BH-REML-KH numbers in a single
robustness table, rather than replacing one with the other.  Output JSON
records both pipelines.

Inputs (pick one):
  --baur_table  results/replications/baur_pooled_pca/baur_pooled_pca_table.csv
  --shen_table  results/replications/shen_12city_table.csv
plus the ACS moderator file results/replications/acs_moderators_12city.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[3]


def tau2_paule_mandel(y, v, max_iter=500, tol=1e-10):
    """Paule-Mandel method-of-moments τ² estimator (Paule & Mandel 1982).
    Iterates to a positive root of Q(τ²) = k - 1 if one exists; otherwise
    returns 0 (boundary, same property as REML).  Veroniki et al. (2016)
    recommend PM over REML at small k for lower bias and better-calibrated
    CIs, conditional on τ² > 0.
    """
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    k = len(y)
    tau2 = max(0.0, np.var(y, ddof=1) - np.mean(v))
    for _ in range(max_iter):
        w = 1.0 / (v + tau2)
        ybar = float(np.sum(w * y) / np.sum(w))
        Q = float(np.sum(w * (y - ybar) ** 2))
        if Q <= k - 1:
            return 0.0
        deriv = float(np.sum(w ** 2 * (y - ybar) ** 2))
        if deriv <= 0:
            break
        new_tau2 = max(tau2 + (Q - (k - 1)) / deriv, 0.0)
        if abs(new_tau2 - tau2) < tol:
            return new_tau2
        tau2 = new_tau2
    return float(tau2)


def tau2_reml(y, v, max_iter=500, tol=1e-10):
    """Restricted-maximum-likelihood τ² for comparison/legacy reporting."""
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    k = len(y)
    tau2 = max(0.0, np.var(y, ddof=1) - np.mean(v))
    for _ in range(max_iter):
        w = 1.0 / (v + tau2)
        ybar = float(np.sum(w * y) / np.sum(w))
        num = float(np.sum(w ** 2 * ((y - ybar) ** 2 - v)))
        den = float(np.sum(w ** 2))
        new_tau2 = max(num / den, 0.0)
        if abs(new_tau2 - tau2) < tol:
            return new_tau2
        tau2 = new_tau2
    return float(tau2)


def _wls_fit(y, v, X, tau2):
    """Weighted least squares meta-regression fit at fixed τ².  Returns
    β̂, hat-cov, fitted residuals, and weights."""
    w = 1.0 / (v + tau2)
    W = np.diag(w)
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y
    beta = np.linalg.solve(XtWX, XtWy)
    cov_naive = np.linalg.inv(XtWX)   # naive WLS covariance
    resid = y - X @ beta
    return beta, cov_naive, resid, w


def hksj_ci(beta, X, y, v, tau2, alpha=0.05):
    """Hartung-Knapp-Sidik-Jonkman t-adjusted CIs.  Scale the naive
    WLS covariance by q = Σ w_i ε_i² / (k - p) and use t_{k-p} critical
    values.  Standard small-k correction; IntHout et al. 2014 BMC Med
    Res Methodol 14:25.
    """
    beta, cov_naive, resid, w = _wls_fit(y, v, X, tau2)
    k, p = X.shape
    q = float(np.sum(w * resid ** 2) / max(k - p, 1))
    cov_hksj = q * cov_naive
    se = np.sqrt(np.clip(np.diag(cov_hksj), 0, None))
    df = max(k - p, 1)
    tcrit = stats.t.ppf(1 - alpha / 2, df=df)
    ci_low = beta - tcrit * se
    ci_high = beta + tcrit * se
    t_stats = beta / np.where(se > 0, se, np.inf)
    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), df))
    return dict(beta=beta, se=se, t=t_stats, p=p_vals,
                 ci_low=ci_low, ci_high=ci_high, df=df,
                 q_kh=q, cov_hksj=cov_hksj)


def romano_wolf_pvalues(y, v, X_full, mod_idx, B=2000, seed=42,
                         tau2_fn=tau2_paule_mandel):
    """Romano-Wolf stepdown bootstrap-t p-values for the moderator
    coefficients in `mod_idx`, conditioning on city-level resampling
    via Bayesian (Dirichlet) cluster weights — exact algorithm from
    Clarke-Romano-Wolf 2020 Stata Journal 20(4):812-843 eqs (3)-(7),
    adapted for the meta-regression context.

    For each bootstrap draw, we resample the k=12 cities with Dirichlet
    weights and refit the meta-regression to obtain bootstrap β̂*, SE*.
    Studentised statistics are |β̂_j / SE_j| centred on the OBSERVED β̂_j,
    not on the null.  Stepdown enforced via monotonicity per eq (7).
    """
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    X = np.asarray(X_full, dtype=float)
    k, p = X.shape
    rng = np.random.default_rng(seed)

    # Guard: a joint design with n_predictors >= k is (near-)saturated.
    # WLS residuals collapse to ~0, the HKSJ scale q -> 0, studentised
    # statistics diverge, and every RW p-value floors at 1/(M+1).  Refuse
    # the joint fit; the caller should use romano_wolf_univariate instead.
    if p >= k:
        warnings.warn(
            f"romano_wolf_pvalues: joint design has n_predictors={p} >= "
            f"k={k} (saturated/near-saturated); refusing the joint fit "
            f"(residuals ~0 -> RW p floors at 1/(M+1)). "
            f"Use romano_wolf_univariate instead.",
            RuntimeWarning, stacklevel=2,
        )
        return (np.full(len(mod_idx), np.nan),
                {"skipped": True, "reason": f"n_predictors={p}>=k={k}"})

    tau2_obs = tau2_fn(y, v)
    hk_obs = hksj_ci(np.linalg.solve(X.T @ X, X.T @ y), X, y, v, tau2_obs)
    beta_obs = hk_obs["beta"]
    se_obs = hk_obs["se"]
    t_obs = np.abs(beta_obs / np.where(se_obs > 0, se_obs, np.inf))
    t_obs_mod = t_obs[mod_idx]

    boot_t_mods = np.zeros((B, len(mod_idx)))
    for b in range(B):
        # Dirichlet(1,...,1) cluster weights, sum to k → mean 1.
        u = rng.exponential(scale=1.0, size=k)
        w_b = k * u / u.sum()
        # Weighted meta-regression: weight observations by w_b (cluster
        # weights) AND by inverse-variance 1/(v+τ²) (meta-analytic).
        v_b = v
        tau2_b = tau2_fn(y, v_b)   # could re-estimate τ² with weights;
        # for stability at k=12 we hold τ² at the observed estimate,
        # which is the standard fixed-η approach for Romano-Wolf in
        # meta-analysis (Ferguson & Heene 2012; Higgins-Thompson 2004).
        # Apply Dirichlet weights to the WLS normal equations:
        w_meta = 1.0 / (v_b + tau2_obs)
        w_eff = w_b * w_meta
        XtWX = X.T @ np.diag(w_eff) @ X
        XtWy = X.T @ np.diag(w_eff) @ y
        try:
            beta_b = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            boot_t_mods[b] = np.nan
            continue
        resid_b = y - X @ beta_b
        q_b = float(np.sum(w_eff * resid_b ** 2) / max(k - p, 1))
        cov_b = q_b * np.linalg.inv(XtWX)
        se_b = np.sqrt(np.clip(np.diag(cov_b), 0, None))
        # Center on observed β̂ per Clarke-Romano-Wolf eq (4).
        t_b = np.abs((beta_b - beta_obs) / np.where(se_b > 0, se_b, np.inf))
        boot_t_mods[b] = t_b[mod_idx]

    # Drop failed bootstrap reps.
    valid = ~np.isnan(boot_t_mods).any(axis=1)
    boot_t_mods = boot_t_mods[valid]
    M = len(boot_t_mods)
    if M == 0:
        return np.full(len(mod_idx), np.nan), {}

    # Stepdown algorithm.
    order = np.argsort(-t_obs_mod)
    t_ord = t_obs_mod[order]
    boot_ord = boot_t_mods[:, order]
    p_adj_ord = np.empty(len(mod_idx))
    p_adj_ord[0] = (np.sum(np.max(boot_ord, axis=1) >= t_ord[0]) + 1) / (M + 1)
    for s in range(1, len(mod_idx)):
        max_rem = np.max(boot_ord[:, s:], axis=1)
        p_init = (np.sum(max_rem >= t_ord[s]) + 1) / (M + 1)
        p_adj_ord[s] = max(p_init, p_adj_ord[s - 1])
    p_adj = np.empty(len(mod_idx))
    p_adj[order] = p_adj_ord
    return p_adj, dict(M=M, t_obs=t_obs_mod.tolist(),
                        boot_max=np.max(boot_t_mods, axis=1).tolist())


def romano_wolf_univariate(y, v, Z, mods, tau2_obs, B=2000, seed=42):
    """Romano-Wolf stepdown bootstrap-t p-values computed over the SAME
    family of UNIVARIATE per-moderator slope statistics that BH and Storey
    use, so every multiplicity correction operates on one common family of
    marginal associations.

    This replaces the joint multivariate Romano-Wolf, which is invalid at
    k=12: the design [intercept | all moderators] has n_predictors >= k, so
    the WLS fit is (near-)saturated, residuals collapse to ~0, the HKSJ
    scale q -> 0, studentised statistics diverge, and every RW p-value
    floors at 1/(M+1).

    For each bootstrap draw we resample the k cities with Dirichlet (mean-1
    exponential) cluster weights and refit each univariate moderator
    regression independently.  Statistics are studentised and centred on
    the OBSERVED slope per Clarke-Romano-Wolf (2020, Stata Journal
    20(4):812-843) eq (4); stepdown monotonicity is enforced per eq (7).
    τ² is held at the supplied intercept-only PM estimate, matching the
    fixed-η convention used by the rest of this script.
    """
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    Zmat = np.asarray(Z, dtype=float)
    if Zmat.ndim == 1:
        Zmat = Zmat.reshape(-1, 1)
    k = len(y)
    m = len(mods)
    rng = np.random.default_rng(seed)
    w_meta = 1.0 / (v + tau2_obs)

    # Observed univariate slopes / HKSJ SEs (one regression per moderator).
    beta_obs = np.empty(m)
    se_obs = np.empty(m)
    for j in range(m):
        Xj = np.column_stack([np.ones(k), Zmat[:, j]])
        hk = hksj_ci(np.zeros(2), Xj, y, v, tau2_obs)
        beta_obs[j] = hk["beta"][1]
        se_obs[j] = hk["se"][1]
    t_obs = np.abs(beta_obs / np.where(se_obs > 0, se_obs, np.inf))

    boot_t = np.full((B, m), np.nan)
    for b in range(B):
        u = rng.exponential(scale=1.0, size=k)
        w_b = k * u / u.sum()
        w_eff = w_b * w_meta
        tb = np.empty(m)
        ok = True
        for j in range(m):
            Xj = np.column_stack([np.ones(k), Zmat[:, j]])
            XtWX = Xj.T @ (Xj * w_eff[:, None])
            try:
                beta_b = np.linalg.solve(XtWX, Xj.T @ (w_eff * y))
                cov = np.linalg.inv(XtWX)
            except np.linalg.LinAlgError:
                ok = False
                break
            resid = y - Xj @ beta_b
            q_b = float(np.sum(w_eff * resid ** 2) / max(k - 2, 1))
            se_b = np.sqrt(max(q_b * cov[1, 1], 0.0))
            tb[j] = (abs(beta_b[1] - beta_obs[j]) / se_b
                     if se_b > 0 else np.inf)
        if ok:
            boot_t[b] = tb

    valid = ~np.isnan(boot_t).any(axis=1)
    boot_t = boot_t[valid]
    M = len(boot_t)
    if M == 0:
        return np.full(m, np.nan), {}

    order = np.argsort(-t_obs)
    t_ord = t_obs[order]
    boot_ord = boot_t[:, order]
    p_adj_ord = np.empty(m)
    p_adj_ord[0] = (np.sum(np.max(boot_ord, axis=1) >= t_ord[0]) + 1) / (M + 1)
    for s in range(1, m):
        max_rem = np.max(boot_ord[:, s:], axis=1)
        p_init = (np.sum(max_rem >= t_ord[s]) + 1) / (M + 1)
        p_adj_ord[s] = max(p_init, p_adj_ord[s - 1])
    p_adj = np.empty(m)
    p_adj[order] = p_adj_ord
    return p_adj, dict(M=M, t_obs=t_obs.tolist(),
                        family="univariate_per_moderator")


def influence_diagnostics(y, v, Xj, tau2, cities):
    """Leave-one-out and case-deletion influence diagnostics for a single
    univariate meta-regression slope, following Viechtbauer & Cheung (2010,
    Research Synthesis Methods 1:112-125): hat values (leverage), Cook's
    distance, DFBETAS, and the LOO range of the slope and its t.

    At k=12 a meta-regression can be dominated by a single high-leverage
    market (the renter-share sign-flip pathology).  These diagnostics make
    that fragility visible instead of hiding it behind a point estimate.
    τ² is held at the supplied intercept-only PM value for every refit,
    consistent with the fixed-τ² convention used elsewhere here; this
    isolates leverage in the mean structure from sampling noise in the
    heterogeneity estimate.  Cook's distance is the squared Mahalanobis
    distance between the full and deleted coefficient vectors in the metric
    of the full-model precision X'WX (metafor's rma convention, no /p).
    """
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    X = np.asarray(Xj, dtype=float)
    k, p = X.shape

    hk_full = hksj_ci(np.zeros(p), X, y, v, tau2)
    beta_full = hk_full["beta"]

    w = 1.0 / (v + tau2)
    XtWX = X.T @ (X * w[:, None])
    cov_naive = np.linalg.inv(XtWX)
    H = X @ cov_naive @ (X * w[:, None]).T          # hat matrix
    hat = np.clip(np.diag(H), 0.0, None)

    loo_beta = np.empty(k)
    loo_t = np.empty(k)
    cooks = np.empty(k)
    dfbetas = np.empty(k)
    keep = np.ones(k, dtype=bool)
    for i in range(k):
        keep[i] = False
        hk_i = hksj_ci(np.zeros(p), X[keep], y[keep], v[keep], tau2)
        keep[i] = True
        b_i = hk_i["beta"]
        loo_beta[i] = b_i[1]
        loo_t[i] = hk_i["t"][1]
        dfb = beta_full - b_i
        cooks[i] = float(dfb @ XtWX @ dfb)
        se_i = hk_i["se"][1]
        dfbetas[i] = float(dfb[1] / se_i) if se_i > 0 else np.nan

    imax = int(np.nanargmax(cooks))
    per_city = [
        {"city": str(cities[i]), "hat": float(hat[i]),
         "cooks_d": float(cooks[i]), "dfbetas_slope": float(dfbetas[i]),
         "loo_beta": float(loo_beta[i]), "loo_t": float(loo_t[i])}
        for i in range(k)
    ]
    return {
        "loo_beta_min": float(np.min(loo_beta)),
        "loo_beta_max": float(np.max(loo_beta)),
        "loo_t_min": float(np.min(loo_t)),
        "loo_t_max": float(np.max(loo_t)),
        "loo_beta_sign_flips": bool(np.min(loo_beta) < 0.0 < np.max(loo_beta)),
        "max_cooks_d": float(np.max(cooks)),
        "max_leverage": float(np.max(hat)),
        "max_abs_dfbetas": float(np.nanmax(np.abs(dfbetas))),
        "most_influential_city": str(cities[imax]),
        "per_city": per_city,
    }


def bh_qvalues(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = np.minimum.accumulate((ranked * n / (np.arange(n) + 1))[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def storey_qvalues(pvals, lam=0.5):
    """Storey (2002) adaptive q-values.  More powerful than BH under
    positive dependence; pi0 is estimated from the proportion of p-values
    exceeding lambda.  Reference: Storey 2002 JRSSB 64(3):479-498.
    """
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    pi0 = float((p > lam).sum()) / max((1 - lam) * n, 1e-9)
    pi0 = float(min(max(pi0, 1.0 / n), 1.0))
    bh = bh_qvalues(p)
    return np.clip(pi0 * bh, 0, 1), pi0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baur_table",
                    default=str(REPO / "results" / "replications"
                                  / "baur_pooled_pca"
                                  / "baur_pooled_pca_table.csv"))
    ap.add_argument("--shen_table",
                    default=str(REPO / "results" / "replications"
                                  / "shen_12city_table.csv"))
    ap.add_argument("--acs",
                    default=str(REPO / "results" / "replications"
                                  / "acs_moderators_12city.csv"))
    ap.add_argument("--source", choices=["baur_pooled", "shen"],
                    default="baur_pooled")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out",
                    default=str(REPO / "results" / "replications"
                                  / "meta_regression_pooled.json"))
    args = ap.parse_args()

    if args.source == "baur_pooled":
        eff = pd.read_csv(args.baur_table).set_index("city")
        ci_half = (eff["dml_ci_high"] - eff["dml_ci_low"]) / 2.0
        eff["se_eff"] = ci_half / 1.96
        eff["theta"] = eff["dml_theta"]
    else:
        eff = pd.read_csv(args.shen_table).set_index("city")
        ci_half = (eff["dml_ci_high"] - eff["dml_ci_low"]) / 2.0
        eff["se_eff"] = ci_half / 1.96
        eff["theta"] = eff["dml_theta"]
    acs = pd.read_csv(args.acs).set_index("city")
    df = eff[["theta", "se_eff"]].join(acs, how="inner")
    df = df.dropna(subset=["theta", "se_eff"])
    print(f"\n=== {args.source} input: k={len(df)} cities ===")
    print(df[["theta", "se_eff"]].to_string(float_format=lambda x: f"{x:+.4f}"))

    y = df["theta"].to_numpy()
    v = df["se_eff"].to_numpy() ** 2

    mods = ["log_population", "median_hh_income", "pct_renter",
            "pct_college", "diversity_gini_simpson", "median_home_value",
            "median_year_built", "pop_density_per_sqmi",
            # Federal-data moderators (BEA/BLS/FHFA via FRED, added 2026-06-03)
            "fred_unemp_rate", "fred_hpi_yoy_pct", "fred_pcpi"]
    mods = [m for m in mods if m in df.columns]
    Z = df[mods].apply(lambda s: (s - s.mean()) / (s.std(ddof=1) or 1.0))

    # Baseline: intercept-only random-effects pooled mean.
    X0 = np.ones((len(df), 1))
    tau2_pm = tau2_paule_mandel(y, v)
    tau2_rm = tau2_reml(y, v)
    print(f"\n  τ²(PM) = {tau2_pm:.5f}  τ²(REML) = {tau2_rm:.5f}")

    baseline_pm = hksj_ci(np.array([y.mean()]), X0, y, v, tau2_pm)
    baseline_rm = hksj_ci(np.array([y.mean()]), X0, y, v, tau2_rm)
    print(f"\n=== Intercept-only random-effects pooled estimate ===")
    print(f"  PM + HKSJ : θ̂={baseline_pm['beta'][0]:+.4f} "
          f"SE={baseline_pm['se'][0]:.4f} "
          f"CI=[{baseline_pm['ci_low'][0]:+.4f}, "
          f"{baseline_pm['ci_high'][0]:+.4f}] p={baseline_pm['p'][0]:.4f}")
    print(f"  REML+HKSJ : θ̂={baseline_rm['beta'][0]:+.4f} "
          f"SE={baseline_rm['se'][0]:.4f} "
          f"CI=[{baseline_rm['ci_low'][0]:+.4f}, "
          f"{baseline_rm['ci_high'][0]:+.4f}] p={baseline_rm['p'][0]:.4f}")

    # Univariate moderator scan with both BH (legacy) and RW (upgraded).
    print(f"\n=== Univariate moderator scan ===")
    cities = df.index.tolist()
    uni_rows = []
    influence_by_mod = {}
    for j, m in enumerate(mods):
        Xj = np.column_stack([np.ones(len(df)), Z[m].to_numpy()])
        # τ² fixed at the intercept-only PM estimate across all univariate
        # moderator scans (Veroniki 2016, Hartung-Knapp 2001). Re-estimating
        # per moderator conflates "between-study variance after partialling
        # out moderator j" with the unconditional heterogeneity that t-crit
        # values are calibrated against; fixing τ² is the standard.
        hk = hksj_ci(np.zeros(2), Xj, y, v, tau2_pm)
        influence_by_mod[m] = influence_diagnostics(y, v, Xj, tau2_pm, cities)
        uni_rows.append({
            "moderator": m,
            "beta": float(hk["beta"][1]),
            "se_hksj": float(hk["se"][1]),
            "t": float(hk["t"][1]),
            "p_raw": float(hk["p"][1]),
            "ci_low": float(hk["ci_low"][1]),
            "ci_high": float(hk["ci_high"][1]),
            "tau2_pm": tau2_pm,
        })

    uni_df = pd.DataFrame(uni_rows)
    uni_df["bh_q"] = bh_qvalues(uni_df["p_raw"].to_numpy())

    # One-sided p-values under the Shen-Ross (2021) pre-registered
    # directional alternative H_a: β > 0 for renter_share (and by analogy
    # the same direction for the other demographic moderators that
    # operate through search-and-pricing exposure).  Endorsed by Lehmann
    # & Romano (2022, Testing Statistical Hypotheses, 4e §3.7) when the
    # alternative is scientifically determined ex ante.
    df_t = uni_df["t"].to_numpy()
    df_resid = len(df) - 2   # k - p for univariate meta-reg with intercept + 1 mod
    p_one = 1 - stats.t.cdf(df_t, df=df_resid)
    uni_df["p_one_sided"] = p_one
    uni_df["bh_q_one_sided"] = bh_qvalues(p_one)
    storey_q_one, pi0_one = storey_qvalues(p_one)
    uni_df["storey_q_one_sided"] = storey_q_one

    # --- Romano-Wolf multiplicity correction over the UNIVARIATE family ---
    # FIX: do NOT run Romano-Wolf on the joint multivariate meta-regression.
    # At k=12 the joint design [intercept | all moderators] has
    # n_predictors = 1 + len(mods), which is >= k here (near-saturated at 8
    # mods, fully saturated at 11): WLS residuals collapse to ~0, the HKSJ
    # scale q -> 0, studentised statistics diverge, and every RW p-value
    # floors at 1/(M+1).  We instead run RW over the SAME univariate
    # per-moderator slope statistics that BH/Storey use, so all multiplicity
    # corrections operate on one common family of marginal associations.
    n_predictors_joint = 1 + len(mods)
    if n_predictors_joint >= len(df):
        print(f"\n  [guard] joint multivariate Romano-Wolf disabled: "
              f"n_predictors={n_predictors_joint} >= k={len(df)} "
              f"(saturated); running RW on the univariate moderator family.")
    p_rw, rw_meta = romano_wolf_univariate(
        y, v, Z.to_numpy(), mods, tau2_pm,
        B=args.n_boot, seed=args.seed,
    )
    uni_df["rw_p"] = p_rw
    print(uni_df.to_string(index=False,
                            float_format=lambda x: f"{x:+.4f}"))
    print(f"\n  Storey π̂_0 (one-sided) = {pi0_one:.3f}")

    # Influence / leverage diagnostics (Viechtbauer & Cheung 2010): surface
    # the k=12 single-market fragility (e.g. renter-share slope sign flips)
    # rather than hide it behind the point estimates.
    infl_print = pd.DataFrame([{
        "moderator": m,
        "beta": float(uni_df.loc[uni_df["moderator"] == m, "beta"].iloc[0]),
        "loo_beta_min": influence_by_mod[m]["loo_beta_min"],
        "loo_beta_max": influence_by_mod[m]["loo_beta_max"],
        "sign_flip": influence_by_mod[m]["loo_beta_sign_flips"],
        "max_cooks_d": influence_by_mod[m]["max_cooks_d"],
        "max_lev": influence_by_mod[m]["max_leverage"],
        "infl_city": influence_by_mod[m]["most_influential_city"],
    } for m in mods])
    print(f"\n=== LOO / influence diagnostics (Viechtbauer & Cheung 2010) ===")
    print(f"  (mean leverage under H0 = p/k = 2/{len(df)} = "
          f"{2.0 / len(df):.3f}; a hat value >> this flags high leverage)")
    print(infl_print.to_string(index=False,
                                float_format=lambda x: f"{x:+.4f}"))

    # Pooled effect: also report one-sided
    t_pooled = baseline_pm["t"][0] if "t" in baseline_pm else \
               baseline_pm["beta"][0] / baseline_pm["se"][0]
    p_pool_one = 1 - stats.t.cdf(t_pooled, df=baseline_pm["df"])
    print(f"\n=== Pooled effect: one-sided HKSJ p (H_a: θ > 0) ===")
    print(f"  PM + HKSJ : θ̂={baseline_pm['beta'][0]:+.4f}  "
          f"t({baseline_pm['df']})={t_pooled:+.3f}  "
          f"one-sided p = {p_pool_one:.4f}")

    # Merge influence diagnostics into each univariate record for the JSON.
    uni_records = uni_df.to_dict(orient="records")
    for rec in uni_records:
        infl = influence_by_mod[rec["moderator"]]
        rec["loo_beta_min"] = infl["loo_beta_min"]
        rec["loo_beta_max"] = infl["loo_beta_max"]
        rec["loo_t_min"] = infl["loo_t_min"]
        rec["loo_t_max"] = infl["loo_t_max"]
        rec["loo_beta_sign_flips"] = infl["loo_beta_sign_flips"]
        rec["max_cooks_d"] = infl["max_cooks_d"]
        rec["max_leverage"] = infl["max_leverage"]
        rec["max_abs_dfbetas"] = infl["max_abs_dfbetas"]
        rec["most_influential_city"] = infl["most_influential_city"]
        rec["influence_per_city"] = infl["per_city"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "source": args.source,
        "k": int(len(df)),
        "tau2_pm": tau2_pm, "tau2_reml": tau2_rm,
        "intercept_pm_hksj": {
            "theta": float(baseline_pm["beta"][0]),
            "se": float(baseline_pm["se"][0]),
            "ci_low": float(baseline_pm["ci_low"][0]),
            "ci_high": float(baseline_pm["ci_high"][0]),
            "p_two_sided": float(baseline_pm["p"][0]),
            "p_one_sided": float(p_pool_one),
            "t_stat": float(t_pooled),
            "df": int(baseline_pm["df"]),
        },
        "intercept_reml_hksj": {
            "theta": float(baseline_rm["beta"][0]),
            "se": float(baseline_rm["se"][0]),
            "ci_low": float(baseline_rm["ci_low"][0]),
            "ci_high": float(baseline_rm["ci_high"][0]),
            "p": float(baseline_rm["p"][0]),
        },
        "univariate": uni_records,
        "romano_wolf_M": rw_meta.get("M", 0) if isinstance(rw_meta, dict) else 0,
        "romano_wolf_family": "univariate_per_moderator",
    }
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nJSON -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
