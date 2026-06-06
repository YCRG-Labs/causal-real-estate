"""Portland-excluded sensitivity for the panel meta-analysis.

Phase A diagnostic (data/scripts/replications/portland_diagnose.py) showed
that Portland's per-city DML θ extreme (Baur -1.39, Shen +1.89, opposite
signs) is driven by INCOMPLETE CONFOUNDER COVERAGE, not by the PC1 axis
itself: raw treatment scale is normal (std 0.172, identical to other
cities), and Flury misalignment is worse in Boston / Denver / NYC without
producing comparable θ blowups. The mechanism is a documented data
quality issue (~5% of Portland is outside the Multnomah County parcel
file, with no explicit parcel-to-listing join), causing unabsorbed
location-text confounding to leak through DML's residualization step.

Following Cooper & Hedges 2019 (Handbook of Research Synthesis, ch. 12)
on documented data-quality exclusions, Viechtbauer & Cheung 2010 (RSM
1:112-125) on sign-flip outliers across methods, and Chetty et al 2018
(REStud) precedent for excluding one MSA with implausible sign-flip,
this script regenerates the panel headline numbers with Portland dropped
and prints a side-by-side comparison with the full-panel version.

Usage:
    python data/scripts/replications/portland_sensitivity.py

Outputs:
    results/replications/portland_sensitivity_table.csv
    results/replications/portland_sensitivity.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
REPL = REPO / "results" / "replications"


def paule_mandel_tau2(y: np.ndarray, v: np.ndarray,
                       tol: float = 1e-8, max_iter: int = 200) -> float:
    """Paule-Mandel estimator of between-study variance.  Preferred over
    DerSimonian-Laird in small-G meta-analysis per Veroniki 2016 RSM."""
    tau2 = max(np.var(y, ddof=1) - np.mean(v), 0.0)
    for _ in range(max_iter):
        w = 1.0 / (v + tau2)
        mu = np.sum(w * y) / np.sum(w)
        F = np.sum(w * (y - mu) ** 2) - (len(y) - 1)
        if abs(F) < tol:
            return float(max(tau2, 0.0))
        dF = -np.sum((w ** 2) * (y - mu) ** 2)
        if dF == 0:
            break
        tau2 = max(tau2 - F / dF, 0.0)
    return float(tau2)


def hksj_pooled(y: np.ndarray, v: np.ndarray,
                 tau2: float) -> dict[str, float]:
    """Hartung-Knapp-Sidik-Jonkman intercept-only random-effects pool."""
    w = 1.0 / (v + tau2)
    mu = float(np.sum(w * y) / np.sum(w))
    k = len(y)
    q = float(np.sum(w * (y - mu) ** 2) / (k - 1))
    se = float(np.sqrt(q / np.sum(w)))
    t_crit = float(stats.t.ppf(0.975, df=k - 1))
    p = float(2 * (1 - stats.t.cdf(abs(mu / se), df=k - 1)))
    return {
        "theta": mu, "se": se,
        "ci_low": mu - t_crit * se,
        "ci_high": mu + t_crit * se,
        "p_two_sided": p,
        "p_one_sided_pos": float(1 - stats.t.cdf(mu / se, df=k - 1)),
        "tau2_pm": tau2,
        "k": k,
        "Q": float(np.sum((1.0 / v) * (y - mu) ** 2)),
    }


def hotelling_t2(z_shen: float, z_baur: float, z_cf: float) -> dict:
    """χ²_3 joint-null test, methods asymptotically independent."""
    T2 = z_shen ** 2 + z_baur ** 2 + z_cf ** 2
    return {"T2": float(T2),
            "p_chi2_3df": float(1 - stats.chi2.cdf(T2, df=3))}


def load_per_city_baur() -> pd.DataFrame:
    return pd.read_csv(REPL / "baur_pooled_pca" / "baur_pooled_pca_table.csv")


def load_per_city_shen() -> pd.DataFrame:
    return pd.read_csv(REPL / "shen_12city_table.csv")


def load_per_city_cf() -> pd.DataFrame:
    return pd.read_csv(REPO / "results" / "counterfactual"
                          / "counterfactual_12city_table.csv")


def both_pool(name: str, df: pd.DataFrame, theta_col: str,
              se_col: str) -> tuple[dict, dict]:
    """PM+HKSJ pool of df, with and without Portland."""
    full = df.dropna(subset=[theta_col, se_col]).copy()
    drop = full[full["city"].astype(str).str.lower() != "portland"].copy()
    y_f = full[theta_col].to_numpy(dtype=float)
    v_f = full[se_col].to_numpy(dtype=float) ** 2
    y_d = drop[theta_col].to_numpy(dtype=float)
    v_d = drop[se_col].to_numpy(dtype=float) ** 2
    full_res = hksj_pooled(y_f, v_f, paule_mandel_tau2(y_f, v_f))
    drop_res = hksj_pooled(y_d, v_d, paule_mandel_tau2(y_d, v_d))
    return full_res, drop_res


def main() -> int:
    out: dict[str, dict] = {}

    # Baur.
    try:
        baur = load_per_city_baur()
        full_b, drop_b = both_pool("baur", baur,
                                     "dml_theta", "dml_se")
        out["baur"] = {"full_panel_k12": full_b, "drop_portland_k11": drop_b}
        print("=== Baur pooled (PM + HKSJ) ===")
        print(f"  k=12 (full):   theta = {full_b['theta']:+.4f}  "
              f"SE = {full_b['se']:.4f}  "
              f"CI = [{full_b['ci_low']:+.4f}, {full_b['ci_high']:+.4f}]  "
              f"p = {full_b['p_two_sided']:.4f}  tau2_PM = {full_b['tau2_pm']:.5f}")
        print(f"  k=11 (no PDX): theta = {drop_b['theta']:+.4f}  "
              f"SE = {drop_b['se']:.4f}  "
              f"CI = [{drop_b['ci_low']:+.4f}, {drop_b['ci_high']:+.4f}]  "
              f"p = {drop_b['p_two_sided']:.4f}  tau2_PM = {drop_b['tau2_pm']:.5f}")
        print()
    except FileNotFoundError as e:
        print(f"WARN: Baur table not found ({e})", file=sys.stderr)

    # Shen.
    try:
        shen = load_per_city_shen()
        full_s, drop_s = both_pool("shen", shen,
                                     "dml_theta", "dml_se")
        out["shen"] = {"full_panel_k12": full_s, "drop_portland_k11": drop_s}
        print("=== Shen pooled (PM + HKSJ) ===")
        print(f"  k=12 (full):   theta = {full_s['theta']:+.4f}  "
              f"SE = {full_s['se']:.4f}  "
              f"CI = [{full_s['ci_low']:+.4f}, {full_s['ci_high']:+.4f}]  "
              f"p = {full_s['p_two_sided']:.4f}  tau2_PM = {full_s['tau2_pm']:.5f}")
        print(f"  k=11 (no PDX): theta = {drop_s['theta']:+.4f}  "
              f"SE = {drop_s['se']:.4f}  "
              f"CI = [{drop_s['ci_low']:+.4f}, {drop_s['ci_high']:+.4f}]  "
              f"p = {drop_s['p_two_sided']:.4f}  tau2_PM = {drop_s['tau2_pm']:.5f}")
        print()
    except FileNotFoundError as e:
        print(f"WARN: Shen table not found ({e})", file=sys.stderr)

    # Hotelling joint-null rejection count, full and drop Portland.
    try:
        cf = load_per_city_cf()
        shen = load_per_city_shen()
        baur = load_per_city_baur()
        merged = (baur[["city", "dml_theta", "dml_se"]]
                   .rename(columns={"dml_theta": "baur_theta",
                                       "dml_se": "baur_se"})
                   .merge(shen[["city", "dml_theta", "dml_se"]]
                           .rename(columns={"dml_theta": "shen_theta",
                                              "dml_se": "shen_se"}),
                           on="city"))
        cf_se = (cf["te_ci_high"] - cf["te_ci_low"]) / (2 * 1.96)
        cf_m = pd.DataFrame({
            "city": cf["city"].astype(str).str.lower().str.replace(" ", "_"),
            "cf_theta": cf["te_delta_logprice"],
            "cf_se": cf_se,
        })
        # Map display names to slugs to match merged.
        slug_map = {
            "new_york": "nyc", "san_francisco": "sf",
            "washington_dc": "dc",
        }
        cf_m["city"] = cf_m["city"].replace(slug_map)
        m = merged.merge(cf_m, on="city")
        m["z_shen"] = m["shen_theta"] / m["shen_se"]
        m["z_baur"] = m["baur_theta"] / m["baur_se"]
        m["z_cf"] = m["cf_theta"] / m["cf_se"]
        T2 = (m["z_shen"] ** 2 + m["z_baur"] ** 2 + m["z_cf"] ** 2)
        m["T2"] = T2
        m["p_chi2"] = 1 - stats.chi2.cdf(T2, df=3)
        full_rej = int((m["p_chi2"] < 0.05).sum())
        drop = m[m["city"] != "portland"]
        drop_rej = int((drop["p_chi2"] < 0.05).sum())
        out["hotelling"] = {
            "full_panel_k12": {"rejecting_raw_05": full_rej,
                               "total": int(len(m))},
            "drop_portland_k11": {"rejecting_raw_05": drop_rej,
                                  "total": int(len(drop))},
        }
        print("=== Hotelling joint-null rejection rate ===")
        print(f"  k=12 (full):   {full_rej} / {len(m)} reject at raw p<0.05")
        print(f"  k=11 (no PDX): {drop_rej} / {len(drop)} reject at raw p<0.05")
        print()
    except Exception as e:
        print(f"WARN: Hotelling sensitivity skipped ({type(e).__name__}: {e})",
              file=sys.stderr)

    # Write JSON for §6 paper.
    out_path = REPL / "portland_sensitivity.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"JSON -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
