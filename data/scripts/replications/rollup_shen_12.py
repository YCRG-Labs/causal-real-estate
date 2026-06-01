"""Roll up the 12-city Shen Doc2Vec JSONs into a single paper-ready table.

Reads results/replications/shen_{city}.json for each of the 12 cities,
extracts (n, OLS uniqueness coef, DML theta, IF SE, 95% CI, cell-FE coef,
power vs Shen's published +0.149) and prints:
  - a markdown table for screen review
  - a CSV at results/replications/shen_12city_table.csv

Also computes a random-effects pooled estimate (DerSimonian-Laird) over the
12 per-city DML thetas using IF SEs, with Q-test heterogeneity stats.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "results" / "replications"

CITY_ORDER = ["boston", "nyc", "sf", "dc", "philadelphia", "chicago",
              "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]
DISPLAY = {"boston": "Boston", "nyc": "New York", "sf": "San Francisco",
           "dc": "Washington DC", "philadelphia": "Philadelphia",
           "chicago": "Chicago", "seattle": "Seattle", "denver": "Denver",
           "atlanta": "Atlanta", "portland": "Portland", "phoenix": "Phoenix",
           "dallas": "Dallas"}


def _pull(d, *keys, default=None):
    cur = d
    for k in keys:
        if cur is None or k not in cur:
            return default
        cur = cur[k]
    return cur


def random_effects_pool(theta, se):
    theta = np.asarray(theta, dtype=float)
    se = np.asarray(se, dtype=float)
    w_fe = 1.0 / np.maximum(se ** 2, 1e-12)
    theta_fe = float(np.sum(w_fe * theta) / np.sum(w_fe))
    Q = float(np.sum(w_fe * (theta - theta_fe) ** 2))
    k = len(theta)
    c = float(np.sum(w_fe) - np.sum(w_fe ** 2) / np.sum(w_fe))
    tau2 = max(0.0, (Q - (k - 1)) / max(c, 1e-12)) if k > 1 else 0.0
    w_re = 1.0 / (se ** 2 + tau2)
    theta_re = float(np.sum(w_re * theta) / np.sum(w_re))
    se_re = float(math.sqrt(1.0 / np.sum(w_re)))
    I2 = max(0.0, (Q - (k - 1)) / max(Q, 1e-12)) * 100.0 if k > 1 else 0.0
    return {"theta_re": theta_re, "se_re": se_re, "tau2": tau2,
            "Q": Q, "df": k - 1, "I2_pct": I2,
            "theta_fe": theta_fe, "se_fe": float(math.sqrt(1.0 / np.sum(w_fe)))}


def main():
    rows = []
    missing = []
    for city in CITY_ORDER:
        p = RESULTS / f"shen_{city}.json"
        if not p.exists():
            missing.append(city); continue
        d = json.loads(p.read_text())
        ols = _pull(d, "ols_uniqueness") or {}
        dml = _pull(d, "dml_uniqueness") or {}
        cell = _pull(d, "cell_fe_result", "ols") or {}
        pwr = _pull(d, "power_at_published_effect") or {}
        rows.append({
            "city": city,
            "n": int(d.get("n", 0)),
            "ols_beta": ols.get("coef"),
            "ols_se": ols.get("se"),
            "ols_p": ols.get("p"),
            "ols_pct_per_sd": ols.get("pct_per_sd"),
            "cell_fe_beta": cell.get("coef"),
            "cell_fe_se": cell.get("se"),
            "cell_fe_p": cell.get("p"),
            "dml_theta": dml.get("theta"),
            "dml_se": dml.get("se"),
            "dml_se_if": dml.get("se_if"),
            "dml_n_boot": dml.get("n_boot"),
            "dml_ci_low": dml.get("ci_low"),
            "dml_ci_high": dml.get("ci_high"),
            "dml_excludes_zero": (dml.get("contains_zero") is False),
            "power_vs_pub": pwr.get("power"),
        })

    if missing:
        print(f"warning: missing JSONs for {missing}", file=sys.stderr)

    if not rows:
        print("no JSONs read; aborting", file=sys.stderr); return 1

    any_boot = any((r.get("dml_n_boot") or 0) > 0 for r in rows)
    print()
    se_header = "se_b/se_if" if any_boot else "(se)"
    print(f"{'city':<14}{'n':>5}  "
          f"{'OLS β':>9}{'(se)':>8}  "
          f"{'%/σ':>6}  "
          f"{'cellFE β':>10}{'(se)':>8}  "
          f"{'DML θ':>9}{se_header:>14}  "
          f"{'95% CI':>22}  "
          f"{'≠0':>3}  "
          f"{'pwr':>5}")
    print("-" * 128)
    for r in rows:
        ci = f"[{r['dml_ci_low']:+.3f}, {r['dml_ci_high']:+.3f}]" \
            if r["dml_ci_low"] is not None else "—"
        excl = "Y" if r["dml_excludes_zero"] else "n"
        if any_boot:
            se_str = f"{(r['dml_se'] or 0):>5.3f}/{(r['dml_se_if'] or 0):.3f}"
        else:
            se_str = f"({(r['dml_se'] or 0):>5.3f})"
        print(f"{DISPLAY[r['city']]:<14}{r['n']:>5}  "
              f"{(r['ols_beta'] or 0):>+9.4f}({(r['ols_se'] or 0):>5.3f})  "
              f"{(r['ols_pct_per_sd'] or 0):>+5.1f}  "
              f"{(r['cell_fe_beta'] or 0):>+10.4f}({(r['cell_fe_se'] or 0):>5.3f})  "
              f"{(r['dml_theta'] or 0):>+9.4f}{se_str:>14}  "
              f"{ci:>22}  "
              f"{excl:>3}  "
              f"{(r['power_vs_pub'] or 0):>5.3f}")

    theta = np.array([r["dml_theta"] for r in rows if r["dml_theta"] is not None])
    se = np.array([r["dml_se"] for r in rows if r["dml_se"] is not None])
    pool = random_effects_pool(theta, se)
    print("-" * 122)
    print(f"{'POOLED (RE, DL)':<14}{'':>5}  "
          f"{'':>9}{'':>8}  "
          f"{'':>6}  "
          f"{'':>10}{'':>8}  "
          f"{pool['theta_re']:>+9.4f}({pool['se_re']:>5.3f})  "
          f"[{pool['theta_re']-1.96*pool['se_re']:+.3f}, "
          f"{pool['theta_re']+1.96*pool['se_re']:+.3f}]   "
          f"Q={pool['Q']:.1f}  τ²={pool['tau2']:.3f}  I²={pool['I2_pct']:.0f}%")
    print(f"{'POOLED (FE, IV)':<14}{'':>5}  "
          f"{'':>9}{'':>8}  "
          f"{'':>6}  "
          f"{'':>10}{'':>8}  "
          f"{pool['theta_fe']:>+9.4f}({pool['se_fe']:>5.3f})")

    print(f"\nShen 2021 published (Atlanta MLS-Area × Year FE): +0.149 (SE 0.034)")
    n_excl = sum(1 for r in rows if r["dml_excludes_zero"])
    print(f"Cities with DML 95% CI excluding 0: {n_excl}/{len(rows)}")

    csv_path = RESULTS / "shen_12city_table.csv"
    cols = ["city", "n", "ols_beta", "ols_se", "ols_p", "ols_pct_per_sd",
            "cell_fe_beta", "cell_fe_se", "cell_fe_p",
            "dml_theta", "dml_se", "dml_ci_low", "dml_ci_high",
            "dml_excludes_zero", "power_vs_pub"]
    with csv_path.open("w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"\nCSV -> {csv_path}")

    pool_path = RESULTS / "shen_12city_pooled.json"
    pool_path.write_text(json.dumps(pool, indent=2))
    print(f"pooled JSON -> {pool_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
