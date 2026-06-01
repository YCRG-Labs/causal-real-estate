"""Roll up the 12-city Baur et al. 2023 JSONs into a paper-ready table.

Reads results/replications/baur_{city}.json for each of the 12 cities and
emits the headline contrast:

  - Predictive gain from adding BERT to the structured baseline:
    delta_MAE, delta_MAE_relative (% of structured MAE), delta_MAPE.
  - DML theta on PC1 of the BERT embedding (causal estimate after full
    confounder residualization). Bootstrap CI + IF SE shown side-by-side.
  - Random-effects pooled DML theta across the 12 metros (DerSimonian-
    Laird), with Q-test heterogeneity stats.

The Baur paper's headline finding is that adding text to the structured
baseline improves prediction. Our DML pass tests whether that predictive
gain survives full causal residualization. Pooled DML theta near zero with
positive delta_MAE confirms the spatial-confounding story: text helps
prediction precisely because it proxies location, not because it carries
an independent causal channel.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]

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
        if cur is None or not isinstance(cur, dict) or k not in cur:
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir",
                    default=str(REPO / "results" / "replications"))
    args = ap.parse_args()
    results_dir = Path(args.results_dir)

    rows = []
    missing = []
    for city in CITY_ORDER:
        p = results_dir / f"baur_{city}.json"
        if not p.exists():
            missing.append(city); continue
        d = json.loads(p.read_text())
        struct_mae = _pull(d, "cv_structured", "mae")
        text_mae = _pull(d, "cv_structured_plus_bert", "mae")
        struct_mape = _pull(d, "cv_structured", "mape")
        text_mape = _pull(d, "cv_structured_plus_bert", "mape")
        delta_mae = _pull(d, "predictive_gain", "delta_mae")
        rel_mae = _pull(d, "predictive_gain", "delta_mae_relative")
        delta_mape = _pull(d, "predictive_gain", "delta_mape")
        dml = _pull(d, "dml_bert") or {}
        rows.append({
            "city": city,
            "n": int(d.get("n", 0)),
            "engine": d.get("engine"),
            "n_struct": _pull(d, "n_structured_features"),
            "embed_dim": _pull(d, "embedding_dim"),
            "struct_mae": struct_mae, "text_mae": text_mae,
            "struct_mape": struct_mape, "text_mape": text_mape,
            "delta_mae": delta_mae, "rel_mae": rel_mae,
            "delta_mape": delta_mape,
            "dml_theta": dml.get("theta"),
            "dml_se": dml.get("se"),
            "dml_se_if": dml.get("se_if"),
            "dml_n_boot": dml.get("n_boot"),
            "dml_ci_low": dml.get("ci_low"),
            "dml_ci_high": dml.get("ci_high"),
            "dml_excludes_zero": (dml.get("contains_zero") is False),
        })

    if missing:
        print(f"warning: missing JSONs for {missing}", file=sys.stderr)
    if not rows:
        print("no JSONs read", file=sys.stderr); return 1

    print()
    print(f"{'city':<14}{'n':>5}  "
          f"{'struct MAPE':>11}{'+BERT MAPE':>11}{'ΔMAE%':>7}  "
          f"{'DML θ':>9}{'se_b/se_if':>14}  "
          f"{'95% CI':>22}  {'≠0':>3}")
    print("-" * 110)
    for r in rows:
        ci = (f"[{r['dml_ci_low']:+.3f}, {r['dml_ci_high']:+.3f}]"
              if r["dml_ci_low"] is not None else "—")
        excl = "Y" if r["dml_excludes_zero"] else "n"
        if r.get("dml_n_boot"):
            se_str = f"{(r['dml_se'] or 0):>5.3f}/{(r['dml_se_if'] or 0):.3f}"
        else:
            se_str = f"({(r['dml_se'] or 0):>5.3f})"
        print(f"{DISPLAY[r['city']]:<14}{r['n']:>5}  "
              f"{100*(r['struct_mape'] or 0):>10.2f}%"
              f"{100*(r['text_mape'] or 0):>10.2f}%"
              f"{100*(r['rel_mae'] or 0):>+6.2f}%  "
              f"{(r['dml_theta'] or 0):>+9.4f}{se_str:>14}  "
              f"{ci:>22}  {excl:>3}")

    thetas = np.array([r["dml_theta"] for r in rows if r["dml_theta"] is not None])
    ses = np.array([r["dml_se"] for r in rows if r["dml_se"] is not None])
    pool = random_effects_pool(thetas, ses)

    print("-" * 110)
    print(f"{'POOLED (RE)':<14}{'':>5}  {'':>11}{'':>11}{'':>7}  "
          f"{pool['theta_re']:>+9.4f}({pool['se_re']:>5.3f})        "
          f"[{pool['theta_re']-1.96*pool['se_re']:+.3f}, "
          f"{pool['theta_re']+1.96*pool['se_re']:+.3f}]   "
          f"Q={pool['Q']:.1f}  I²={pool['I2_pct']:.0f}%")
    print(f"{'POOLED (FE)':<14}{'':>5}  {'':>11}{'':>11}{'':>7}  "
          f"{pool['theta_fe']:>+9.4f}({pool['se_fe']:>5.3f})")

    mean_rel_mae = float(np.mean([r["rel_mae"] for r in rows if r["rel_mae"] is not None]))
    print(f"\nMean ΔMAE from +BERT: {100*mean_rel_mae:+.2f}% of structured MAE "
          f"({sum(1 for r in rows if (r['rel_mae'] or 0) > 0)}/{len(rows)} cities show predictive gain)")
    n_excl = sum(1 for r in rows if r["dml_excludes_zero"])
    print(f"Cities with DML CI excluding 0: {n_excl}/{len(rows)}")

    csv_path = results_dir / "baur_12city_table.csv"
    cols = ["city", "n", "engine", "n_struct", "embed_dim",
            "struct_mae", "text_mae", "struct_mape", "text_mape",
            "delta_mae", "rel_mae", "delta_mape",
            "dml_theta", "dml_se", "dml_se_if", "dml_n_boot",
            "dml_ci_low", "dml_ci_high", "dml_excludes_zero"]
    with csv_path.open("w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(c) is None else str(r[c]) for c in cols) + "\n")
    print(f"\nCSV -> {csv_path}")
    pool_path = results_dir / "baur_12city_pooled.json"
    pool_path.write_text(json.dumps(pool, indent=2))
    print(f"pooled JSON -> {pool_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
