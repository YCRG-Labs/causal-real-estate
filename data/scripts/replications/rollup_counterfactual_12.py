"""Roll up the 12-city counterfactual JSONs into a paper-ready table.

Reads results/counterfactual/{city}.json for each of the 12 cities and
emits the §6.3 mechanism panel:

  - n_listings, n_swap_valid, n_strip_valid
  - validation pass rates (slot_preserved, classifier_flipped_toward_target)
  - NDE (style-stripped):  mean Δ log-price, 95% CI, % price implied
  - TE  (style-swap):      mean Δ log-price, 95% CI, % price implied
  - random-effects pooled NDE and TE across the 12 metros (DerSimonian-Laird)

The TE sign is the key reading: negative TE means rewriting toward
alternative submarkets pulls predicted price downward (text-as-spatial-
proxy mechanism present); near-zero TE means the text channel does not
encode price-relevant location signal in that city.
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


def se_from_ci(lo, hi):
    return abs(float(hi) - float(lo)) / (2.0 * 1.96)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir",
                    default=str(REPO / "results" / "counterfactual"))
    args = ap.parse_args()
    results_dir = Path(args.results_dir)

    rows = []
    missing = []
    for city in CITY_ORDER:
        p = results_dir / f"{city}.json"
        if not p.exists():
            missing.append(city); continue
        d = json.loads(p.read_text())
        nde = _pull(d, "natural_direct_effect_style_stripped") or {}
        te = _pull(d, "total_effect_style_swap") or {}
        pass_rates = _pull(d, "validation_pass_rates") or {}
        dml = _pull(d, "dml") or {}
        n_listings = (d.get("n_listings_processed")
                      or d.get("n_listings_requested")
                      or len(d.get("listings", [])))
        rows.append({
            "city": city,
            "n_listings": int(n_listings) if n_listings else 0,
            "dml_theta": dml.get("theta"),
            "dml_se": dml.get("se"),
            "nde_mean": nde.get("mean_delta_logprice"),
            "nde_ci_low": nde.get("ci_low"),
            "nde_ci_high": nde.get("ci_high"),
            "nde_n_valid": nde.get("n_valid"),
            "nde_pct": nde.get("pct_change_implied"),
            "te_mean": te.get("mean_delta_logprice"),
            "te_ci_low": te.get("ci_low"),
            "te_ci_high": te.get("ci_high"),
            "te_n_valid": te.get("n_valid"),
            "te_pct": te.get("pct_change_implied"),
            "slot_preserved": pass_rates.get("slot_preserved"),
            "classifier_flipped": pass_rates.get("classifier_flipped_toward_target"),
            "overall_pass": pass_rates.get("overall_pass"),
        })

    if missing:
        print(f"warning: missing JSONs for {missing}", file=sys.stderr)
    if not rows:
        print("no JSONs read", file=sys.stderr); return 1

    print()
    print(f"{'city':<14}{'n':>5}{'n_str':>6}{'n_sw':>6}  "
          f"{'NDE (Δlog)':>14}{'95% CI':>22}{'%':>7}  "
          f"{'TE (Δlog)':>14}{'95% CI':>22}{'%':>7}  "
          f"{'slot':>5}{'flip':>5}")
    print("-" * 140)
    for r in rows:
        nde_ci = (f"[{r['nde_ci_low']:+.3f}, {r['nde_ci_high']:+.3f}]"
                  if r["nde_ci_low"] is not None else "—")
        te_ci = (f"[{r['te_ci_low']:+.3f}, {r['te_ci_high']:+.3f}]"
                 if r["te_ci_low"] is not None else "—")
        print(f"{DISPLAY[r['city']]:<14}"
              f"{(r['n_listings'] or 0):>5}"
              f"{(r['nde_n_valid'] or 0):>6}"
              f"{(r['te_n_valid'] or 0):>6}  "
              f"{(r['nde_mean'] or 0):>+14.4f}{nde_ci:>22}{(r['nde_pct'] or 0):>+6.1f}%  "
              f"{(r['te_mean'] or 0):>+14.4f}{te_ci:>22}{(r['te_pct'] or 0):>+6.1f}%  "
              f"{(r['slot_preserved'] or 0):>5.3f}{(r['classifier_flipped'] or 0):>5.3f}")

    nde_theta = np.array([r["nde_mean"] for r in rows if r["nde_mean"] is not None])
    nde_se = np.array([se_from_ci(r["nde_ci_low"], r["nde_ci_high"])
                       for r in rows if r["nde_ci_low"] is not None])
    te_theta = np.array([r["te_mean"] for r in rows if r["te_mean"] is not None])
    te_se = np.array([se_from_ci(r["te_ci_low"], r["te_ci_high"])
                      for r in rows if r["te_ci_low"] is not None])

    nde_pool = random_effects_pool(nde_theta, nde_se)
    te_pool = random_effects_pool(te_theta, te_se)

    print("-" * 140)
    print(f"{'NDE pooled':<14}{'':>5}{'':>6}{'':>6}  "
          f"{nde_pool['theta_re']:>+14.4f}"
          f"  [{nde_pool['theta_re']-1.96*nde_pool['se_re']:+.3f}, "
          f"{nde_pool['theta_re']+1.96*nde_pool['se_re']:+.3f}]"
          f"  Q={nde_pool['Q']:.1f}  I²={nde_pool['I2_pct']:.0f}%")
    print(f"{'TE pooled':<14}{'':>5}{'':>6}{'':>6}  "
          f"{te_pool['theta_re']:>+14.4f}"
          f"  [{te_pool['theta_re']-1.96*te_pool['se_re']:+.3f}, "
          f"{te_pool['theta_re']+1.96*te_pool['se_re']:+.3f}]"
          f"  Q={te_pool['Q']:.1f}  I²={te_pool['I2_pct']:.0f}%")

    n_te_neg = sum(1 for r in rows if (r["te_ci_high"] or 1) < 0)
    n_te_pos = sum(1 for r in rows if (r["te_ci_low"] or -1) > 0)
    print(f"\nTE direction: {n_te_neg}/{len(rows)} cities with CI strictly negative "
          f"(spatial proxy mechanism present); {n_te_pos}/{len(rows)} strictly positive; "
          f"{len(rows)-n_te_neg-n_te_pos} contain zero")

    csv_path = results_dir / "counterfactual_12city_table.csv"
    cols = ["city", "n_listings", "dml_theta", "dml_se",
            "nde_mean", "nde_ci_low", "nde_ci_high", "nde_n_valid", "nde_pct",
            "te_mean", "te_ci_low", "te_ci_high", "te_n_valid", "te_pct",
            "slot_preserved", "classifier_flipped", "overall_pass"]
    with csv_path.open("w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(c) is None else str(r[c]) for c in cols) + "\n")
    print(f"\nCSV -> {csv_path}")
    pool_path = results_dir / "counterfactual_12city_pooled.json"
    pool_path.write_text(json.dumps({"nde": nde_pool, "te": te_pool}, indent=2))
    print(f"pooled JSON -> {pool_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
