"""Roll up the 12-city LEACE (+ optional IGBP) JSONs into a paper-ready table.

Reads results/leace12/{city}_{variant}.json for each of the 12 cities and
emits:
  - per-city row: n_train, n_test, routing (continuous lat/lon vs ZIP),
                  Ridge R^2(lat/lon) raw -> erased,
                  MLP   R^2(lat/lon) raw -> erased,
                  IGBP last-iter adversary MSE (when --postnl_igbp was used),
                  linear_guardedness_holds (Ridge R^2_erased < 0.05)
  - pooled summary: mean / max erased Ridge and MLP R^2 across cities,
                    count of cities where guardedness holds
  - CSV at results/leace12/leace_12city_table.csv

If categorical probes were also computed (LogReg ZIP accuracy and MDL),
they're appended as a second block with raw vs erased.
"""
from __future__ import annotations

import argparse
import json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="leace", choices=["leace", "splince"])
    ap.add_argument("--results_dir", default=str(REPO / "results" / "leace12"))
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    rows = []
    missing = []
    for city in CITY_ORDER:
        p = results_dir / f"{city}_{args.variant}.json"
        if not p.exists():
            missing.append(city); continue
        d = json.loads(p.read_text())
        ridge_raw = _pull(d, "continuous", "raw", "r2_avg_test")
        ridge_erase = _pull(d, "continuous", "erased", "r2_avg_test")
        mlp_raw = _pull(d, "mlp_continuous", "raw", "r2_avg_test")
        mlp_erase = _pull(d, "mlp_continuous", "erased", "r2_avg_test")
        igbp_hist = d.get("igbp_history") or []
        igbp_last = (igbp_hist[-1].get("adv_train_mse_postproj")
                     if igbp_hist else None)
        cat_raw = _pull(d, "categorical", "raw", "linear_acc")
        cat_erase = _pull(d, "categorical", "erased", "linear_acc")
        mdl_raw = _pull(d, "mdl_codelength", "raw", "compression")
        mdl_erase = _pull(d, "mdl_codelength", "erased", "compression")
        rows.append({
            "city": city,
            "n_train": int(d.get("n_train", 0)),
            "n_test": int(d.get("n_test", 0)),
            "routing": d.get("routing", "?"),
            "n_zips": int(d.get("n_zips", 0)),
            "guardedness_residual": d.get("guardedness_residual"),
            "ridge_r2_raw": ridge_raw,
            "ridge_r2_erased": ridge_erase,
            "mlp_r2_raw": mlp_raw,
            "mlp_r2_erased": mlp_erase,
            "igbp_last_mse": igbp_last,
            "cat_acc_raw": cat_raw,
            "cat_acc_erased": cat_erase,
            "mdl_compression_raw": mdl_raw,
            "mdl_compression_erased": mdl_erase,
            "linear_guardedness_holds": bool(d.get("linear_guardedness_holds")),
        })

    if missing:
        print(f"warning: missing JSONs for {missing}", file=sys.stderr)
    if not rows:
        print("no JSONs read; aborting", file=sys.stderr); return 1

    print()
    print(f"{'city':<14}{'n_tr':>5}{'n_te':>5}  {'route':<5}  "
          f"{'Ridge R^2(raw→erased)':>26}  "
          f"{'MLP R^2(raw→erased)':>24}  "
          f"{'IGBP':>7}  {'guard':>5}")
    print("-" * 100)
    for r in rows:
        route = "cont" if r["routing"] == "continuous_latlon" else "cat"
        ridge = f"{(r['ridge_r2_raw'] or 0):>+7.3f} → {(r['ridge_r2_erased'] or 0):>+7.3f}"
        mlp = f"{(r['mlp_r2_raw'] or 0):>+7.3f} → {(r['mlp_r2_erased'] or 0):>+7.3f}"
        igbp = f"{(r['igbp_last_mse'] or 0):>7.3f}" if r["igbp_last_mse"] is not None else "    —  "
        guard = "Y" if r["linear_guardedness_holds"] else "n"
        print(f"{DISPLAY[r['city']]:<14}{r['n_train']:>5}{r['n_test']:>5}  "
              f"{route:<5}  {ridge:>26}  {mlp:>24}  {igbp:>7}  {guard:>5}")

    print("-" * 100)
    arr_ridge_erase = np.array([r["ridge_r2_erased"] for r in rows
                                if r["ridge_r2_erased"] is not None])
    arr_mlp_erase = np.array([r["mlp_r2_erased"] for r in rows
                              if r["mlp_r2_erased"] is not None])
    n_guard = sum(1 for r in rows if r["linear_guardedness_holds"])
    print(f"{'POOLED (mean)':<14}{'':>5}{'':>5}  {'':<5}  "
          f"{'':>10}{arr_ridge_erase.mean():>+10.3f}  "
          f"{'':>10}{arr_mlp_erase.mean():>+10.3f}  "
          f"{'':>7}  {n_guard:>2}/{len(rows)}")
    print(f"{'POOLED (max) ':<14}{'':>5}{'':>5}  {'':<5}  "
          f"{'':>10}{arr_ridge_erase.max():>+10.3f}  "
          f"{'':>10}{arr_mlp_erase.max():>+10.3f}")

    has_cat = any(r.get("cat_acc_raw") is not None for r in rows)
    if has_cat:
        print(f"\n{'city':<14}  ZIP-LogReg acc (raw→erased)  MDL compression (raw→erased)")
        print("-" * 90)
        for r in rows:
            if r["cat_acc_raw"] is None:
                continue
            cat_str = f"{r['cat_acc_raw']:.3f} → {r['cat_acc_erased']:.3f}"
            mdl_str = (f"{r['mdl_compression_raw']:.2f}x → {r['mdl_compression_erased']:.2f}x"
                       if r["mdl_compression_raw"] is not None else "—")
            print(f"{DISPLAY[r['city']]:<14}  {cat_str:<28}  {mdl_str}")

    csv_path = results_dir / "leace_12city_table.csv"
    cols = ["city", "n_train", "n_test", "routing", "n_zips",
            "guardedness_residual",
            "ridge_r2_raw", "ridge_r2_erased",
            "mlp_r2_raw", "mlp_r2_erased",
            "igbp_last_mse",
            "cat_acc_raw", "cat_acc_erased",
            "mdl_compression_raw", "mdl_compression_erased",
            "linear_guardedness_holds"]
    with csv_path.open("w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(c) is None else str(r[c])
                              for c in cols) + "\n")
    print(f"\nCSV -> {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
