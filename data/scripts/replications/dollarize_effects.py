"""Translate the per-city DML theta estimates into dollar implications.

For each city, multiply theta_hat (log-price coefficient per standardised
text unit) by the city's median sale price to get a one-sigma-of-text
dollar impact. Useful for substantive-readers and the JBES referee who
asks "what is the magnitude in dollars."

Reads the three existing rollup CSVs (Shen, Baur, counterfactual) and
joins them with per-city median price pulled from the embeddings parquet.
Writes results/replications/effects_dollarized.csv with one row per city
and columns for each estimator's dollar-implication estimate.

A 1-sigma move in text-uniqueness corresponds to about a sigma_T standard
deviation in the text feature. Reporting "theta x median_price" is the
on-conventional-scale magnitude that maps cleanly to "what fraction of
the typical home price does the text channel account for."
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "results" / "replications"
CF_RESULTS = REPO / "results" / "counterfactual"
PROCESSED = REPO / "data" / "processed"

CITY_ORDER = ["boston", "nyc", "sf", "dc", "philadelphia", "chicago",
              "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]
DISPLAY = {"boston": "Boston", "nyc": "New York", "sf": "San Francisco",
           "dc": "Washington DC", "philadelphia": "Philadelphia",
           "chicago": "Chicago", "seattle": "Seattle", "denver": "Denver",
           "atlanta": "Atlanta", "portland": "Portland", "phoenix": "Phoenix",
           "dallas": "Dallas"}


def _median_price(city: str) -> float | None:
    p = PROCESSED / f"{city}_embeddings.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p, columns=["price"])
    return float(df["price"].dropna().median())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shen", default=str(RESULTS / "shen_12city_table.csv"))
    ap.add_argument("--baur", default=str(RESULTS / "baur_12city_table.csv"))
    ap.add_argument("--cf",   default=str(CF_RESULTS / "counterfactual_12city_table.csv"))
    args = ap.parse_args()

    shen = pd.read_csv(args.shen).set_index("city")
    baur = pd.read_csv(args.baur).set_index("city")
    cf = pd.read_csv(args.cf).set_index("city")

    rows = []
    for city in CITY_ORDER:
        price = _median_price(city)
        if price is None:
            print(f"  [warn] missing parquet for {city}", file=sys.stderr); continue
        shen_theta = float(shen.loc[city, "dml_theta"]) if city in shen.index else float("nan")
        baur_theta = float(baur.loc[city, "dml_theta"]) if city in baur.index else float("nan")
        cf_te = float(cf.loc[city, "te_mean"]) if city in cf.index else float("nan")
        rows.append({
            "city": city,
            "median_price_usd": price,
            "shen_theta": shen_theta,
            "shen_dollar_per_sigma": shen_theta * price if not np.isnan(shen_theta) else float("nan"),
            "shen_pct_per_sigma": 100.0 * (np.exp(shen_theta) - 1.0) if not np.isnan(shen_theta) else float("nan"),
            "baur_theta": baur_theta,
            "baur_dollar_per_sigma": baur_theta * price if not np.isnan(baur_theta) else float("nan"),
            "baur_pct_per_sigma": 100.0 * (np.exp(baur_theta) - 1.0) if not np.isnan(baur_theta) else float("nan"),
            "cf_te": cf_te,
            "cf_dollar_per_swap": cf_te * price if not np.isnan(cf_te) else float("nan"),
            "cf_pct_per_swap": 100.0 * (np.exp(cf_te) - 1.0) if not np.isnan(cf_te) else float("nan"),
        })

    df = pd.DataFrame(rows)
    print()
    print(f"{'city':<14}  {'median $':>12}  "
          f"{'Shen θ':>9}{'$':>11}{'%':>7}  "
          f"{'Baur θ':>9}{'$':>11}{'%':>7}  "
          f"{'CF TE':>9}{'$':>11}{'%':>7}")
    print("-" * 130)
    for _, r in df.iterrows():
        print(f"{DISPLAY[r['city']]:<14}  ${r['median_price_usd']:>11,.0f}  "
              f"{r['shen_theta']:>+9.4f}${r['shen_dollar_per_sigma']:>+9,.0f}{r['shen_pct_per_sigma']:>+6.2f}%  "
              f"{r['baur_theta']:>+9.4f}${r['baur_dollar_per_sigma']:>+9,.0f}{r['baur_pct_per_sigma']:>+6.2f}%  "
              f"{r['cf_te']:>+9.4f}${r['cf_dollar_per_swap']:>+9,.0f}{r['cf_pct_per_swap']:>+6.2f}%")

    print(f"\nMean median price across 12 cities: ${df['median_price_usd'].mean():,.0f}")
    print(f"Mean Shen dollar/sigma:  ${df['shen_dollar_per_sigma'].mean():+,.0f}")
    print(f"Mean Baur dollar/sigma:  ${df['baur_dollar_per_sigma'].mean():+,.0f}")
    print(f"Mean CF dollar/swap:     ${df['cf_dollar_per_swap'].mean():+,.0f}")

    out_csv = RESULTS / "effects_dollarized.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nCSV -> {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
