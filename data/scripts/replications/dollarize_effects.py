"""Translate the per-city DML theta estimates into dollar implications.

For each city, multiply theta_hat (log-price coefficient per standardised
text unit) by the city's median sale price to get a one-sigma-of-text
dollar impact. Useful for substantive-readers and the JBES referee who
asks "what is the magnitude in dollars."

Reads the three canonical rollup CSVs (Shen, pooled-PCA Baur,
counterfactual) and joins them with per-city median sale price pulled from
data/processed/{city}_listings.parquet, which covers all 12 cities (the
embeddings parquet exists for only three, so the earlier embeddings-parquet
lookup silently dropped nine markets). Writes
results/replications/effects_dollarized.csv with one row per city and a
dollar-implication column per estimator.

Because the outcome is log(price), a theta in log points maps to a dollar
figure as price * (exp(theta) - 1), not price * theta. The per-city table
reports that conversion. The summary reports the inverse-variance pooled
theta dollarized at the median city price rather than an unweighted mean of
per-city dollars, which would mix heterogeneous prices and signs and is what
produced a positive headline Baur figure despite a negative pooled theta.
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
    # Listings parquet is the unified source present for all 12 cities; the
    # embeddings parquet exists only for boston/nyc/sf.
    p = PROCESSED / f"{city}_listings.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p, columns=["price"])
    prices = pd.to_numeric(df["price"], errors="coerce").dropna()
    if prices.empty:
        return None
    return float(prices.median())


def _dollars(theta: float, price: float) -> float:
    """Dollar implication of a log-price coefficient: price * (exp(theta)-1).
    The outcome is log(price), so a theta in log points is a multiplicative
    move, not an additive one; price * theta would be a first-order error that
    grows with |theta|."""
    if np.isnan(theta) or np.isnan(price):
        return float("nan")
    return float(price * np.expm1(theta))


def _inv_var_pool(thetas, ses):
    """Fixed-effect inverse-variance pooled estimate and its SE, ignoring
    cities with missing theta/SE. Returns (theta_pooled, se_pooled, k)."""
    thetas = np.asarray(thetas, dtype=float)
    ses = np.asarray(ses, dtype=float)
    ok = np.isfinite(thetas) & np.isfinite(ses) & (ses > 0)
    if not ok.any():
        return float("nan"), float("nan"), 0
    w = 1.0 / ses[ok] ** 2
    theta_p = float(np.sum(w * thetas[ok]) / np.sum(w))
    se_p = float(np.sqrt(1.0 / np.sum(w)))
    return theta_p, se_p, int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shen", default=str(RESULTS / "shen_12city_table.csv"))
    ap.add_argument("--baur", default=str(RESULTS / "baur_pooled_pca" / "baur_pooled_pca_table.csv"))
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
        shen_se = float(shen.loc[city, "dml_se"]) if city in shen.index else float("nan")
        baur_theta = float(baur.loc[city, "dml_theta"]) if city in baur.index else float("nan")
        baur_se = float(baur.loc[city, "dml_se"]) if city in baur.index else float("nan")
        if city in cf.index:
            cf_te = float(cf.loc[city, "te_mean"])
            cf_lo = float(cf.loc[city, "te_ci_low"])
            cf_hi = float(cf.loc[city, "te_ci_high"])
            cf_se = (cf_hi - cf_lo) / (2.0 * 1.96) if np.isfinite(cf_lo) and np.isfinite(cf_hi) else float("nan")
        else:
            cf_te = float("nan"); cf_se = float("nan")
        rows.append({
            "city": city,
            "median_price_usd": price,
            "shen_theta": shen_theta,
            "shen_se": shen_se,
            "shen_dollar_per_sigma": _dollars(shen_theta, price),
            "shen_pct_per_sigma": 100.0 * np.expm1(shen_theta) if not np.isnan(shen_theta) else float("nan"),
            "baur_theta": baur_theta,
            "baur_se": baur_se,
            "baur_dollar_per_sigma": _dollars(baur_theta, price),
            "baur_pct_per_sigma": 100.0 * np.expm1(baur_theta) if not np.isnan(baur_theta) else float("nan"),
            "cf_te": cf_te,
            "cf_se": cf_se,
            "cf_dollar_per_swap": _dollars(cf_te, price),
            "cf_pct_per_swap": 100.0 * np.expm1(cf_te) if not np.isnan(cf_te) else float("nan"),
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

    n_present = len(df)
    ref_price = float(df["median_price_usd"].median())
    print(f"\nCities present: {n_present} of {len(CITY_ORDER)} requested")
    print(f"Median of per-city median prices (dollarization reference): ${ref_price:,.0f}")
    print("\nInverse-variance pooled theta, dollarized at the reference price")
    print("(an unweighted mean of per-city dollars would mix prices and signs):")
    for label, th_col, se_col in (
        ("Shen", "shen_theta", "shen_se"),
        ("Baur", "baur_theta", "baur_se"),
        ("CF",   "cf_te",      "cf_se"),
    ):
        th, se, k = _inv_var_pool(df[th_col].to_numpy(), df[se_col].to_numpy())
        pct = 100.0 * np.expm1(th) if np.isfinite(th) else float("nan")
        print(f"  {label:<5} theta={th:+.4f} (se {se:.4g}, k={k})  "
              f"{pct:+.2f}%  ${_dollars(th, ref_price):+,.0f}")

    out_csv = RESULTS / "effects_dollarized.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nCSV -> {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
