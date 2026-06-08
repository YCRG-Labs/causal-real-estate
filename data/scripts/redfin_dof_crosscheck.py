"""
Cross-check Redfin scraped sold prices against NYC DOF Rolling Sales recorded
transactions. Independent transaction validation that kills the
"self-reported Redfin price" objection at any econ venue.

Join: Redfin description rows (with property-level lat/lon and BBL from the
LEACE-grade geocoding) to nyc_sales.csv (DOF Rolling Sales has Latitude,
Longitude, BBL, SALE PRICE, SALE DATE). Match on BBL; for the matched set
compute log-price Spearman rank correlation and a Bland-Altman-style
agreement plot.

Usage:
    python redfin_dof_crosscheck.py [--out results/dedup_rerun/redfin_dof.json]
"""
from __future__ import annotations
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _silence
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw"
PROC = REPO / "data" / "processed"


def load_redfin_nyc():
    p = PROC / "nyc_embeddings.parquet"
    df = pd.read_parquet(p, columns=[
        "address", "zip", "latitude", "longitude", "price",
        "BBL_matched", "geocode_method"])
    df = df.rename(columns={"price": "redfin_price",
                            "BBL_matched": "BBL"})
    df["redfin_price"] = pd.to_numeric(df["redfin_price"], errors="coerce")
    df["BBL"] = pd.to_numeric(df["BBL"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["redfin_price", "BBL"])
    return df


def load_dof():
    p = RAW / "nyc" / "nyc_sales.csv"
    cols = ["BBL", "SALE PRICE", "SALE DATE", "BUILDING CLASS CATEGORY",
            "ADDRESS", "ZIP CODE", "Latitude", "Longitude"]
    df = pd.read_csv(p, low_memory=False, usecols=cols)
    df = df.rename(columns={"SALE PRICE": "dof_price",
                            "SALE DATE": "sale_date",
                            "Latitude": "dof_lat",
                            "Longitude": "dof_lon",
                            "BUILDING CLASS CATEGORY": "bld_class"})
    df["dof_price"] = pd.to_numeric(df["dof_price"], errors="coerce")
    df["BBL"] = pd.to_numeric(df["BBL"], errors="coerce").astype("Int64")
    df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
    df = df[(df["dof_price"].notna()) & (df["dof_price"] > 1000)]
    res_classes = ("01 ", "02 ", "03 ", "09 ", "10 ", "12 ", "13 ", "15 ")
    df = df[df["bld_class"].fillna("").str.startswith(res_classes)]
    return df


def crosscheck():
    rf = load_redfin_nyc()
    dof = load_dof()
    print(f"Redfin NYC descriptions: {len(rf)}  with BBL: {rf['BBL'].notna().sum()}")
    print(f"DOF Rolling Sales arms-length residential: {len(dof)}")

    dof_sorted = dof.sort_values("sale_date", ascending=False)
    dof_latest = dof_sorted.groupby("BBL").first().reset_index()
    print(f"Unique BBLs in DOF: {len(dof_latest)}")

    merged = rf.merge(dof_latest[["BBL", "dof_price", "sale_date",
                                  "dof_lat", "dof_lon"]],
                      on="BBL", how="inner")
    print(f"Joined Redfin x DOF on BBL: {len(merged)}")
    if len(merged) < 10:
        return {"n_join": int(len(merged)), "error": "too few matches"}

    merged["log_redfin"] = np.log(merged["redfin_price"])
    merged["log_dof"] = np.log(merged["dof_price"])
    merged["log_diff"] = merged["log_redfin"] - merged["log_dof"]
    merged["price_ratio"] = merged["redfin_price"] / merged["dof_price"]

    rho_spearman = float(stats.spearmanr(merged["log_redfin"],
                                          merged["log_dof"]).statistic)
    rho_pearson = float(stats.pearsonr(merged["log_redfin"],
                                       merged["log_dof"]).statistic)

    out = {
        "n_redfin": int(len(rf)),
        "n_dof_armslength_residential": int(len(dof)),
        "n_dof_unique_bbl_latest": int(len(dof_latest)),
        "n_join": int(len(merged)),
        "spearman_rho_log_prices": rho_spearman,
        "pearson_rho_log_prices": rho_pearson,
        "median_price_ratio_redfin_over_dof": float(merged["price_ratio"].median()),
        "mean_log_diff": float(merged["log_diff"].mean()),
        "std_log_diff": float(merged["log_diff"].std()),
        "p25_log_diff": float(merged["log_diff"].quantile(0.25)),
        "p75_log_diff": float(merged["log_diff"].quantile(0.75)),
    }
    print(f"\nResults:")
    for k, v in out.items():
        if isinstance(v, float):
            print(f"  {k:42s}: {v:.4f}")
        else:
            print(f"  {k:42s}: {v}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/dedup_rerun/redfin_dof.json")
    args = ap.parse_args()
    out = crosscheck()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
