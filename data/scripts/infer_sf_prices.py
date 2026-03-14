import sys
import pandas as pd
import numpy as np
from config import RAW_DIR

SF_DIR = RAW_DIR / "sf"


def infer_prices():
    print("Loading SF assessor rolls...")
    df = pd.read_csv(SF_DIR / "sf_assessor_rolls.csv", low_memory=False)

    df["closed_roll_year"] = pd.to_numeric(df["closed_roll_year"], errors="coerce")
    df["assessed_land_value"] = pd.to_numeric(df["assessed_land_value"], errors="coerce")
    df["assessed_improvement_value"] = pd.to_numeric(df["assessed_improvement_value"], errors="coerce")
    df["current_sales_date"] = pd.to_datetime(df["current_sales_date"], errors="coerce")

    df["total_assessed"] = df["assessed_land_value"].fillna(0) + df["assessed_improvement_value"].fillna(0)
    df = df.sort_values(["parcel_number", "closed_roll_year"])

    df["prev_assessed"] = df.groupby("parcel_number")["total_assessed"].shift(1)
    df["prev_sale_date"] = df.groupby("parcel_number")["current_sales_date"].shift(1)

    df["sale_date_changed"] = df["current_sales_date"] != df["prev_sale_date"]
    df["assessed_change_pct"] = (df["total_assessed"] - df["prev_assessed"]) / df["prev_assessed"].replace(0, np.nan)

    sales = df[
        df["sale_date_changed"]
        & df["current_sales_date"].notna()
        & (df["total_assessed"] > 0)
        & (df["assessed_change_pct"].abs() > 0.05)
    ].copy()

    sales["inferred_sale_price"] = sales["total_assessed"]
    sales = sales.sort_values("current_sales_date").drop_duplicates(
        subset=["parcel_number"], keep="last"
    )

    result = sales[["parcel_number", "inferred_sale_price", "current_sales_date"]].rename(
        columns={
            "parcel_number": "parcel_id",
            "inferred_sale_price": "sale_price",
            "current_sales_date": "sale_date",
        }
    )

    out_path = SF_DIR / "sf_inferred_prices.csv"
    result.to_csv(out_path, index=False)
    print(f"Inferred {len(result)} sale prices")
    print(f"  Median: ${result['sale_price'].median():,.0f}")
    print(f"  Mean: ${result['sale_price'].mean():,.0f}")
    print(f"  Range: ${result['sale_price'].min():,.0f} - ${result['sale_price'].max():,.0f}")
    print(f"  Saved → {out_path}")


def main():
    infer_prices()


if __name__ == "__main__":
    main()
