"""Generate temporal 70/15/15 train/val/test splits for NYC and SF.

Splits are by sale date (NYC: sale_date; SF: last_sale_date). Earliest 70% of
sales -> train, next 15% -> val, latest 15% -> test. Boston has no sale-date
field and is omitted (paper uses all Boston parcels for training).
"""
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "release" / "splits" / "temporal_splits.csv"

CITY_DATE = {"nyc": "sale_date", "sf": "last_sale_date"}


def split_city(city: str, date_col: str) -> pd.DataFrame:
    df = pd.read_parquet(REPO / "release" / "data" / city / "parcels.parquet",
                          columns=["parcel_id", date_col])
    df = df.dropna(subset=[date_col]).copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    n = len(df)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    df["split"] = "test"
    df.loc[: n_train - 1, "split"] = "train"
    df.loc[n_train : n_train + n_val - 1, "split"] = "val"
    df["city"] = city
    return df[["city", "parcel_id", "split"]]


def main() -> None:
    parts = [split_city(c, d) for c, d in CITY_DATE.items()]
    out = pd.concat(parts, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(REPO)} ({len(out):,} rows)")
    print(out.groupby(["city", "split"]).size())


if __name__ == "__main__":
    main()
