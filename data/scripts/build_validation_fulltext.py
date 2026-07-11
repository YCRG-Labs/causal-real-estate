from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "data" / "scripts"))
from scrape_remarks_full import full_remarks

OUT = REPO / "results" / "sold_scrape"
DST = OUT / "validation_fulltext.parquet"
CAP = 695
ALL_10 = ["boston", "sf", "dc", "philadelphia", "chicago", "seattle",
          "denver", "atlanta", "portland", "phoenix"]


def sample(per_city, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for c in ALL_10:
        df = pd.read_parquet(OUT / f"{c}_sold.parquet",
                             columns=["property_id", "url", "description"])
        dl = df["description"].fillna("").str.len()
        cand = df[(dl >= CAP)].dropna(subset=["url"])
        if len(cand) > per_city:
            cand = cand.iloc[rng.choice(len(cand), per_city, replace=False)]
        for _, r in cand.iterrows():
            rows.append({"city": c, "property_id": r["property_id"],
                         "url": r["url"], "gis_remarks": r["description"]})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_city", type=int, default=450)
    ap.add_argument("--delay", type=float, default=1.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    todo = sample(args.per_city, args.seed).reset_index(drop=True)
    print(f"sampled {len(todo)} truncated sold homes across {todo.city.nunique()} "
          f"metros; ~{len(todo) * args.delay / 60:.0f} min at {args.delay}s\n", flush=True)

    full, gains = [], []
    t0 = time.time()
    for i, r in todo.iterrows():
        f = full_remarks(r["url"], r["gis_remarks"])
        full.append(f)
        gains.append(len(f) - len(r["gis_remarks"]))
        time.sleep(args.delay)
        if i == 49:
            rr = np.mean([g > 5 for g in gains])
            print(f"  probe@50: recovery {rr * 100:.0f}%", flush=True)
            if rr < 0.15:
                print("  ABORT: IP challenged (recovery <15%); cooldown and retry",
                      flush=True)
                todo = todo.iloc[:len(full)]
                break
        if (i + 1) % 250 == 0:
            rr = np.mean([g > 5 for g in gains])
            d = todo.iloc[:len(full)].copy()
            d["full_remarks"] = full
            d.to_parquet(DST)
            print(f"  {i + 1}/{len(todo)} | recovery {rr * 100:.0f}% | "
                  f"mean gain +{np.mean([g for g in gains if g > 5]):.0f} | "
                  f"{time.time() - t0:.0f}s", flush=True)

    d = todo.iloc[:len(full)].copy()
    d["full_remarks"] = full
    d["gis_len"] = d["gis_remarks"].str.len()
    d["full_len"] = d["full_remarks"].str.len()
    d["gain"] = d["full_len"] - d["gis_len"]
    d.to_parquet(DST)
    rec = d[d["gain"] > 5]
    print(f"\nDONE {len(d)} homes in {(time.time() - t0) / 60:.0f} min | "
          f"recovered {100 * len(rec) / len(d):.0f}% | "
          f"on recovered: mean gain +{rec['gain'].mean():.0f} median +{rec['gain'].median():.0f} "
          f"full-len median {rec['full_len'].median():.0f}")
    print("per-city recovery:")
    for c, g in d.groupby("city"):
        r = g[g["gain"] > 5]
        print(f"  {c:13s} n={len(g):4d} recovered {100 * len(r) / len(g):3.0f}% "
              f"mean gain +{r['gain'].mean() if len(r) else 0:.0f}")
    print(f"\nwrote {DST}")


if __name__ == "__main__":
    main()
