from __future__ import annotations

import argparse
import html
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "sold_scrape"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
HDR = {"User-Agent": UA, "Accept": "text/html,application/xhtml+xml"}
CAP = 695
ALL_10 = ["boston", "sf", "dc", "philadelphia", "chicago", "seattle",
          "denver", "atlanta", "portland", "phoenix"]


def _page(url, timeout=25):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HDR),
                                timeout=timeout) as r:
        return r.read().decode("utf-8", "ignore")


def full_remarks(url, gis_remarks, retries=3):
    if not url.startswith("http"):
        url = "https://www.redfin.com" + url
    stub = re.escape((gis_remarks or "")[20:50])
    if len(stub) < 20:
        return gis_remarks
    pat = re.compile(r':\s*"((?:[^"\\]|\\.)*' + stub + r'(?:[^"\\]|\\.)*)"')
    for i in range(retries):
        try:
            page = _page(url)
        except Exception:
            time.sleep(1.2 * (i + 1))
            continue
        best = gis_remarks
        for m in pat.finditer(page):
            raw = m.group(1)
            try:
                dec = pd.io.json.loads('"' + raw + '"') if False else _decode(raw)
            except Exception:
                continue
            dec = html.unescape(dec)
            if dec.startswith((gis_remarks or "")[:60]) and len(dec) > len(best):
                best = dec
        return best
    return gis_remarks


def _decode(raw):
    import json
    try:
        return json.loads('"' + raw + '"')
    except Exception:
        return raw.encode().decode("unicode_escape", errors="ignore")


def backfill(city, delay, workers):
    dst = OUT / f"{city}_sold.parquet"
    df = pd.read_parquet(dst)
    if "description_full" not in df.columns:
        df["description_full"] = df["description"]
    dl = df["description"].fillna("").str.len()
    todo = df.index[(dl >= CAP) & (df["description_full"].fillna("").str.len()
                                   <= dl + 1)].tolist()
    print(f"{city}: {len(todo)} truncated rows to backfill", flush=True)
    done = [0]

    def work(i):
        r = df.loc[i]
        full = full_remarks(r["url"], r["description"])
        time.sleep(delay)
        return i, full

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, i) for i in todo]
        for f in as_completed(futs):
            i, full = f.result()
            df.at[i, "description_full"] = full
            done[0] += 1
            if done[0] % 500 == 0:
                gained = (df["description_full"].str.len()
                          - df["description"].str.len()).clip(lower=0)
                print(f"  {city}: {done[0]}/{len(todo)} | mean gain "
                      f"+{gained[todo].mean():.0f} chars", flush=True)
    df.to_parquet(dst)
    g = (df["description_full"].str.len() - df["description"].str.len()).clip(lower=0)
    print(f"{city}: done, mean gain on truncated +{g[todo].mean():.0f} chars, "
          f"recovered {(g[todo] > 5).mean() * 100:.0f}% of them", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--delay", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    for c in (ALL_10 if args.all else [args.city]):
        backfill(c, args.delay, args.workers)


if __name__ == "__main__":
    main()
