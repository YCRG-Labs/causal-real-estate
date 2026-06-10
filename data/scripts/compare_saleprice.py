"""Sale-price validity check: does the per-market text-price effect replicate when
the outcome is the realized recorded SALE price instead of the agent's asking
(listing) price? Covers the four markets that publish recorded sales
(Philadelphia, Chicago, DC, NYC).

Holds the treatment fixed (oriented PC1 of the embedding) and the controls fixed
(property attributes + lat/lon), and estimates the DML effect on (a) log listing
price and (b) log recorded sale price, the latter with sale-quarter fixed effects
because recorded sales span years in nominal dollars. Agreement says the text
effect is a property/market signal, not an artifact of the agent pricing the
language; a much smaller sale-price effect would flag the asking-price
simultaneity.

Run on Brev (needs the embeddings):
  python3 data/scripts/compare_saleprice.py
Writes results/robustness/saleprice_vs_listprice.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data" / "processed"
sys.path.insert(0, str(REPO / "data" / "scripts"))
sys.path.insert(0, str(REPO / "data" / "scripts" / "replications"))
from spatial_basis import quadratic_basis
from meta_pooling import re_meta

MARKETS = ["philadelphia", "chicago", "dc", "nyc"]
PROP = ["beds", "baths", "sqft", "year_built", "lot_size"]
_AL = np.logspace(-3, 3, 25)


def _pc1(emb, y, seed=42):
    p = PCA(n_components=1, random_state=seed).fit_transform(emb - emb.mean(0))[:, 0]
    if np.corrcoef(p, y)[0, 1] < 0:
        p = -p
    return (p - p.mean()) / (p.std(ddof=1) or 1.0)


def _dml(T, C, Y, k=5, seed=42):
    C = StandardScaler().fit_transform(np.asarray(C, float))
    T = (T - T.mean()) / (T.std() or 1.0)
    n = len(Y)
    Tr, Yr = np.zeros(n), np.zeros(n)
    for tr, te in KFold(k, shuffle=True, random_state=seed).split(np.arange(n)):
        Tr[te] = T[te] - RidgeCV(alphas=_AL).fit(C[tr], T[tr]).predict(C[te])
        Yr[te] = Y[te] - RidgeCV(alphas=_AL).fit(C[tr], Y[tr]).predict(C[te])
    den = np.mean(Tr ** 2)
    if den <= 0:
        return float("nan"), float("nan")
    th = np.mean(Tr * Yr) / den
    psi = (Yr - th * Tr) * Tr / den
    return float(th), float(np.sqrt(np.var(psi) / n))


def run_city(city):
    ep = PROC / f"{city}_embeddings.parquet"
    spp = PROC / f"{city}_listings_saleprice.parquet"
    if not ep.exists() or not spp.exists():
        print(f"  {city}: missing embeddings or saleprice join; skip")
        return None
    e = pd.read_parquet(ep)
    e["sha16"] = e["source_html_sha256"].astype(str).str[:16]
    emb_cols = [c for c in e.columns if c.startswith("emb_")]
    sp = pd.read_parquet(spp)
    L = pd.read_parquet(PROC / f"{city}_listings.parquet")
    L["sha16"] = L["source_html_sha256"].astype(str).str[:16]
    have = [c for c in PROP if c in L.columns]

    df = e.merge(sp, on="sha16", how="inner").merge(
        L[["sha16"] + have].drop_duplicates("sha16"), on="sha16", how="left")
    df = df[(df["sale_price"] > 1e4) & (df["list_price"] > 1e4)]
    df = df[np.isfinite(df["lat"]) & np.isfinite(df["lon"])].reset_index(drop=True)
    if len(df) < 150:
        print(f"  {city}: only {len(df)} matched; skip")
        return None

    T = _pc1(df[emb_cols].to_numpy(float), np.log(df["list_price"].to_numpy(float)))
    prop = np.column_stack([pd.to_numeric(df[c], errors="coerce") for c in have]) if have else np.empty((len(df), 0))
    for j in range(prop.shape[1]):
        col = prop[:, j]
        prop[~np.isfinite(col), j] = np.nanmedian(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0
    geo = quadratic_basis(df["lat"].to_numpy(), df["lon"].to_numpy())
    Cbase = np.column_stack([p for p in (prop, geo) if p.shape[1]])
    q = pd.to_datetime(df["sale_date"], errors="coerce", utc=True)
    qd = pd.get_dummies(q.dt.year.astype("Int64").astype(str) + "Q" + q.dt.quarter.astype("Int64").astype(str),
                        drop_first=True).to_numpy(float)

    th_list, se_list = _dml(T, Cbase, np.log(df["list_price"].to_numpy(float)))
    th_sale, se_sale = _dml(T, np.column_stack([Cbase, qd]), np.log(df["sale_price"].to_numpy(float)))
    print(f"  {city:13s} n={len(df):5d}  list θ={th_list:+.3f} (se {se_list:.3f})  "
          f"sale θ={th_sale:+.3f} (se {se_sale:.3f})  ratio list/sale med {(df['list_price']/df['sale_price']).median():.2f}")
    return {"city": city, "n": len(df), "theta_list": th_list, "se_list": se_list,
            "theta_sale": th_sale, "se_sale": se_sale}


def main():
    rows = [r for r in (run_city(c) for c in MARKETS) if r]
    if not rows:
        raise SystemExit("no markets ran (need embeddings + saleprice joins)")
    df = pd.DataFrame(rows)
    out = REPO / "results" / "robustness" / "saleprice_vs_listprice.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    for col, lab in [("theta_list", "listing price"), ("theta_sale", "sale price")]:
        se = f"se_{col.split('_')[1]}"
        if len(df) >= 3:
            r = re_meta(df[col].to_numpy(), (df[se].to_numpy()) ** 2, n=df["n"].to_numpy())
            print(f"\npooled {lab}: RE μ={r['mu']:+.3f}  HKSJ[{r['ci_HKSJ'][0]:+.3f},{r['ci_HKSJ'][1]:+.3f}]")
    print(f"\nCSV -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
