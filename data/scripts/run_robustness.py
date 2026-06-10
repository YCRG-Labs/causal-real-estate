"""Revision robustness runner: does the per-market text-price effect survive
(a) a flexible spatial basis (TPRS) replacing the lat/lon quadratic, and
(b) sale-time fixed effects recovered from the cached HTML?

Self-contained: needs only data/processed/<city>_embeddings.parquet (sha key),
<city>_listings.parquet (property attrs by sha), and <city>_sold_dates.parquet
(sale_year/quarter by sha). Holds the treatment fixed (oriented PC1 of the
embedding) and toggles the control set, in the manner of the paper's Gelbach
decomposition. The structured controls here are property attributes + ZIP
demeaning; this isolates the spatial-basis and time-FE deltas, which are the
quantities under review. Run the full census/crime/amenity confounder version
through leace_price_decomposition.py once these deltas are understood.

  python3 data/scripts/run_robustness.py --all_12
  python3 data/scripts/run_robustness.py --city nyc --ks 10,30,50

Writes results/robustness/timefe_spatial.csv and prints a pooled mKH summary.
"""
from __future__ import annotations

import argparse
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
from spatial_basis import thin_plate_basis, quadratic_basis
sys.path.insert(0, str(REPO / "data" / "scripts" / "replications"))
from meta_pooling import re_meta

ALL_12 = ["boston", "nyc", "sf", "dc", "philadelphia", "chicago",
          "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]
PROP_COLS = ["beds", "baths", "sqft", "year_built", "lot_size"]
_ALPHAS = np.logspace(-3, 3, 25)


def _oriented_pc1(emb, y, seed=42):
    p = PCA(n_components=1, random_state=seed).fit_transform(emb - emb.mean(0))[:, 0]
    if np.corrcoef(p, y)[0, 1] < 0:
        p = -p
    return (p - p.mean()) / (p.std(ddof=1) or 1.0)


def _demean_by_group(M, g):
    df = pd.DataFrame(np.atleast_2d(M).reshape(len(g), -1))
    return (df - df.groupby(np.asarray(g)).transform("mean")).to_numpy()


def _dml_ridge(T, C, Y, k=5, seed=42):
    C = StandardScaler().fit_transform(np.asarray(C, float))
    T = (T - T.mean()) / (T.std() or 1.0)
    n = len(Y)
    Tr = np.zeros(n)
    Yr = np.zeros(n)
    for tr, te in KFold(k, shuffle=True, random_state=seed).split(np.arange(n)):
        Tr[te] = T[te] - RidgeCV(alphas=_ALPHAS).fit(C[tr], T[tr]).predict(C[te])
        Yr[te] = Y[te] - RidgeCV(alphas=_ALPHAS).fit(C[tr], Y[tr]).predict(C[te])
    den = np.mean(Tr ** 2)
    if den <= 0:
        return float("nan"), float("nan")
    th = np.mean(Tr * Yr) / den
    psi = (Yr - th * Tr) * Tr / den
    return float(th), float(np.sqrt(np.var(psi) / n))


def _dummies(labels):
    return pd.get_dummies(pd.Series(labels), drop_first=True).to_numpy(dtype=float)


def load_market(city):
    ep = PROC / f"{city}_embeddings.parquet"
    if not ep.exists():
        return None
    e = pd.read_parquet(ep)
    emb_cols = [c for c in e.columns if c.startswith("emb_")]
    if "source_html_sha256" not in e.columns or not emb_cols:
        return None
    e["sha16"] = e["source_html_sha256"].astype(str).str[:16]

    L = pd.read_parquet(PROC / f"{city}_listings.parquet")
    L["sha16"] = L["source_html_sha256"].astype(str).str[:16]
    have = [c for c in PROP_COLS if c in L.columns]
    SD = pd.read_parquet(PROC / f"{city}_sold_dates.parquet")[["sha16", "sale_year", "sale_quarter"]]

    df = e.merge(L[["sha16"] + have].drop_duplicates("sha16"), on="sha16", how="left")
    df = df.merge(SD.drop_duplicates("sha16"), on="sha16", how="left")

    price = pd.to_numeric(df["price"], errors="coerce")
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    ok = (price > 1e4) & (price < 1e8) & lat.notna() & lon.notna() & df["sale_year"].notna()
    df = df[ok].reset_index(drop=True)
    if len(df) < 100:
        return None

    prop = np.column_stack([pd.to_numeric(df[c], errors="coerce") for c in have]) if have else np.empty((len(df), 0))
    prop = np.where(np.isfinite(prop), prop, np.nan)
    for j in range(prop.shape[1]):
        col = prop[:, j]
        prop[~np.isfinite(col), j] = np.nanmedian(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0
    return {
        "city": city, "n": len(df),
        "T": df[emb_cols].to_numpy(float),
        "Y": np.log(pd.to_numeric(df["price"], errors="coerce").to_numpy(float)),
        "lat": pd.to_numeric(df["latitude"], errors="coerce").to_numpy(),
        "lon": pd.to_numeric(df["longitude"], errors="coerce").to_numpy(),
        "prop": prop,
        "zip": df["zip"].astype(str).to_numpy() if "zip" in df.columns else None,
        "year": df["sale_year"].astype(int).to_numpy(),
        "quarter": df["sale_quarter"].astype(str).to_numpy(),
    }


def run_city(m, ks, use_zip):
    pc = _oriented_pc1(m["T"], m["Y"])
    geo = {
        "none": np.empty((m["n"], 0)),
        "quad": quadratic_basis(m["lat"], m["lon"]),
    }
    for k in ks:
        geo[f"tprs{k}"] = thin_plate_basis(m["lat"], m["lon"], k=k)

    rows = []
    for gname, G in geo.items():
        for tname, TFE in [("none", None), ("quarter", _dummies(m["quarter"]))]:
            parts = [m["prop"], G]
            if TFE is not None:
                parts.append(TFE)
            C = np.column_stack([p for p in parts if p.shape[1] > 0]) if any(p.shape[1] for p in parts) else np.ones((m["n"], 1))
            T, Y = pc.copy(), m["Y"].copy()
            if use_zip and m["zip"] is not None:
                Y = _demean_by_group(Y, m["zip"]).ravel()
                T = _demean_by_group(T, m["zip"]).ravel()
                C = _demean_by_group(C, m["zip"])
            th, se = _dml_ridge(T, C, Y)
            rows.append({"city": m["city"], "n": m["n"], "spatial": gname,
                         "time": tname, "theta": th, "se": se})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city")
    ap.add_argument("--all_12", action="store_true")
    ap.add_argument("--ks", default="30", help="comma list of TPRS ranks, e.g. 10,30,50")
    ap.add_argument("--zip", action="store_true", help="demean Y,T,C within ZIP (FWL fixed effects)")
    ap.add_argument("--headline", default="tprs30:quarter",
                    help="spatial:time spec to pool across markets")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "robustness" / "timefe_spatial.csv")
    args = ap.parse_args()

    cities = ALL_12 if args.all_12 else [args.city]
    if any(c is None for c in cities):
        raise SystemExit("specify --city or --all_12")
    ks = [int(x) for x in args.ks.split(",") if x.strip()]

    all_rows = []
    for c in cities:
        m = load_market(c)
        if m is None:
            print(f"  {c}: skipped (missing embeddings/sha key or n<100)")
            continue
        rows = run_city(m, ks, args.zip)
        all_rows.extend(rows)
        base = next(r for r in rows if r["spatial"] == "quad" and r["time"] == "none")
        head = next((r for r in rows if f"{r['spatial']}:{r['time']}" == args.headline), base)
        print(f"  {c:13s} n={m['n']:6d}  quad/none θ={base['theta']:+.4f}  "
              f"{args.headline} θ={head['theta']:+.4f}  Δ={head['theta']-base['theta']:+.4f}")

    df = pd.DataFrame(all_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nCSV -> {args.out}")

    sp, tm = args.headline.split(":")
    head = df[(df.spatial == sp) & (df.time == tm) & df.theta.notna()]
    base = df[(df.spatial == "quad") & (df.time == "none") & df.theta.notna()]
    for label, d in [("quad / no-time (current paper basis)", base),
                     (f"{args.headline} (TPRS + sale-quarter FE)", head)]:
        if len(d) >= 3:
            r = re_meta(d.theta.to_numpy(), (d.se.to_numpy()) ** 2, n=d.n.to_numpy())
            print(f"\n== pooled: {label}  (K={r['k']}, I2={r['I2']:.0f}%) ==")
            print(f"   RE μ={r['mu']:+.4f}  HKSJ 95% [{r['ci_HKSJ'][0]:+.4f},{r['ci_HKSJ'][1]:+.4f}]")
            print(f"   mKH 95% [{r['ci_mKH'][0]:+.4f},{r['ci_mKH'][1]:+.4f}]  (mKH binds: {r['mkh_binds']})")
            print(f"   listing-weighted FE μ={r['FE_mu']:+.4f}  size-wtd μ={r['N_mu']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
