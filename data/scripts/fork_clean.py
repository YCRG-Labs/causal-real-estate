"""Clean assessor-vs-listing fork with single-unit restriction + Cinelli-Hazlett RV.

Fixes the multi-unit matching problem: in condo-heavy markets the nearest assessor
parcel is a whole building, so its bedroom/sqft count is the building's, not the
unit's. Restricting to single-family + townhouse (parcel == unit) makes the
assessor attribute an unambiguous match.

For each market with assessor data, on the single-unit subset:
  ASSESSOR arm : control for exogenous county attributes (the defensible causal spec)
  LISTING  arm : control for agent-entered attributes (the M-bias upper bound)
Reports each theta, whether the CI covers zero, and the Cinelli-Hazlett robustness
value RV_{q=1} of the assessor-arm estimate (the partial-R^2 an unobserved
confounder would need with both treatment and outcome to drive it to zero).

    python data/scripts/fork_clean.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "data" / "scripts"))
from replications.compare_to_dml import run_dml
from causal_inference import load_analysis_data, get_features_and_target

PROC = REPO / "data" / "processed"
FULL = REPO / "results" / "experiment_length_full"
CENSUS = REPO / "results" / "census_bg"
ASSR = REPO / "results" / "assessor"
ALL_12 = ["boston", "nyc", "sf", "dc", "philadelphia", "chicago",
          "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]
EMB = [f"emb_{i}" for i in range(768)]
DEMO = ["median_household_income", "median_home_value", "median_gross_rent",
        "pct_white", "pct_black", "pct_asian", "pct_hispanic", "pct_bachelors",
        "labor_force_participation", "pct_under_25", "pct_over_60"]
# assessor building-sqft/bed column names differ per market's parcel file
PARCEL_COLS = {
    "boston": (32619, "bedrooms", "living_area_sqft"),
    "nyc": (32618, "bedrooms", "res_area_sqft"),
    "sf": (32610, "bedrooms", "property_area_sqft"),
    "philadelphia": (32618, "bedrooms", "bldg_area_sqft"),
}
UTM = {"boston": 32619, "nyc": 32618, "sf": 32610, "dc": 32618, "philadelphia": 32618,
       "chicago": 32616, "seattle": 32610, "denver": 32613, "atlanta": 32616,
       "portland": 32610, "phoenix": 32612, "dallas": 32614}


def imp(a):
    a = np.array(a, float, copy=True)
    m = np.nan_to_num(np.nanmedian(a, axis=0), nan=0.0)
    ix = np.where(np.isnan(a))
    a[ix] = np.take(m, ix[1] if a.ndim > 1 else 0)
    return np.nan_to_num(a, nan=0.0)


def winsor(a, k=5.0):
    a = np.array(a, float, copy=True)
    for j in range(a.shape[1]):
        col = a[:, j]; med = np.nanmedian(col)
        mad = np.nanmedian(np.abs(col - med)) or (np.nanstd(col) or 1.0)
        lo, hi = med - k * 1.4826 * mad, med + k * 1.4826 * mad
        a[:, j] = np.clip(col, lo, hi)
    return a


def sanitize(emb):
    if "zip" in emb.columns:
        z = emb["zip"].astype("string").str.strip().str.slice(0, 5)
        emb = emb.copy()
        emb["zip"] = z.replace("", pd.NA).astype("Float64")
    return emb


def pooled_treatment():
    blocks, sizes = [], []
    for c in ALL_12:
        r = pd.read_parquet(FULL / f"{c}_reembed.parquet").sort_values("row").reset_index(drop=True)
        blocks.append(r[EMB].to_numpy(np.float32))
        sizes.append((c, len(r)))
    Xc = np.vstack([b - b.mean(0, keepdims=True) for b in blocks]).astype(np.float64)
    d = PCA(1, random_state=0).fit(Xc).components_[0]
    if d.sum() < 0:
        d = -d
    sc = Xc @ d
    out, i = {}, 0
    for c, n in sizes:
        s = sc[i:i + n]
        out[c] = (s - s.mean()) / (s.std(ddof=1) or 1.0)
        i += n
    return out


def rv(theta, se, n, k_x, q=1.0):
    """Cinelli-Hazlett robustness value (partial-R^2 scale)."""
    if se == 0 or np.isnan(se):
        return float("nan")
    t = abs(theta) / se
    df = max(n - k_x - 1, 1)
    f2 = (q * t) ** 2 / df
    return 0.5 * (math.sqrt(f2 ** 2 + 4 * f2) - f2)


def assessor_property(city, lat, lon):
    """Return (bedrooms, sqft, year) from the exogenous assessor, nearest-parcel."""
    p = ASSR / f"{city}_assessor.parquet"
    if p.exists():
        a = pd.read_parquet(p)
        cols = a[["lat", "lon"]].to_numpy(float)
        good = np.isfinite(cols).all(1)
        tree = cKDTree(cols[good])
        q = np.column_stack([np.nan_to_num(lat, nan=np.nanmedian(lat)),
                             np.nan_to_num(lon, nan=np.nanmedian(lon))])
        _, idx = tree.query(q)
        src = a[good].reset_index(drop=True)
        return (pd.to_numeric(src.bedrooms, errors="coerce").to_numpy()[idx],
                pd.to_numeric(src.bldg_area_sqft, errors="coerce").to_numpy()[idx],
                pd.to_numeric(src.year_built, errors="coerce").to_numpy()[idx])
    # fall back to the parcel gpkg used by the pipeline
    epsg, bcol, scol = PARCEL_COLS[city]
    g = gpd.read_file(PROC / f"{city}_parcels_micro_geo.gpkg", layer=city).to_crs(epsg)
    cc = g.geometry.centroid
    px, py = cc.x.values, cc.y.values
    ok = np.isfinite(px) & np.isfinite(py)
    tree = cKDTree(np.column_stack([px[ok], py[ok]]))
    latf = np.nan_to_num(lat, nan=np.nanmedian(lat)); lonf = np.nan_to_num(lon, nan=np.nanmedian(lon))
    lp = gpd.GeoSeries(gpd.points_from_xy(lonf, latf), crs=4326).to_crs(epsg)
    qx = np.nan_to_num(lp.x.values, nan=np.nanmedian(lp.x.values)); qy = np.nan_to_num(lp.y.values, nan=np.nanmedian(lp.y.values))
    _, idx = tree.query(np.column_stack([qx, qy]))
    src = np.where(ok)[0][idx]
    def col(name):
        return pd.to_numeric(g[name], errors="coerce").to_numpy()[src] if name in g.columns else np.full(len(lat), np.nan)
    return col(bcol), col(scol), col("year_built")


def census_demo(city, lat, lon):
    bg = gpd.read_file(CENSUS / f"{city}_census_bg.gpkg", layer="bg")
    latf = np.nan_to_num(lat, nan=np.nanmedian(lat)); lonf = np.nan_to_num(lon, nan=np.nanmedian(lon))
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lonf, latf), crs=4326)
    j = gpd.sjoin(pts, bg[DEMO + ["geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")].sort_index()
    return imp(j[DEMO].to_numpy(float))


def run_city(city, T):
    emb, parc = load_analysis_data(city)
    emb = sanitize(emb)
    _t, _cf, Y, meta = get_features_and_target(emb, parc, drop_mismatched_crime=True)
    v = np.asarray(meta["valid_mask"], bool)
    lst = emb.iloc[np.where(v)[0]]
    Tv = T[city][v]
    L = pd.read_parquet(FULL / f"{city}_reembed.parquet").sort_values("row")["log_len"].to_numpy()[v]
    lat = pd.to_numeric(lst.latitude, errors="coerce").to_numpy()
    lon = pd.to_numeric(lst.longitude, errors="coerce").to_numpy()
    pt = lst.property_type.astype(str).str.lower()
    single = (pt.str.contains("single") | pt.str.contains("town")).to_numpy()
    Yv = Y

    lb = pd.to_numeric(lst.beds, errors="coerce").to_numpy()
    ls = pd.to_numeric(lst.sqft, errors="coerce").to_numpy()
    ly = pd.to_numeric(lst.year_built, errors="coerce").to_numpy()
    ab, asq, ay = assessor_property(city, lat, lon)
    demo = census_demo(city, lat, lon)
    latlon = np.column_stack([lat, lon])
    list_prop = winsor(imp(np.column_stack([lb, ls, ly])))
    ass_prop = winsor(imp(np.column_stack([ab, asq, ay])))
    demo = winsor(demo)

    fin = np.isfinite(Yv) & np.isfinite(lat) & np.isfinite(lon)
    m = fin & single
    if m.sum() < 150:
        return None

    def est(prop):
        X = np.nan_to_num(np.column_stack([latlon[m], prop[m], demo[m], L[m], L[m] ** 2]), nan=0.0)
        r = run_dml(Tv[m].reshape(-1, 1), StandardScaler().fit_transform(X), Yv[m],
                    label="x", ci_method="if", n_boot=None, use_ridge=True, seed=42, n_pca=1)
        return abs(float(r.theta)), float(r.se), bool(r.ci_low < 0 < r.ci_high), X.shape[1]

    ta, sa, za, ka = est(ass_prop)
    tl, sl, zl, kl = est(list_prop)
    return {"city": city, "n_single": int(m.sum()),
            "assessor": {"theta": ta, "se": sa, "covers0": za,
                         "rv": rv(ta, sa, m.sum(), ka)},
            "listing": {"theta": tl, "se": sl, "covers0": zl}}


def main():
    T = pooled_treatment()
    have = [c for c in ALL_12 if (ASSR / f"{c}_assessor.parquet").exists() or c in PARCEL_COLS]
    print(f"markets with assessor data: {have}\n")
    print(f"{'city':13s}{'n_SF':>7s}{'ASSESSOR theta':>16s}{'RV':>8s}{'LISTING theta':>15s}")
    print("-" * 60)
    rows = []
    for c in have:
        try:
            r = run_city(c, T)
        except Exception as e:  # noqa: BLE001
            print(f"{c:13s}  ERROR {type(e).__name__}: {str(e)[:40]}")
            continue
        if not r:
            continue
        rows.append(r)
        a, l = r["assessor"], r["listing"]
        print(f"{r['city']:13s}{r['n_single']:7d}"
              f"   {a['theta']:.3f}{'*' if a['covers0'] else ' '} (se {a['se']:.3f})"
              f"{a['rv']:8.3f}   {l['theta']:.3f}{'*' if l['covers0'] else ' '}")
    print("\n* = 95% CI covers zero. ASSESSOR = exogenous control (defensible causal arm);")
    print("LISTING = agent-entered control (M-bias upper bound). RV = Cinelli-Hazlett robustness value.")
    import json
    (REPO / "results" / "fork_clean.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
