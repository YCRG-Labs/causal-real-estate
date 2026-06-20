import sys, os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from spatial_basis import thin_plate_basis

PROC = Path(__file__).resolve().parents[1] / "processed"
OUT = Path(__file__).resolve().parents[2] / "results" / "interpretability"
ALL_12 = ["boston", "nyc", "sf", "dc", "philadelphia", "chicago",
          "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]

GENERIC_LOC = {
    "north", "south", "east", "west", "northeast", "northwest", "southeast",
    "southwest", "subway", "train", "station", "metro", "rail", "line", "stop",
    "transit", "park", "downtown", "uptown", "midtown", "district", "neighborhood",
    "blocks", "block", "steps", "waterfront", "riverfront", "avenue", "ave",
    "street", "road", "boulevard", "blvd", "lane", "drive", "located", "location",
}


def geo_r2(proj: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> float:
    B = thin_plate_basis(lat, lon, k=30)
    B = np.column_stack([np.ones(len(B)), B])
    beta, *_ = np.linalg.lstsq(B, proj, rcond=None)
    resid = proj - B @ beta
    return float(1.0 - resid.var() / proj.var())


def zip_concentrated(term: str, texts: pd.Series, zips: pd.Series, thresh=0.30) -> bool:
    hit = texts.str.contains(rf"\b{term.split()[0]}\b", regex=True, na=False)
    z = zips[hit].dropna()
    if len(z) < 10:
        return False
    return (z.value_counts(normalize=True).iloc[0]) >= thresh


def run_city(city: str, topk: int = 80) -> dict:
    d = pd.read_parquet(PROC / f"{city}_embeddings.parquet")
    emb = [c for c in d.columns if c.startswith("emb_")]
    txt = "clean_description" if "clean_description" in d.columns else "description"
    d = d[d[txt].astype(str).str.len() > 20].reset_index(drop=True)

    X = d[emb].to_numpy(float); X = X - X.mean(0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    proj = X @ vt[0]

    lat = pd.to_numeric(d["latitude"], errors="coerce").to_numpy()
    lon = pd.to_numeric(d["longitude"], errors="coerce").to_numpy()
    ok = np.isfinite(lat) & np.isfinite(lon)
    g = geo_r2(proj[ok], lat[ok], lon[ok])

    V = TfidfVectorizer(min_df=30, ngram_range=(1, 2), stop_words="english")
    T = V.fit_transform(d[txt].astype(str))
    coef = np.abs(Ridge(alpha=1.0).fit(T, proj).coef_)
    vocab = np.array(V.get_feature_names_out())
    top = np.argsort(coef)[::-1][:topk]
    zips = d["zip"].astype(str) if "zip" in d.columns else pd.Series([""] * len(d))
    texts = d[txt].astype(str)

    loc_mass, tot_mass = 0.0, 0.0
    for i in top:
        term, w = vocab[i], coef[i]
        tot_mass += w
        is_loc = (term in GENERIC_LOC or any(t in GENERIC_LOC for t in term.split())
                  or zip_concentrated(term, texts, zips))
        if is_loc:
            loc_mass += w
    return {"city": city, "n": int(len(d)), "geo_r2": round(g, 3),
            "loc_share": round(loc_mass / tot_mass, 3)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [run_city(c) for c in ALL_12]
    df = pd.DataFrame(rows).sort_values("geo_r2", ascending=False)
    df.to_csv(OUT / "geographic_continuum.csv", index=False)
    rho, p = spearmanr(df["geo_r2"], df["loc_share"])
    print(df.to_string(index=False))
    print(f"\nSpearman(geo_r2, loc_share) = {rho:.3f}  (p={p:.4f})")
    print(f"wrote {OUT / 'geographic_continuum.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
