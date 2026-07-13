"""Spatially-honest inference for the pooled-PC text-effect DML, all 12 markets.

For each market: treatment = fixed pooled leading PC of the sentence-embedding
(the Baur construction), residualize log-price and treatment on structural +
geographic confounders via cross-fit gradient boosting, then report the effect
under three variance estimators:
  - iid influence-function SE (what the paper currently uses),
  - zip-clustered SE (neighborhood-scale spatial dependence),
  - Conley spatial-HAC SE (continuous distance-decay kernel on lat/lon).

Shows how much honest spatial inference widens the intervals and which effects
survive. This is a correctness fix (listing language clusters in space, so iid
SEs are too narrow), not a novelty claim.
"""
from __future__ import annotations
import glob, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree
sys.path.insert(0, "data/scripts")
from booster import make_regressor

CONF = ["beds", "baths", "sqft", "year_built", "lot_size", "latitude", "longitude"]
RESULTS = "results/spatial_robust"


def winsor(X):
    X = X.copy()
    for j in range(X.shape[1]):
        c = X[:, j]; m = np.nanmedian(c); md = np.nanmedian(np.abs(c - m)) + 1e-9
        c = np.clip(c, m - 5 * md, m + 5 * md); c[~np.isfinite(c)] = m; X[:, j] = c
    return X


def dml_psi(Y, T, X, seed=42):
    n = len(Y); Xs = StandardScaler().fit_transform(X)
    kf = KFold(5, shuffle=True, random_state=seed)
    Yr = np.zeros(n); Tr = np.zeros(n)
    for tr, te in kf.split(np.arange(n)):
        my = make_regressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=1)
        my.fit(Xs[tr], Y[tr]); Yr[te] = Y[te] - my.predict(Xs[te])
        mt = make_regressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=1)
        mt.fit(Xs[tr], T[tr]); Tr[te] = T[te] - mt.predict(Xs[te])
    den = float(np.mean(Tr ** 2)); th = float(np.mean(Tr * Yr)) / den
    psi = (Yr - th * Tr) * Tr / den
    return th, psi


def se_iid(psi):
    return float(np.sqrt(np.var(psi, ddof=1) / len(psi)))


def se_cluster(psi, g):
    s = pd.Series(psi).groupby(g).sum().to_numpy()
    return float(np.sqrt(np.sum(s ** 2)) / len(psi))


def se_conley(psi, lat, lon, bw_km=2.0):
    # meters-ish: 1 deg lat ~111km; scale lon by cos(lat)
    xy = np.column_stack([lat * 111.0, lon * 111.0 * np.cos(np.radians(np.nanmedian(lat)))])
    ok = np.isfinite(xy).all(1)
    psi = psi[ok]; xy = xy[ok]; n = len(psi)
    tree = cKDTree(xy)
    pairs = tree.query_ball_tree(tree, r=bw_km)
    V = 0.0
    for i, nb in enumerate(pairs):
        for j in nb:
            d = np.hypot(*(xy[i] - xy[j]))
            w = 1.0 - d / bw_km  # Bartlett kernel
            V += w * psi[i] * psi[j]
    return float(np.sqrt(V) / n), n


def main():
    os.makedirs(RESULTS, exist_ok=True)
    f = sorted([x for x in glob.glob("data/processed/*_embeddings.parquet") if "all_MiniLM" not in x])
    emb = [c for c in pd.read_parquet(f[0]).columns if c.startswith("emb_")]
    frames = {}
    for x in f:
        df = pd.read_parquet(x)
        sub = df[["price", "zip"] + CONF + emb].copy()
        sub = sub[(sub["price"] > 1e4) & (sub["price"] < 1e8)].reset_index(drop=True)
        frames[os.path.basename(x).split("_")[0]] = sub
    # pooled PC1, held fixed
    Eall = np.vstack([fr[emb].to_numpy(np.float64) for fr in frames.values()])
    mu = Eall.mean(0); _, _, Vt = np.linalg.svd(Eall - mu, full_matrices=False); v1 = Vt[0]

    rows = []
    for mkt, fr in frames.items():
        E = fr[emb].to_numpy(np.float64); Y = np.log(fr["price"].to_numpy(float))
        X = winsor(fr[CONF].to_numpy(float))
        t = (E - mu) @ v1; t = (t - t.mean()) / (t.std() or 1.0)
        th, psi = dml_psi(Y, t, X)
        s_iid = se_iid(psi); s_zip = se_cluster(psi, fr["zip"].astype(str).to_numpy())
        s_con, _ = se_conley(psi, fr["latitude"].to_numpy(float), fr["longitude"].to_numpy(float))
        rows.append({"market": mkt, "n": len(Y), "theta": th,
                     "se_iid": s_iid, "se_zip": s_zip, "se_conley": s_con,
                     "infl_zip": s_zip / s_iid, "infl_conley": s_con / s_iid,
                     "t_iid": th / s_iid, "t_zip": th / s_zip, "t_conley": th / s_con,
                     "sig_iid": abs(th / s_iid) > 1.96, "sig_zip": abs(th / s_zip) > 1.96,
                     "sig_conley": abs(th / s_con) > 1.96})
        r = rows[-1]
        print(f"{mkt:12} n={len(Y):5} th={th:+.4f} | t_iid={r['t_iid']:+.1f} "
              f"t_zip={r['t_zip']:+.1f} t_conley={r['t_conley']:+.1f} | "
              f"inflZ={r['infl_zip']:.2f} inflC={r['infl_conley']:.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS}/spatial_robust_table.csv", index=False)
    print(f"\nsig under iid: {df.sig_iid.sum()}/12  zip: {df.sig_zip.sum()}/12  "
          f"conley: {df.sig_conley.sum()}/12", flush=True)
    print(f"median SE inflation  zip: {df.infl_zip.median():.2f}  conley: {df.infl_conley.median():.2f}", flush=True)
    print(f"wrote {RESULTS}/spatial_robust_table.csv", flush=True)


if __name__ == "__main__":
    main()
