"""Design-based check: within-building fixed-effects DML.

Units in the same building share location exactly (same lat/lon, neighborhood,
school, amenities), so estimating the text price-effect from variation BETWEEN
units of the SAME building differences out spatial confounding by construction
rather than bounding it by assumption. This is design-based identification of the
paper's primary threat. Residual threat is within-building unit quality, addressed
(partly) by structural controls, not by the design.

Building key = lat/lon rounded to 4 decimals (~11 m). Keep buildings with >=2
listings. Within-building demean the treatment, log price, and structural controls,
then DML on the demeaned data (location controls are absorbed by the fixed effect).
Cluster the influence function by building.
"""
from __future__ import annotations
import glob, json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import numpy as np, pandas as pd
from numpy.linalg import svd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, "data/scripts")
from booster import make_regressor

STRUCT = ["beds", "baths", "sqft", "year_built", "lot_size"]
OUT = "results/within_building"


def winsor(X):
    X = X.copy()
    for j in range(X.shape[1]):
        c = X[:, j]; m = np.nanmedian(c); md = np.nanmedian(np.abs(c - m)) + 1e-9
        c = np.clip(c, m - 5*md, m + 5*md); c[~np.isfinite(c)] = m; X[:, j] = c
    return X


def demean(A, g):
    df = pd.DataFrame(A); df["_g"] = g
    return (df.groupby("_g").transform(lambda s: s - s.mean())).to_numpy()


def fe_dml(Y, T, X, g, seed=42):
    """Within-building FE DML with building-clustered SE."""
    Yd = demean(Y.reshape(-1, 1), g).ravel()
    Td = demean(T.reshape(-1, 1), g).ravel()
    Xd = demean(X, g)
    keep = np.abs(Td) > 1e-10  # drop singleton-building rows (no within variation)
    Yd, Td, Xd, gk = Yd[keep], Td[keep], Xd[keep], np.asarray(g)[keep]
    n = len(Yd)
    if n < 100:
        return None
    Xs = StandardScaler().fit_transform(Xd)
    kf = KFold(5, shuffle=True, random_state=seed)
    Yr = np.zeros(n); Tr = np.zeros(n)
    from sklearn.linear_model import RidgeCV
    alphas = np.logspace(-3, 3, 13)
    for tr, te in kf.split(np.arange(n)):
        my = RidgeCV(alphas=alphas); my.fit(Xs[tr], Yd[tr]); Yr[te] = Yd[te] - my.predict(Xs[te])
        mt = RidgeCV(alphas=alphas); mt.fit(Xs[tr], Td[tr]); Tr[te] = Td[te] - mt.predict(Xs[te])
    den = float(np.mean(Tr**2))
    if den < 1e-12:
        return None
    th = float(np.mean(Tr*Yr))/den
    psi = (Yr - th*Tr)*Tr/den
    se_iid = float(np.sqrt(np.var(psi, ddof=1)/n))
    s = pd.Series(psi).groupby(gk).sum().to_numpy()
    se_cl = float(np.sqrt(np.sum(s**2))/n)
    return th, se_iid, se_cl, n, len(np.unique(gk))


def main():
    os.makedirs(OUT, exist_ok=True)
    f = sorted([x for x in glob.glob("data/processed/*_embeddings.parquet") if "all_MiniLM" not in x])
    emb = [c for c in pd.read_parquet(f[0]).columns if c.startswith("emb_")]
    frames = {}
    for x in f:
        df = pd.read_parquet(x)
        df = df[(df["price"] > 1e4) & (df["price"] < 1e8)].copy()
        df = df.dropna(subset=["latitude", "longitude"])
        df["bldg"] = df["latitude"].round(4).astype(str) + "_" + df["longitude"].round(4).astype(str)
        frames[os.path.basename(x).split("_")[0]] = df.reset_index(drop=True)

    # pooled within-city-centered leading PC (Baur channel)
    blocks = [fr[emb].to_numpy(np.float64) - fr[emb].to_numpy(np.float64).mean(0) for fr in frames.values()]
    Ec = np.vstack(blocks); _, _, Vt = svd(Ec - Ec.mean(0), full_matrices=False); v_lead = Vt[0]

    rows = []
    for mkt, fr in frames.items():
        sizes = fr.groupby("bldg").size()
        multi = set(sizes[sizes >= 2].index)
        sub = fr[fr["bldg"].isin(multi)].copy()
        if len(sub) < 200:
            print(f"  {mkt}: too few within-building rows ({len(sub)})", flush=True); continue
        E = sub[emb].to_numpy(np.float64); E = E - E.mean(0)
        T = E @ v_lead; T = (T - T.mean())/(T.std() or 1)
        Y = np.log(sub["price"].to_numpy(float))
        X = winsor(sub[STRUCT].to_numpy(float))
        res = fe_dml(Y, T, X, sub["bldg"].to_numpy())
        if res is None:
            print(f"  {mkt}: FE-DML failed", flush=True); continue
        th, se_i, se_c, n_eff, n_b = res
        rows.append({"market": mkt, "n_within": len(sub), "n_eff": n_eff, "n_bldg": n_b,
                     "theta_fe": th, "se_iid": se_i, "se_bldg_clustered": se_c,
                     "t_clustered": th/se_c, "sig": bool(abs(th/se_c) > 1.96)})
        r = rows[-1]
        print(f"  {mkt:12} n={len(sub):5} bldgs={n_b:4}  theta_FE={th:+.4f}  "
              f"t(clustered)={th/se_c:+.1f}  {'SIG' if r['sig'] else 'ns'}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/within_building_table.csv", index=False)
    th = df.theta_fe.to_numpy(float); se = df.se_bldg_clustered.to_numpy(float)
    w = 1/se**2; pth = float((th*w).sum()/w.sum()); pse = float(np.sqrt(1/w.sum()))
    summ = {"treatment": "pooled within-city-centered leading PC (Baur)",
            "pooled_theta_fe": pth, "pooled_se": pse, "pooled_t": pth/pse,
            "sig_markets": int(df.sig.sum()), "n_markets": len(df)}
    (Path(OUT)/"within_building_summary.json").write_text(json.dumps(summ, indent=2))
    print(f"\nPOOLED within-building FE (inverse-variance, building-clustered SE):")
    print(f"  theta={pth:+.4f} se={pse:.4f} (t={pth/pse:+.1f})  sig {int(df.sig.sum())}/{len(df)} markets")
    print(f"\nwrote {OUT}/within_building_table.csv")


if __name__ == "__main__":
    main()
