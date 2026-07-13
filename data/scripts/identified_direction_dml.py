"""Realize Theorem 1' 's prescription: estimate the text price-effect on the
IDENTIFIED direction rather than the leading-variance principal component.

Theorem 1' shows the effect depends on the direction, and is confounded to the
extent the direction loads on the location-predictable (in particular the discrete
zip/neighborhood) channel. The identified direction is the leading principal
component of the embedding AFTER that channel is removed. We build it by
residualizing each embedding coordinate, within market, on a rich location basis
(lat/lon polynomial + zip indicators, the discrete geographic channel Theorem 1
requires the control basis to span), pooling the residuals, and taking their
leading component. We then re-estimate the per-market DML effect on this identified
direction and compare it to the raw leading-PC (Baur) effect.

Same confounder set for both arms so the contrast is apples-to-apples. Absolute
levels use a reduced 7-confounder set and are not the paper's full-confounder Baur
numbers; the transferable quantity is the identified-vs-leading contrast and the
alignment cos(v_id, v_lead).
"""
from __future__ import annotations
import glob, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd
from numpy.linalg import lstsq, svd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, "data/scripts")
from booster import make_regressor

CONF = ["beds", "baths", "sqft", "year_built", "lot_size", "latitude", "longitude"]
RES = "results/identified_direction"


def winsor(X):
    X = X.copy()
    for j in range(X.shape[1]):
        c = X[:, j]; m = np.nanmedian(c); md = np.nanmedian(np.abs(c - m)) + 1e-9
        c = np.clip(c, m - 5 * md, m + 5 * md); c[~np.isfinite(c)] = m; X[:, j] = c
    return X


def loc_basis(df):
    lat = df["latitude"].to_numpy(float); lon = df["longitude"].to_numpy(float)
    lat = np.nan_to_num(lat, nan=np.nanmedian(lat)); lon = np.nan_to_num(lon, nan=np.nanmedian(lon))
    poly = np.column_stack([np.ones(len(df)), lat, lon, lat**2, lon**2, lat*lon])
    zdum = pd.get_dummies(df["zip"].astype(str)).to_numpy(float)  # discrete geographic channel
    return np.column_stack([poly, zdum])


def residualize(M, B):
    beta, *_ = lstsq(B, M, rcond=None)
    return M - B @ beta


def dml(Y, T, X, seed=42):
    n = len(Y); Xs = StandardScaler().fit_transform(X); kf = KFold(5, shuffle=True, random_state=seed)
    Yr = np.zeros(n); Tr = np.zeros(n)
    for tr, te in kf.split(np.arange(n)):
        my = make_regressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=1)
        my.fit(Xs[tr], Y[tr]); Yr[te] = Y[te] - my.predict(Xs[te])
        mt = make_regressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=1)
        mt.fit(Xs[tr], T[tr]); Tr[te] = T[te] - mt.predict(Xs[te])
    den = float(np.mean(Tr**2)); th = float(np.mean(Tr*Yr))/den
    se = float(np.sqrt(np.var((Yr-th*Tr)*Tr/den, ddof=1)/n))
    return th, se


def main():
    os.makedirs(RES, exist_ok=True)
    f = sorted([x for x in glob.glob("data/processed/*_embeddings.parquet") if "all_MiniLM" not in x])
    emb = [c for c in pd.read_parquet(f[0]).columns if c.startswith("emb_")]
    frames = {}
    for x in f:
        df = pd.read_parquet(x)
        sub = df[["price", "zip"] + CONF + emb].copy()
        sub = sub[(sub["price"] > 1e4) & (sub["price"] < 1e8)].reset_index(drop=True)
        frames[os.path.basename(x).split("_")[0]] = sub

    # leading-variance pooled PC (raw) = Baur direction
    Eall = np.vstack([fr[emb].to_numpy(np.float64) for fr in frames.values()])
    mu = Eall.mean(0); _, _, Vt = svd(Eall - mu, full_matrices=False); v_lead = Vt[0]

    # identified direction: pooled leading PC of the WITHIN-MARKET location-residualized embedding
    resid_blocks = []
    for mkt, fr in frames.items():
        M = fr[emb].to_numpy(np.float64)
        B = loc_basis(fr)
        resid_blocks.append(residualize(M, B))
        print(f"  residualized {mkt} ({len(fr)} rows, {B.shape[1]} loc features)", flush=True)
    R = np.vstack(resid_blocks); muR = R.mean(0)
    _, _, VtR = svd(R - muR, full_matrices=False); v_id = VtR[0]
    if float(v_id @ v_lead) < 0: v_id = -v_id
    cos_il = float(abs(v_id @ v_lead))
    print(f"\n|cos(v_identified, v_leading)| = {cos_il:.4f}\n", flush=True)

    rows = []
    off = 0
    for mkt, fr in frames.items():
        n = len(fr); Y = np.log(fr["price"].to_numpy(float)); X = winsor(fr[CONF].to_numpy(float))
        E = fr[emb].to_numpy(np.float64)
        Rblk = resid_blocks[list(frames).index(mkt)]
        t_lead = E @ v_lead; t_lead = (t_lead - t_lead.mean()) / (t_lead.std() or 1)
        t_id = Rblk @ v_id;   t_id = (t_id - t_id.mean()) / (t_id.std() or 1)
        th_l, se_l = dml(Y, t_lead, X)
        th_i, se_i = dml(Y, t_id, X)
        rows.append({"market": mkt, "n": n, "cos_il": cos_il,
                     "theta_leading": th_l, "se_leading": se_l, "t_leading": th_l/se_l,
                     "theta_identified": th_i, "se_identified": se_i, "t_identified": th_i/se_i,
                     "sig_lead": abs(th_l/se_l) > 1.96, "sig_id": abs(th_i/se_i) > 1.96})
        r = rows[-1]
        print(f"{mkt:12} n={n:5}  leading th={th_l:+.4f}(t={th_l/se_l:+.1f})  "
              f"identified th={th_i:+.4f}(t={th_i/se_i:+.1f})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/identified_vs_leading.csv", index=False)
    # pooled (inverse-variance) summary
    def pooled(col, secol):
        w = 1/df[secol]**2; return float((df[col]*w).sum()/w.sum()), float(np.sqrt(1/w.sum()))
    pl, sl = pooled("theta_leading", "se_leading"); pi, si = pooled("theta_identified", "se_identified")
    print(f"\nPOOLED (inverse-variance):")
    print(f"  leading    theta={pl:+.4f} se={sl:.4f} (t={pl/sl:+.1f})  sig markets {int(df.sig_lead.sum())}/12")
    print(f"  identified theta={pi:+.4f} se={si:.4f} (t={pi/si:+.1f})  sig markets {int(df.sig_id.sum())}/12")
    print(f"  ratio identified/leading = {pi/pl:.2f}")
    print(f"\nwrote {RES}/identified_vs_leading.csv")


if __name__ == "__main__":
    main()
