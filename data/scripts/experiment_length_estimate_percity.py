"""Clean per-city version: no pooling-composition confound.

For each of the 4 re-embedded cities, fit PCA on THAT city's html-fixed embedding
(within-city centered), giving PC1_raw. Then residualize each embedding dim on
[1, log_len, log_len^2] and re-fit PCA, giving PC1_resid. Estimate the same ridge
DML for each, on that city's own confounder block.

Also, the FWL check that needs no re-PCA at all: take the PUBLISHED treatment and
just add log_len+log_len^2 to the controls (this is the defensible headline, since
it keeps the paper's exact treatment and only adds a control).
"""
import sys, json, gc
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/jacobcrainic/causal-real-estate/data/scripts")
from sklearn.decomposition import PCA
from replications.compare_to_dml import run_dml
from causal_inference import load_analysis_data, get_features_and_target

REPO = "/Users/jacobcrainic/causal-real-estate"
CITIES = ["boston", "nyc", "sf", "dc"]
EMB = [f"emb_{i}" for i in range(768)]


def sanitize(emb):
    if "zip" in emb.columns:
        z = emb["zip"].astype("string").str.strip().str.slice(0, 5)
        emb = emb.copy(); emb["zip"] = z.replace("", pd.NA).astype("Float64")
    return emb


def pc1(X):
    d = PCA(1, random_state=0).fit(X).components_[0]
    if d.sum() < 0: d = -d
    s = X @ d
    return (s - s.mean()) / (s.std(ddof=1) or 1.0)


def th(T, conf, Y):
    r = run_dml(np.asarray(T).reshape(-1, 1), conf, Y, label="x", ci_method="if",
                n_boot=None, use_ridge=True, seed=42, n_pca=1)
    return abs(float(r.theta)), float(r.se), bool(r.ci_low < 0 < r.ci_high)


pub = pd.read_csv(f"{REPO}/results/replications/pooled_pca_treatment.csv")
rows = []
for c in CITIES:
    re = pd.read_parquet(f"{REPO}/results/experiment_length/{c}_reembed.parquet")
    X = re[EMB].to_numpy(np.float64)
    X = X - X.mean(0, keepdims=True)
    logtok = re["log_len"].to_numpy(float)

    pc_raw = pc1(X)
    B = np.column_stack([np.ones_like(logtok), logtok, logtok**2])
    Xr = X - B @ np.linalg.lstsq(B, X, rcond=None)[0]
    pc_res = pc1(Xr)

    # published treatment, aligned by position (trim to min length)
    pc_pub = pub[pub.city == c].reset_index(drop=True)["treatment_z"].to_numpy(float)

    corr_raw = abs(np.corrcoef(pc_raw, logtok)[0, 1])
    corr_res = abs(np.corrcoef(pc_res, logtok)[0, 1])
    del X, Xr; gc.collect()

    emb, parc = load_analysis_data(c)
    emb = sanitize(emb)
    _t, conf, Y, meta = get_features_and_target(emb, parc, drop_mismatched_crime=True)
    valid = np.asarray(meta["valid_mask"], bool)
    rich = meta["has_rich_confounders"]

    n = len(re)
    # FWL: published treatment + length controls, on the published treatment's own rows
    m = min(len(pc_pub), n)
    pub_full = np.zeros(n); pub_full[:m] = pc_pub[:m]
    logv = logtok[valid]
    conf_len = np.column_stack([conf, logv, logv**2])

    res = {"city": c, "n": int(valid.sum()), "rich": rich,
           "corr_pc1_len_raw": float(corr_raw), "corr_pc1_len_resid": float(corr_res)}
    for label, T, cf in [
        ("percity_raw", pc_raw[valid], conf),
        ("percity_resid", pc_res[valid], conf),
        ("published", pub_full[valid], conf),
        ("published_plus_len_ctrl", pub_full[valid], conf_len),
    ]:
        t, se, z = th(T, cf, Y)
        res[label] = {"abs_theta": t, "se": se, "covers0": z}
    rows.append(res)
    del re, emb, conf, Y; gc.collect()

json.dump(rows, open(f"{REPO}/results/experiment_length/results_percity.json", "w"), indent=2)

print(f"\n{'city':7s}{'rich':>6s}{'n':>7s}{'r(PC1,len)raw':>14s}{'r resid':>9s}"
      f"{'raw|θ|':>9s}{'resid|θ|':>10s}{'pub|θ|':>9s}{'pub+len':>9s}")
print("-"*80)
for r in rows:
    f = lambda k: f"{r[k]['abs_theta']:.3f}" + ("*" if r[k]['covers0'] else " ")
    print(f"{r['city']:7s}{str(r['rich']):>6s}{r['n']:7d}{r['corr_pc1_len_raw']:14.3f}"
          f"{r['corr_pc1_len_resid']:9.3f}{f('percity_raw'):>9s}{f('percity_resid'):>10s}"
          f"{f('published'):>9s}{f('published_plus_len_ctrl'):>9s}")
print("\n* = CI covers zero. percity = PCA on this city's html-fixed embedding.")
print("published+len = paper's exact treatment, adding log_len+log_len^2 as controls (the FWL check).")
