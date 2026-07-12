"""Full 12-city length experiment on the HTML-fixed pooled treatment.

Uses the Brev re-embeddings (results/experiment_length_full/<city>_reembed.parquet),
rebuilds the paper's pooled-within-city-centered PC1 exactly, plus a
length-residualized variant, and runs the ridge DML per market against whatever
confounder block load_analysis_data returns — which is now the real parcel block
for boston/nyc/sf AND philadelphia (rebuilt), lat/lon-only elsewhere.

    python data/scripts/experiment_pooled_estimate.py
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "data" / "scripts"))
from replications.compare_to_dml import run_dml
from causal_inference import load_analysis_data, get_features_and_target

FULL = REPO / "results" / "experiment_length_full"
OUT = REPO / "results" / "experiment_length" / "pooled_results.json"
ALL_12 = ["boston", "nyc", "sf", "dc", "philadelphia", "chicago",
          "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]
EMB = [f"emb_{i}" for i in range(768)]


def build_treatments():
    blocks, lens, sizes = [], [], []
    for c in ALL_12:
        r = pd.read_parquet(FULL / f"{c}_reembed.parquet").sort_values("row").reset_index(drop=True)
        blocks.append(r[EMB].to_numpy(np.float32))
        lens.append(r["log_len"].to_numpy(float))
        sizes.append((c, len(r)))
        del r
    Xc = np.vstack([b - b.mean(0, keepdims=True) for b in blocks]).astype(np.float64)
    del blocks
    gc.collect()
    logtok = np.concatenate(lens)

    def pc1(M):
        d = PCA(1, random_state=0).fit(M).components_[0]
        if d.sum() < 0:
            d = -d
        return M @ d

    def zsplit(scores):
        out, i = {}, 0
        for c, n in sizes:
            s = scores[i:i + n]
            out[c] = (s - s.mean()) / (s.std(ddof=1) or 1.0)
            i += n
        return out

    T_html = zsplit(pc1(Xc))
    B = np.column_stack([np.ones_like(logtok), logtok, logtok ** 2])
    Xr = Xc - B @ np.linalg.lstsq(B, Xc, rcond=None)[0]
    T_resid = zsplit(pc1(Xr))
    del Xc, Xr
    gc.collect()
    logmap, i = {}, 0
    for c, n in sizes:
        logmap[c] = logtok[i:i + n]
        i += n
    return T_html, T_resid, logmap, dict(sizes)


def sanitize(emb):
    if "zip" in emb.columns:
        z = emb["zip"].astype("string").str.strip().str.slice(0, 5)
        emb = emb.copy()
        emb["zip"] = z.replace("", pd.NA).astype("Float64")
    return emb


def run():
    T_html, T_resid, logmap, sizes = build_treatments()
    rows = []
    for c in ALL_12:
        emb, parc = load_analysis_data(c)
        emb = sanitize(emb)
        _t, conf, Y, meta = get_features_and_target(emb, parc, drop_mismatched_crime=True)
        valid = np.asarray(meta["valid_mask"], bool)
        rich = meta["has_rich_confounders"]
        nconf = conf.shape[1]
        lv = logmap[c][valid]
        conf_len = np.column_stack([conf, lv, lv ** 2])

        for name, tmap, cf in [("html_fixed", T_html, conf),
                               ("html_fixed_plus_len", T_html, conf_len),
                               ("length_residualized", T_resid, conf)]:
            T = tmap[c]
            if len(T) != len(valid):
                print(f"  {c}/{name}: len {len(T)} != {len(valid)}, skip")
                continue
            r = run_dml(T[valid].reshape(-1, 1), StandardScaler().fit_transform(cf), Y,
                        label=f"{name}:{c}", ci_method="if", n_boot=None,
                        use_ridge=True, seed=42, n_pca=1)
            rows.append({"city": c, "spec": name, "n": int(valid.sum()),
                         "n_confounders": int(cf.shape[1]), "rich": bool(rich),
                         "abs_theta": abs(float(r.theta)), "se": float(r.se),
                         "covers_zero": bool(r.ci_low < 0 < r.ci_high)})
        del emb, conf, Y
        gc.collect()

    OUT.write_text(json.dumps(rows, indent=2))
    print(f"\n{'city':13s}{'rich':>6s}{'nconf':>6s}{'html|θ|':>9s}{'+len|θ|':>9s}{'resid|θ|':>10s}{'lenDrop':>9s}")
    print("-" * 62)
    by = {}
    for r in rows:
        by.setdefault(r["city"], {})[r["spec"]] = r
    for c in ALL_12:
        b = by.get(c, {})
        if "html_fixed" not in b:
            continue
        h = b["html_fixed"]; hl = b.get("html_fixed_plus_len", {}); rd = b.get("length_residualized", {})
        def f(x):
            if not x:
                return "  -  "
            return f"{x['abs_theta']:.3f}" + ("*" if x['covers_zero'] else " ")
        drop = 100 * (1 - hl["abs_theta"] / h["abs_theta"]) if hl and h["abs_theta"] else float("nan")
        print(f"{c:13s}{str(h['rich']):>6s}{h['n_confounders']:6d}{f(h):>9s}{f(hl):>9s}{f(rd):>10s}{drop:8.1f}%")
    print("\n* = 95% CI covers zero.  html_fixed = pooled PC1 on unescaped text;")
    print("+len adds log_len+log_len^2 to controls; resid = length-residualized PC1.")


if __name__ == "__main__":
    run()
