"""Why dropping vacant land moves the text effect.

Three questions, on the markets that still have a full confounder block.

  1. Can the confounder block see landness at all?  Cross-fitted AUC of a logit of
     the land indicator on the confounders, against the same logit on the text
     direction.
  2. Is the drop just tail leverage?  Compare dropping the k land listings against
     dropping k random non-land listings, and against dropping the k non-land
     listings furthest out in the tail of T that land occupies.
  3. How much of the drop is run_dml restandardising T on the subsample?

    python data/scripts/land_mechanism_check.py boston sf nyc
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "data" / "scripts"))

from causal_inference import load_analysis_data, get_features_and_target
from replications.compare_to_dml import run_dml

POOLED_CSV = REPO / "results" / "replications" / "pooled_pca_treatment.csv"
OUT_JSON = REPO / "results" / "land_mechanism_check.json"
NON_RESIDENTIAL = r"land|lot|vacant|parking"


def _setup(city: str):
    emb_df, parcels = load_analysis_data(city)
    if "zip" in emb_df.columns:
        z = emb_df["zip"].astype("string").str.strip().str.slice(0, 5)
        emb_df = emb_df.copy()
        emb_df["zip"] = z.replace("", pd.NA).astype("Float64")

    land_all = (emb_df["property_type"].astype(str)
                .str.contains(NON_RESIDENTIAL, case=False, na=False).to_numpy())

    pool = pd.read_csv(POOLED_CSV)
    pc = pool[pool.city == city]
    if "listing_id" not in emb_df.columns:
        emb_df = emb_df.assign(listing_id=np.arange(len(emb_df)))
    T_all = (emb_df[["listing_id"]]
             .merge(pc[["listing_id", "treatment_z"]], on="listing_id", how="left")
             .treatment_z.fillna(0.0).to_numpy(float))

    _t, conf, Y, meta = get_features_and_target(emb_df, parcels, drop_mismatched_crime=True)
    valid = np.asarray(meta["valid_mask"], bool)
    return T_all[valid], conf, Y, land_all[valid], meta


def _theta(T, conf, Y, mask):
    r = run_dml(T[mask].reshape(-1, 1), conf[mask], Y[mask], label="x",
                ci_method="if", n_boot=None, use_ridge=True, seed=42, n_pca=1)
    return abs(float(r.theta)) if r else float("nan")


def _auc(Z, y):
    p = cross_val_predict(LogisticRegression(max_iter=2000), Z, y, cv=5,
                          method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def check(city: str, n_rand: int = 200, seed: int = 0) -> dict:
    T, conf, Y, land, meta = _setup(city)
    if not meta["has_rich_confounders"]:
        return {"city": city, "skipped": "confounders are lat/lon only"}

    k, n = int(land.sum()), len(T)
    keep = np.ones(n, bool)

    auc_conf = _auc(StandardScaler().fit_transform(conf), land.astype(int))
    auc_text = _auc(T.reshape(-1, 1), land.astype(int))

    th_full = _theta(T, conf, Y, keep)
    th_land = _theta(T, conf, Y, ~land)

    side = np.sign(T[land].mean() - T[~land].mean())
    nonland = np.where(~land)[0]
    tail = nonland[np.argsort(side * T[nonland])[-k:]]
    m = keep.copy()
    m[tail] = False
    th_tail = _theta(T, conf, Y, m)

    rng = np.random.default_rng(seed)
    rand = []
    for _ in range(n_rand):
        m = keep.copy()
        m[rng.choice(nonland, size=k, replace=False)] = False
        rand.append(_theta(T, conf, Y, m))
    rand = np.asarray(rand)

    rescale = T[~land].std(ddof=1) / T.std(ddof=1)
    return {
        "city": city, "n": n, "n_land": k, "pct_land": 100 * k / n,
        "auc_land_given_confounders": auc_conf,
        "auc_land_given_text_pc1": auc_text,
        "theta_full": th_full,
        "theta_drop_land": th_land,
        "theta_drop_tail_nonland": th_tail,
        "theta_drop_random_nonland": {"mean": float(rand.mean()), "sd": float(rand.std()),
                                      "p2.5": float(np.percentile(rand, 2.5)),
                                      "p97.5": float(np.percentile(rand, 97.5))},
        "drop_land_percentile_of_random_null": float(100 * (rand <= th_land).mean()),
        "pct_attenuation": 100 * (1 - th_land / th_full),
        "pct_attenuation_net_of_rescaling": 100 * (1 - (th_land / rescale) / th_full),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cities", nargs="+")
    ap.add_argument("--n_rand", type=int, default=200)
    args = ap.parse_args()

    out = [check(c, n_rand=args.n_rand) for c in args.cities]
    OUT_JSON.write_text(json.dumps(out, indent=2))

    for r in out:
        if "skipped" in r:
            print(f"\n{r['city']}: skipped, {r['skipped']}")
            continue
        print(f"\n=== {r['city']}  n={r['n']}, land={r['n_land']} ({r['pct_land']:.1f}%) ===")
        print(f"  AUC(land ~ confounders) = {r['auc_land_given_confounders']:.3f}")
        print(f"  AUC(land ~ text PC1)    = {r['auc_land_given_text_pc1']:.3f}")
        print(f"  |theta| full            = {r['theta_full']:.4f}")
        print(f"  |theta| drop land       = {r['theta_drop_land']:.4f}")
        print(f"  |theta| drop tail       = {r['theta_drop_tail_nonland']:.4f}")
        rd = r["theta_drop_random_nonland"]
        print(f"  |theta| drop random     = {rd['mean']:.4f} "
              f"[{rd['p2.5']:.4f}, {rd['p97.5']:.4f}]")
        print(f"  land drop at {r['drop_land_percentile_of_random_null']:.1f}th pctile of random null")
        print(f"  attenuation {r['pct_attenuation']:.1f}%, "
              f"net of rescaling {r['pct_attenuation_net_of_rescaling']:.1f}%")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
