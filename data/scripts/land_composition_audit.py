"""Composition audit: vacant-land listings in the twelve-market panel.

property_type never enters the confounder set. PROPERTY is bedrooms,
bldg_area_sqft, lot_area_sqft and year_built, and those are pulled from the
nearest assessor parcel, which for a vacant lot is usually a neighbouring built
parcel. The design therefore cannot represent the land-versus-house distinction:
a logit of the land indicator on the confounder block runs at chance. The listing
text can represent it, and where it does the text direction absorbs the
land-versus-house price gap.

This script reports each market's land share and re-estimates the pooled-PC1 DML
effect with and without those listings.

Only markets with a parcel file (a micro_geo or amenities gpkg under
data/processed) carry the full confounder block. Everywhere else
get_features_and_target falls back to latitude and longitude alone, which is not
the published specification, so attenuation is reported but flagged untrustworthy.
At the time of writing only boston, nyc and sf have parcel files in the tree.

    python data/scripts/land_composition_audit.py --all_12
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "data" / "scripts"))

from causal_inference import load_analysis_data, get_features_and_target
from replications.compare_to_dml import run_dml

POOLED_CSV = REPO / "results" / "replications" / "pooled_pca_treatment.csv"
OUT_JSON = REPO / "results" / "land_composition_audit.json"

ALL_12 = ["boston", "nyc", "sf", "dc", "philadelphia", "chicago",
          "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]

NON_RESIDENTIAL = r"land|lot|vacant|parking"


def _sanitize_zip(emb_df: pd.DataFrame) -> pd.DataFrame:
    if "zip" not in emb_df.columns:
        return emb_df
    z = emb_df["zip"].astype("string").str.strip().str.slice(0, 5)
    emb_df = emb_df.copy()
    emb_df["zip"] = z.replace("", pd.NA).astype("Float64")
    return emb_df


def _pooled_treatment(city: str, emb_df: pd.DataFrame) -> np.ndarray | None:
    pool = pd.read_csv(POOLED_CSV)
    pool_city = pool[pool["city"] == city]
    if len(pool_city) == 0:
        return None
    if "listing_id" not in emb_df.columns:
        emb_df = emb_df.assign(listing_id=np.arange(len(emb_df)))
    merged = emb_df[["listing_id"]].merge(
        pool_city[["listing_id", "treatment_z"]], on="listing_id", how="left")
    merged["treatment_z"] = merged["treatment_z"].fillna(0.0)
    return merged["treatment_z"].to_numpy(float).reshape(-1, 1)


def audit_city(city: str, seed: int = 42, k_folds: int = 5) -> dict | None:
    loaded = load_analysis_data(city)
    if loaded is None:
        return None
    emb_df, parcels = loaded
    emb_df = _sanitize_zip(emb_df)

    T_pooled = _pooled_treatment(city, emb_df)
    if T_pooled is None:
        return None

    is_land = (emb_df["property_type"].astype(str)
               .str.contains(NON_RESIDENTIAL, case=False, na=False)
               .to_numpy())

    feats = get_features_and_target(emb_df, parcels, drop_mismatched_crime=True)
    if feats is None:
        return None
    _T, conf, Y, meta = feats
    valid = np.asarray(meta["valid_mask"], dtype=bool)

    T = T_pooled[valid]
    land_v = is_land[valid]

    out: dict = {"city": city,
                 "n_total": int(valid.sum()),
                 "n_land": int(land_v.sum()),
                 "pct_land": float(100 * land_v.mean()),
                 "n_confounders": int(conf.shape[1]),
                 "has_rich_confounders": bool(meta["has_rich_confounders"])}

    for label, mask in (("full", np.ones_like(land_v)), ("no_land", ~land_v)):
        res = run_dml(T[mask], conf[mask], Y[mask],
                      label=f"{city}:{label}", ci_method="if", n_boot=None,
                      use_ridge=True, seed=seed, n_pca=1)
        if res is None:
            continue
        out[label] = {"n": int(mask.sum()),
                      "theta": float(res.theta), "se": float(res.se),
                      "ci_low": float(res.ci_low), "ci_high": float(res.ci_high),
                      "abs_theta": abs(float(res.theta)),
                      "sd_T": float(T[mask].std(ddof=1)),
                      "excludes_zero": bool(res.ci_low > 0 or res.ci_high < 0)}

    if "full" in out and "no_land" in out:
        a, b = out["full"]["abs_theta"], out["no_land"]["abs_theta"]
        out["pct_attenuation"] = float(100 * (1 - b / a)) if a else float("nan")
        # run_dml restandardises T on each subsample; strip that off so the two
        # estimates are on one sigma scale before the drop is interpreted.
        rescale = out["no_land"]["sd_T"] / out["full"]["sd_T"]
        out["pct_attenuation_net_of_rescaling"] = (
            float(100 * (1 - (b / rescale) / a)) if a else float("nan"))
    return out


def random_effects(theta: np.ndarray, se: np.ndarray) -> tuple[float, float, float]:
    v = se ** 2
    w = 1 / v
    mu = (w * theta).sum() / w.sum()
    Q = (w * (theta - mu) ** 2).sum()
    c = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - (len(theta) - 1)) / c)
    w2 = 1 / (v + tau2)
    return float((w2 * theta).sum() / w2.sum()), float((1 / w2.sum()) ** 0.5), float(tau2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city")
    ap.add_argument("--all_12", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cities = ALL_12 if args.all_12 else [args.city]
    if any(c is None for c in cities):
        raise SystemExit("specify --city or --all_12")

    rows = [r for r in (audit_city(c, seed=args.seed) for c in cities) if r]

    summary: dict = {"per_city": rows}
    thin = [r["city"] for r in rows if not r["has_rich_confounders"]]
    if thin:
        summary["warning_thin_confounders"] = thin
        print("\n!! no parcel file, confounders are latitude/longitude only, "
              f"NOT the published specification: {', '.join(thin)}")
    ok = [r for r in rows
          if "full" in r and "no_land" in r and r["has_rich_confounders"]]
    if len(ok) > 1:
        for key in ("full", "no_land"):
            th = np.array([abs(r[key]["theta"]) for r in ok])
            se = np.array([r[key]["se"] for r in ok])
            mu, mu_se, tau2 = random_effects(th, se)
            summary[f"pooled_{key}"] = {"theta": mu, "se": mu_se, "tau2": tau2,
                                        "k": len(ok)}
        land = np.array([r["pct_land"] for r in ok])
        for key in ("full", "no_land"):
            th = np.array([abs(r[key]["theta"]) for r in ok])
            summary[f"corr_pct_land_vs_theta_{key}"] = float(np.corrcoef(land, th)[0, 1])

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2))

    print(f"\n{'city':14s}{'rich':>6s}{'%land':>8s}{'|θ| full':>10s}"
          f"{'|θ| no-land':>13s}{'atten':>8s}{'net':>8s}")
    for r in sorted(rows, key=lambda r: -r["pct_land"]):
        if "full" not in r or "no_land" not in r:
            continue
        print(f"{r['city']:14s}{str(r['has_rich_confounders']):>6s}{r['pct_land']:7.1f}%"
              f"{r['full']['abs_theta']:10.4f}{r['no_land']['abs_theta']:13.4f}"
              f"{r['pct_attenuation']:7.1f}%{r['pct_attenuation_net_of_rescaling']:7.1f}%")
    if "pooled_full" in summary:
        pf, pn = summary["pooled_full"], summary["pooled_no_land"]
        print(f"\npooled  full   θ={pf['theta']:+.4f} (se {pf['se']:.4f}) τ²={pf['tau2']:.4f}")
        print(f"pooled  no-land θ={pn['theta']:+.4f} (se {pn['se']:.4f}) τ²={pn['tau2']:.4f}")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
