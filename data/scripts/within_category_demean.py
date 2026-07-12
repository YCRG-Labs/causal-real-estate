"""Exact within-property-type demeaning as the composition adjustment.

The published confounder block never encodes property_type, and the nearest-parcel
join cannot supply it: for a vacant lot it returns a neighbouring built parcel's
attributes. A logit of the land indicator on the confounders runs at chance while
the same logit on the text direction runs at 0.9+, so the text axis absorbs the
land-versus-house price gap the design cannot see.

The obvious fix, adding property_type dummies to the RidgeCV nuisance, does not
adjust for property type: the penalty shrinks a rare dummy toward zero, so in
Boston (non-residential 1.9%) it raises |theta| rather than lowering it. Exact
within-type demeaning of T and Y is unpenalised and removes the category means
regardless of prevalence. Residualising T, Y and X on a full set of category
dummies by OLS is algebraically identical to subtracting the category means, so
demeaning is the unpenalised fixed-effects adjustment, computed directly, and the
partialled-out residuals then feed the standard ridge cross-fit for X.

The treatment is standardised once by its raw pooled sigma before demeaning, and
that scale is held fixed across the raw and demeaned estimates (the demeaned core
is called directly, without the per-subsample restandardisation run_dml applies).
Both estimates are therefore d log price per one raw-sigma move along the text
axis, so the change between them is the property-type adjustment, not a rescaling.
No rows are dropped.

Per market:

    raw       theta(T,          X,               Y)     published listing-level spec
    dummy     theta(T,          [X, pt_dummies], Y)     dummies inside the ridge (backfires)
    demean    theta(T_dm,       X_dm,            Y_dm)   within-property_type FWL demeaning

with X the listing-level block [lat, lon, beds, baths, sqft, year_built]. The
composition screen reports the cross-fitted AUC of the land indicator on a
non-leaking imputed structural block (median-imputed, no missingness signature,
which a zero-fill would turn into a perfect land classifier) versus on the text
direction. The published-design blindness (AUC ~0.50 on the parcel confounder
block) is measured separately by land_mechanism_check.py on boston, nyc and sf.

    python data/scripts/within_category_demean.py --all_12
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

from replications.compare_to_dml import _ridge_dml_core, run_dml

POOLED_CSV = REPO / "results" / "replications" / "pooled_pca_treatment.csv"
OUT_JSON = REPO / "results" / "within_category_demean.json"

ALL_12 = ["boston", "nyc", "sf", "dc", "philadelphia", "chicago",
          "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]

STRUCT = ["beds", "baths", "sqft", "year_built"]
NON_RESIDENTIAL = r"land|lot|vacant|parking"


def _load(city: str):
    emb = pd.read_parquet(REPO / "data" / "processed" / f"{city}_embeddings.parquet")
    emb = emb.assign(listing_id=np.arange(len(emb)))

    pool = pd.read_csv(POOLED_CSV)
    pc = pool[pool.city == city][["listing_id", "treatment_z"]]
    df = emb.merge(pc, on="listing_id", how="inner")

    price = pd.to_numeric(df["price"], errors="coerce")
    ok = price.notna() & (price > 0) & df.latitude.notna() & df.longitude.notna()
    df = df[ok].reset_index(drop=True)
    Y = np.log(pd.to_numeric(df["price"], errors="coerce").to_numpy(float))
    return df, Y


def _zerofill_block(df: pd.DataFrame) -> np.ndarray:
    lat = pd.to_numeric(df.latitude, errors="coerce").to_numpy(float)
    lon = pd.to_numeric(df.longitude, errors="coerce").to_numpy(float)
    S = df[STRUCT].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return np.column_stack([lat, lon, np.nan_to_num(S, nan=0.0)])


def _impute_block(df: pd.DataFrame) -> np.ndarray:
    lat = pd.to_numeric(df.latitude, errors="coerce").to_numpy(float)
    lon = pd.to_numeric(df.longitude, errors="coerce").to_numpy(float)
    S = df[STRUCT].apply(pd.to_numeric, errors="coerce")
    S = S.fillna(S.median()).to_numpy(float)
    return np.column_stack([lat, lon, S])


def _demean_within(M: np.ndarray, codes: np.ndarray) -> np.ndarray:
    M = np.atleast_2d(M.T).T if M.ndim == 1 else M
    out = M.astype(float, copy=True)
    for g in np.unique(codes):
        sel = codes == g
        out[sel] -= out[sel].mean(axis=0, keepdims=True)
    return out


def _theta_from_core(T_1d: np.ndarray, X: np.ndarray, Y: np.ndarray) -> dict | None:
    res = _ridge_dml_core(T_1d, X, Y, k_folds=5, seed=42)
    if res is None:
        return None
    theta, se = res
    lo, hi = theta - 1.96 * se, theta + 1.96 * se
    return {"theta": float(theta), "abs_theta": abs(float(theta)), "se": float(se),
            "ci_low": float(lo), "ci_high": float(hi),
            "covers_zero": bool(lo < 0 < hi)}


def _theta_own_sigma(T_1d: np.ndarray, X: np.ndarray, Y: np.ndarray) -> dict | None:
    r = run_dml(T_1d.reshape(-1, 1), X, Y, label="x", ci_method="if", n_boot=None,
                use_ridge=True, seed=42, n_pca=1)
    if r is None:
        return None
    return {"theta": float(r.theta), "abs_theta": abs(float(r.theta)),
            "se": float(r.se), "covers_zero": bool(r.ci_low < 0 < r.ci_high)}


def _auc(Z: np.ndarray, y: np.ndarray) -> float:
    if y.sum() < 3 or (~y.astype(bool)).sum() < 3:
        return float("nan")
    p = cross_val_predict(LogisticRegression(max_iter=2000), Z, y, cv=5,
                          method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def run_city(city: str) -> dict:
    df, Y = _load(city)
    T_raw = df.treatment_z.to_numpy(float)
    t_sd = float(np.std(T_raw, ddof=1))
    T = (T_raw - T_raw.mean()) / t_sd if t_sd > 0 else T_raw

    X = _zerofill_block(df)
    ptype = df.property_type.astype(str)
    codes = pd.Categorical(ptype).codes
    nonres = ptype.str.contains(NON_RESIDENTIAL, case=False, na=False)
    land = nonres.to_numpy().astype(int)

    T_dm = _demean_within(T, codes)
    Y_dm = _demean_within(Y, codes)

    pt_dummies = pd.get_dummies(ptype, prefix="pt", drop_first=True).to_numpy(float)
    X_with_dummies = np.column_stack([X, pt_dummies])

    Ximp = _impute_block(df)
    out = {
        "city": city, "n": int(len(Y)),
        "n_property_types": int(ptype.nunique()),
        "pct_non_residential": float(100 * nonres.mean()),
        "auc_land_given_structure": _auc(StandardScaler().fit_transform(Ximp), land),
        "auc_land_given_text": _auc(T.reshape(-1, 1), land),
        "raw": _theta_from_core(T, X, Y),
        "dummy_in_ridge": _theta_from_core(T, X_with_dummies, Y),
        "demean": _theta_from_core(T_dm, X, Y_dm),
        "demean_own_sigma": _theta_own_sigma(T_dm, X, Y_dm),
        "sd_T_within": float(np.std(T_dm, ddof=1)),
    }
    a = out["raw"]["abs_theta"]
    out["pct_attenuation_demean_fixed_scale"] = (
        float(100 * (1 - out["demean"]["abs_theta"] / a)) if a else float("nan"))
    out["pct_attenuation_demean_own_sigma"] = (
        float(100 * (1 - out["demean_own_sigma"]["abs_theta"] / a))
        if a and out["demean_own_sigma"] else float("nan"))
    out["pct_change_dummy"] = (
        float(100 * (out["dummy_in_ridge"]["abs_theta"] / a - 1)) if a else float("nan"))
    return out


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr = pd.Series(x).rank().to_numpy()
    yr = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(xr, yr)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city")
    ap.add_argument("--all_12", action="store_true")
    args = ap.parse_args()
    cities = ALL_12 if args.all_12 else [args.city]

    rows = []
    for c in cities:
        try:
            rows.append(run_city(c))
        except Exception as e:  # noqa: BLE001
            print(f"  {c}: FAILED {type(e).__name__}: {e}")

    summary: dict = {"per_city": rows}
    graded = [r for r in rows if r["pct_non_residential"] > 0
              and not np.isnan(r["pct_attenuation_demean_own_sigma"])
              and r["city"] != "dc"]
    if len(graded) > 2:
        share = np.array([r["pct_non_residential"] for r in graded])
        for scale in ("own_sigma", "fixed_scale"):
            atten = np.array([r[f"pct_attenuation_demean_{scale}"] for r in graded])
            summary[f"gradient_share_vs_attenuation_{scale}"] = {
                "pearson": float(np.corrcoef(share, atten)[0, 1]),
                "spearman": _spearman(share, atten),
                "k": len(graded), "excludes": ["dc"]}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2))

    hdr = (f"{'city':13s}{'n':>6s}{'%nres':>7s}{'AUCtxt':>8s}{'raw':>8s}"
           f"{'demean_fx':>10s}{'atten_fx':>9s}{'demean_sd':>10s}{'atten_sd':>9s}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: -r["pct_non_residential"]):
        ds = r["demean_own_sigma"]["abs_theta"] if r["demean_own_sigma"] else float("nan")
        print(f"{r['city']:13s}{r['n']:6d}{r['pct_non_residential']:6.1f}%"
              f"{r['auc_land_given_text']:8.3f}{r['raw']['abs_theta']:8.4f}"
              f"{r['demean']['abs_theta']:10.4f}{r['pct_attenuation_demean_fixed_scale']:8.1f}%"
              f"{ds:10.4f}{r['pct_attenuation_demean_own_sigma']:8.1f}%")
    print("\n  _fx = fixed raw-sigma scale (per-unit slope, no rescaling)")
    print("  _sd = re-standardised to the demeaned treatment's own sigma")
    for scale in ("fixed_scale", "own_sigma"):
        key = f"gradient_share_vs_attenuation_{scale}"
        if key in summary:
            g = summary[key]
            print(f"  gradient share vs atten [{scale:11s}] (k={g['k']}, ex-DC): "
                  f"Pearson {g['pearson']:+.2f}  Spearman {g['spearman']:+.2f}")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
