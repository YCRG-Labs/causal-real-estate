"""Negative-control validation for the text-on-price DML estimate.

Two placebos:
  - NCO (negative control outcome): outcome cannot causally depend on text.
    Re-run DML with Y replaced by parcel area or year built. Both are pre-
    treatment physical facts about the property; listing copy cannot affect
    them. A non-null result indicates pipeline bias, not a real effect.
  - NCE (negative control exposure): treatment cannot causally affect outcome.
    Re-run DML with T replaced by row-permuted embeddings (across all rows)
    and within-stratum-permuted embeddings (within zip-code bins, preserves
    coarse covariate-treatment correlation while breaking the row link).

Schuemie empirical calibration: fits a Gaussian empirical null to the panel
of K placebo NCE estimates and reports a calibrated p-value for the focal
real-data DML estimate. Frames the focal estimate against the bias floor of
the pipeline rather than the textbook null.

Refs:
  Lipsitch, Tchetgen Tchetgen & Cohen 2010 — Epidemiology 21(3):383-388
  Shi, Miao, Tchetgen Tchetgen 2020 — Curr Epidemiol Rep 7(4):190-202
  Schuemie et al. 2014, 2016 — Stat Med 33(2), 35(22)

Usage:
  python negative_controls.py --city sf
  python negative_controls.py --city sf --n_nce 50
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from causal_inference import (
    PROPERTY_COLS,
    dml_continuous_treatment,
    get_features_and_target,
    load_analysis_data,
)
from config import CITIES


@dataclass
class DMLResult:
    label: str
    n: int
    theta: float
    se: float
    ci_low: float
    ci_high: float
    contains_zero: bool


def _silent_dml(T, confounders, Y) -> dict | None:
    """Run DML without the chatty prints from causal_inference.py."""
    import io
    import contextlib

    with contextlib.redirect_stdout(io.StringIO()):
        return dml_continuous_treatment(T, confounders, Y)


def _to_result(label: str, n: int, raw: dict | None) -> DMLResult | None:
    if raw is None:
        return None
    lo, hi = raw["ci"]
    return DMLResult(
        label=label,
        n=int(n),
        theta=float(raw["theta"]),
        se=float(raw["se"]),
        ci_low=float(lo),
        ci_high=float(hi),
        contains_zero=bool(lo <= 0 <= hi),
    )


def lift_nco_from_property(
    T: np.ndarray,
    confounders: np.ndarray,
    Y_real: np.ndarray,
    property_cols: list[str],
    target: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Pull a property column out of the confounder set to use as a placebo Y.

    `confounders` here is the FULL confounder matrix passed to DML, which is
    [lat,lon, property_block, contextual_block]. We need the property_cols
    list (in order, as returned from get_features_and_target's joined dict)
    to know where each property column lives in the matrix.

    Returns (T, confounders_minus_target, Y_placebo) or None if not found.
    """
    if target not in property_cols:
        return None
    # confounder layout: [lat, lon, *property_cols, *contextual_cols]
    # so property_cols start at index 2
    PROP_OFFSET = 2
    target_idx = PROP_OFFSET + property_cols.index(target)
    Y_placebo = confounders[:, target_idx].copy()
    keep = np.ones(confounders.shape[1], dtype=bool)
    keep[target_idx] = False
    confounders_drop = confounders[:, keep]
    valid = ~(np.isnan(Y_placebo) | (Y_placebo <= 0))
    if valid.sum() < 50:
        return None
    return T[valid], confounders_drop[valid], np.log(Y_placebo[valid])


def permute_treatment(T: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row-permute the embedding matrix (NCE: random)."""
    idx = rng.permutation(T.shape[0])
    return T[idx]


def permute_within_strata(
    T: np.ndarray, strata: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Permute T rows only within each stratum (NCE: stratified)."""
    out = T.copy()
    for s in np.unique(strata):
        mask = strata == s
        if mask.sum() < 2:
            continue
        idx = np.where(mask)[0]
        out[idx] = T[rng.permutation(idx)]
    return out


def schuemie_calibration(
    placebos: list[DMLResult], focal: DMLResult
) -> dict:
    """Fit a Gaussian empirical null to placebo θ's; calibrate the focal p.

    Following Schuemie et al. 2014/2016, fit N(μ, σ²) to placebo point
    estimates (the null is wider than nominal SE because of systematic bias),
    then compute the calibrated two-sided p-value for the focal estimate
    under that empirical null.

    Returns: {mu, sigma_emp, sigma_inflation, p_calibrated, p_nominal}
    """
    thetas = np.array([p.theta for p in placebos])
    ses = np.array([p.se for p in placebos])
    if len(thetas) < 5:
        return {"error": f"need >=5 placebos, have {len(thetas)}"}
    mu = float(thetas.mean())
    sigma_emp = float(thetas.std(ddof=1))
    nominal_se_med = float(np.median(ses))
    sigma_inflation = sigma_emp / nominal_se_med if nominal_se_med > 0 else float("nan")
    z_cal = (focal.theta - mu) / max(sigma_emp, focal.se)
    p_cal = float(2 * (1 - stats.norm.cdf(abs(z_cal))))
    z_nom = focal.theta / focal.se if focal.se > 0 else float("nan")
    p_nom = float(2 * (1 - stats.norm.cdf(abs(z_nom))))
    return {
        "mu_empirical_null": mu,
        "sigma_empirical_null": sigma_emp,
        "median_nominal_se": nominal_se_med,
        "sigma_inflation_factor": sigma_inflation,
        "z_calibrated": float(z_cal),
        "z_nominal": float(z_nom),
        "p_calibrated": p_cal,
        "p_nominal": p_nom,
    }


def run_negative_controls(city: str, n_nce: int = 30, seed: int = 42) -> dict:
    print(f"\n=== Negative controls: {city} ===")
    loaded = load_analysis_data(city)
    if loaded is None:
        return {"city": city, "error": "no data"}
    emb_df, parcels = loaded
    feats = get_features_and_target(emb_df, parcels, drop_mismatched_crime=True)
    if feats is None:
        return {"city": city, "error": "no features"}
    T, confounders, Y, meta = feats
    print(f"  N={len(Y):,}, embedding dim={T.shape[1]}, confounders={confounders.shape[1]}")

    if not meta["has_rich_confounders"]:
        return {
            "city": city,
            "error": "rich confounders unavailable; negative controls require property-level join",
        }

    available_property_cols = [c for c in PROPERTY_COLS if c in parcels.columns]

    # ---- focal: real DML on log-price ----
    focal_raw = _silent_dml(T, confounders, Y)
    focal = _to_result("real (log-price)", len(Y), focal_raw)
    print(f"\n  focal estimate: θ={focal.theta:+.4f}  se={focal.se:.4f}  "
          f"95%CI=[{focal.ci_low:+.4f}, {focal.ci_high:+.4f}]  "
          f"{'contains 0' if focal.contains_zero else 'EXCLUDES 0'}")

    # ---- NCO panel ----
    print("\n  NCO panel (placebo outcomes):")
    nco_results: list[DMLResult] = []
    for nco_target in ("lot_area_sqft", "year_built"):
        lifted = lift_nco_from_property(
            T, confounders, Y, available_property_cols, nco_target
        )
        if lifted is None:
            print(f"    {nco_target}: skipped (column not present)")
            continue
        T_n, conf_n, Y_n = lifted
        raw = _silent_dml(T_n, conf_n, Y_n)
        res = _to_result(f"NCO: {nco_target}", len(Y_n), raw)
        if res is None:
            print(f"    {nco_target}: DML failed")
            continue
        nco_results.append(res)
        flag = "✓ null" if res.contains_zero else "✗ EFFECT"
        print(f"    {nco_target}: θ={res.theta:+.4f}  se={res.se:.4f}  "
              f"95%CI=[{res.ci_low:+.4f}, {res.ci_high:+.4f}]  {flag}")

    # ---- NCE panel ----
    print(f"\n  NCE panel (placebo treatments, {n_nce} draws each):")
    rng = np.random.default_rng(seed)
    nce_random: list[DMLResult] = []
    nce_strata: list[DMLResult] = []
    strata = meta["zip_labels"]

    for k in range(n_nce):
        T_perm = permute_treatment(T, rng)
        raw = _silent_dml(T_perm, confounders, Y)
        res = _to_result(f"NCE-random[{k}]", len(Y), raw)
        if res is not None:
            nce_random.append(res)

        T_strat = permute_within_strata(T, strata, rng)
        raw = _silent_dml(T_strat, confounders, Y)
        res = _to_result(f"NCE-stratified[{k}]", len(Y), raw)
        if res is not None:
            nce_strata.append(res)

    def _summary(label: str, panel: list[DMLResult]):
        thetas = np.array([r.theta for r in panel])
        coverage = np.mean([r.contains_zero for r in panel])
        print(f"    {label}: K={len(panel)}, "
              f"mean θ={thetas.mean():+.4f}, sd={thetas.std(ddof=1):.4f}, "
              f"95% coverage of 0: {coverage*100:.0f}%")

    _summary("random row-perm  ", nce_random)
    _summary("stratified perm  ", nce_strata)

    # ---- Schuemie empirical calibration ----
    print("\n  Schuemie empirical calibration (against random row-perm panel):")
    cal = schuemie_calibration(nce_random, focal)
    if "error" not in cal:
        print(f"    empirical null: N(μ={cal['mu_empirical_null']:+.4f}, "
              f"σ={cal['sigma_empirical_null']:.4f})")
        print(f"    σ-inflation vs median nominal SE: {cal['sigma_inflation_factor']:.2f}×")
        print(f"    z calibrated: {cal['z_calibrated']:+.2f} → p_cal = {cal['p_calibrated']:.3f}")
        print(f"    z nominal:    {cal['z_nominal']:+.2f} → p_nom = {cal['p_nominal']:.3f}")

    return {
        "city": city,
        "focal": asdict(focal),
        "nco": [asdict(r) for r in nco_results],
        "nce_random": [asdict(r) for r in nce_random],
        "nce_stratified": [asdict(r) for r in nce_strata],
        "calibration": cal,
        "meta": {
            "n_obs": int(len(Y)),
            "embedding_dim": int(T.shape[1]),
            "n_confounders": int(confounders.shape[1]),
            "has_rich_confounders": bool(meta["has_rich_confounders"]),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=CITIES, required=True)
    ap.add_argument("--n_nce", type=int, default=30, help="number of NCE permutation draws")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None,
                    help="optional path to write JSON results")
    args = ap.parse_args()

    result = run_negative_controls(args.city, n_nce=args.n_nce, seed=args.seed)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
