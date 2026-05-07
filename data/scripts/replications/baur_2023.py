"""Replication of Baur, Rosenfelder & Lutz (2023, ESWA).

Baur, K., Rosenfelder, M. & Lutz, B. (2023). "Automated real estate valuation
with machine learning models using property descriptions." Expert Systems
with Applications 213, 119147.

Method (faithful to the paper, scaled to our SF data):

  1. Train a Gradient Boosting regressor on (a) structured features only and
     (b) structured features + 768-dim BERT (mpnet) embedding.
  2. Report MAE / MAPE / RMSE under 5-fold CV. Baur's headline finding is
     that adding BERT to the structured baseline reduces error.
  3. Re-run the project DML pipeline with the BERT embedding as the
     continuous treatment (PC1 inside the DML). This is identical to the
     production analysis on SF and exists here so the predictive gain and
     the causal null can be read off the same record.

Notes on engine choice: the paper uses LightGBM. On a clean CPython 3.9
without libomp the LightGBM dynamic library will not load on macOS; we fall
back to sklearn.GradientBoostingRegressor and record the substitution in the
output JSON. The qualitative finding (text helps prediction; DML θ stays
near zero) does not depend on the booster choice.

Usage:
  python baur_2023.py
  python baur_2023.py --out results/replications/baur.json
  python baur_2023.py --n 200            # subset for smoke tests
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from causal_inference import (
    EMBEDDING_DIM,
    get_features_and_target,
    load_analysis_data,
)
from replications.compare_to_dml import result_to_dict, run_dml


def _try_import_lightgbm():
    """Return a callable ((X_tr, y_tr) -> fitted model) for LightGBM if
    importable, otherwise None."""
    try:
        import lightgbm as lgb  # noqa: F401
        return "lightgbm"
    except Exception:
        return None


@dataclass
class CVMetrics:
    label: str
    n: int
    n_features: int
    mae: float
    mae_sd: float
    mape: float
    mape_sd: float
    rmse: float
    rmse_sd: float


def _make_regressor(engine: str, seed: int = 42):
    if engine == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        )
    return GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=seed,
    )


def _fit_predict(engine: str, X_tr, y_tr, X_te, seed: int = 42) -> np.ndarray:
    model = _make_regressor(engine, seed=seed)
    if engine == "lightgbm":
        model.fit(X_tr, y_tr)
    else:
        model.fit(X_tr, y_tr)
    return model.predict(X_te)


def cv_metrics(
    X: np.ndarray,
    y_log: np.ndarray,
    engine: str,
    label: str,
    k: int = 5,
    seed: int = 42,
) -> CVMetrics:
    """5-fold CV on log_price; metrics reported on the original price scale.

    Following Baur, the GBM's ranking quality is what matters; reporting MAE
    on dollars (after exponentiating) is the headline they use.
    """
    n = len(y_log)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    maes, mapes, rmses = [], [], []
    for tr, te in kf.split(np.arange(n)):
        pred_log = _fit_predict(engine, X[tr], y_log[tr], X[te], seed=seed)
        pred = np.exp(pred_log)
        actual = np.exp(y_log[te])
        err = pred - actual
        maes.append(float(np.mean(np.abs(err))))
        mapes.append(float(np.mean(np.abs(err) / np.maximum(actual, 1.0))))
        rmses.append(float(np.sqrt(np.mean(err ** 2))))

    return CVMetrics(
        label=label,
        n=int(n),
        n_features=int(X.shape[1]),
        mae=float(np.mean(maes)),
        mae_sd=float(np.std(maes, ddof=1)),
        mape=float(np.mean(mapes)),
        mape_sd=float(np.std(mapes, ddof=1)),
        rmse=float(np.mean(rmses)),
        rmse_sd=float(np.std(rmses, ddof=1)),
    )


def run_baur(
    city: str = "sf",
    n_subset: int | None = None,
    seed: int = 42,
    k_folds: int = 5,
) -> dict:
    print(f"\n=== Baur, Rosenfelder & Lutz (2023) replication: {city} ===")
    loaded = load_analysis_data(city)
    if loaded is None:
        return {"city": city, "error": "no data"}
    emb_df, parcels = loaded
    feats = get_features_and_target(emb_df, parcels, drop_mismatched_crime=True)
    if feats is None:
        return {"city": city, "error": "no features"}
    T_emb, confounders, Y_log, meta = feats

    if n_subset is not None and n_subset < len(Y_log):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(Y_log), size=n_subset, replace=False)
        idx.sort()
        T_emb = T_emb[idx]
        confounders = confounders[idx]
        Y_log = Y_log[idx]

    n = len(Y_log)
    print(f"  N={n:,}, structured features={confounders.shape[1]}, "
          f"embedding dim={T_emb.shape[1]}")

    engine = _try_import_lightgbm() or "sklearn-gbr"
    if engine == "lightgbm":
        print("  Engine: LightGBM (matches Baur)")
    else:
        print("  Engine: sklearn.GradientBoostingRegressor "
              "(LightGBM not loadable; recorded as deviation in JSON)")

    print(f"  5-fold CV: structured-only ({confounders.shape[1]} features)...")
    m_struct = cv_metrics(confounders, Y_log, engine, "structured", k=k_folds, seed=seed)
    print(f"    MAE=${m_struct.mae:,.0f}  MAPE={100*m_struct.mape:.2f}%  "
          f"RMSE=${m_struct.rmse:,.0f}")

    X_combined = np.hstack([confounders, T_emb])
    print(f"  5-fold CV: structured + BERT ({X_combined.shape[1]} features)...")
    m_text = cv_metrics(X_combined, Y_log, engine, "structured+BERT", k=k_folds, seed=seed)
    print(f"    MAE=${m_text.mae:,.0f}  MAPE={100*m_text.mape:.2f}%  "
          f"RMSE=${m_text.rmse:,.0f}")

    delta_mae = m_struct.mae - m_text.mae
    delta_mape = m_struct.mape - m_text.mape
    delta_rmse = m_struct.rmse - m_text.rmse
    rel_mae = delta_mae / m_struct.mae if m_struct.mae > 0 else float("nan")
    print(f"  Predictive gain (structured → +BERT):")
    print(f"    ΔMAE  = {delta_mae:+,.0f}  ({100*rel_mae:+.2f}% of structured MAE)")
    print(f"    ΔMAPE = {100*delta_mape:+.2f} pp")
    print(f"    ΔRMSE = {delta_rmse:+,.0f}")

    print("  DML: BERT embedding as continuous treatment (PC1 inside DML)...")
    dml = run_dml(T_emb, confounders, Y_log, label="DML on BERT PC1")
    if dml is None:
        print("    DML failed (treatment fully explained by confounders)")
    else:
        flag = "contains 0" if dml.contains_zero else "EXCLUDES 0"
        print(f"    DML θ={dml.theta:+.4f}  se={dml.se:.4f}  "
              f"95%CI=[{dml.ci_low:+.4f}, {dml.ci_high:+.4f}]  {flag}")
        print(f"    MDE: ±{dml.mde:.4f}  ({100*(np.exp(dml.mde)-1):+.2f}% in price)")

    print("\n  Headline contrast:")
    print(f"    Predictive MAE gain from text:  ${delta_mae:+,.0f} "
          f"({100*rel_mae:+.2f}% of structured MAE)")
    if dml is not None:
        print(f"    DML θ on PC1 of BERT (causal):  {dml.theta:+.4f} per σ "
              f"(95% CI [{dml.ci_low:+.4f}, {dml.ci_high:+.4f}])")
        if dml.contains_zero and delta_mae > 0:
            print("    → text reduces predictive error but causal effect cannot be "
                  "distinguished from zero under full confounder adjustment.")

    return {
        "paper": "Baur, Rosenfelder & Lutz 2023 (ESWA)",
        "city": city,
        "n": int(n),
        "engine": engine,
        "n_structured_features": int(confounders.shape[1]),
        "embedding_dim": int(T_emb.shape[1]),
        "k_folds": int(k_folds),
        "cv_structured": asdict(m_struct),
        "cv_structured_plus_bert": asdict(m_text),
        "predictive_gain": {
            "delta_mae": float(delta_mae),
            "delta_mae_relative": float(rel_mae),
            "delta_mape": float(delta_mape),
            "delta_rmse": float(delta_rmse),
        },
        "dml_bert": result_to_dict(dml),
        "method_notes": {
            "engine_planned": "lightgbm",
            "engine_used": engine,
            "engine_substituted": engine != "lightgbm",
            "engine_substitution_reason": (
                None if engine == "lightgbm"
                else "lightgbm dynamic library failed to load on this host "
                     "(missing libomp); sklearn GradientBoostingRegressor used "
                     "as the closest in-distribution boosted-tree learner"
            ),
            "embedding_model": "all-mpnet-base-v2",
            "embedding_dim_planned": EMBEDDING_DIM,
            "metric_basis": "log price → exponentiated to dollars",
            "dml_pipeline": "causal_inference.dml_continuous_treatment",
        },
        "meta": {
            "has_rich_confounders": bool(meta["has_rich_confounders"]),
            "crime_dropped": bool(meta["crime_dropped"]),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="sf")
    ap.add_argument("--n", type=int, default=None,
                    help="optional subset size for fast smoke runs")
    ap.add_argument("--k_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None,
                    help="path to write JSON results")
    args = ap.parse_args()

    result = run_baur(
        city=args.city, n_subset=args.n, seed=args.seed, k_folds=args.k_folds,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
