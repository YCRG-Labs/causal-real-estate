"""Per-listing leverage diagnostic for the 12-city DML run.

Reuses fast_bootstrap_dml_v2._dml_core with the SAME ridge learner and the
SAME 5-fold split as the production run (return_residuals=True surfaces
Y_resid/T_resid) so the IF scores match exactly what the wild bootstrap saw.

Reports per city:
  - theta_hat, se_IF
  - kappa_DML proxy (1 / E[T_resid^2]) — diverges when the score denominator
    collapses (Saco 2026 arXiv:2512.07083)
  - max h_leverage and top-{1,5,10}% variance share of |IC|
  - skewness and kurtosis of the IC distribution
  - top-5 |IC| rows
  - theta_{-k} sensitivity for k in {1, 2, 5, 10}
  - Huber-trimmed theta_HUB (c=1.345)
  - fold-jackknife: theta dispersion across the 5 folds (any single fold can
    pin the bootstrap if its psi-mean blows up)

A consolidated CSV at results/leverage_all12.csv lets the JBES forest-plot
script flag cities where se_cb dispersion is leverage-driven (legit) vs
fold-pathological (bug to fix before B=1000).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "data" / "scripts"))

from fast_bootstrap_dml_v2 import (
    _build_features, _dml_core, _ridge_oof_predict, _pc1,
)
from diagnose_leverage import (
    compute_influence_and_leverage, theta_minus_top_k, trimmed_dml_huber,
)

CITIES_12 = ["boston", "nyc", "sf", "dc", "philadelphia", "chicago",
             "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]


def fold_jackknife_theta(Y_resid, T_resid, fold_ids):
    """Per-fold theta and IF-SE: each fold computes theta on its own residuals.

    If one fold's E[T_resid^2] is ~0 or its theta is wildly different from the
    full-sample theta, the bootstrap (which keeps the fold structure under
    resampling) inherits that pathology and explodes.
    """
    rows = []
    for k in sorted(set(fold_ids.tolist())):
        mask = fold_ids == k
        Yk, Tk = Y_resid[mask], T_resid[mask]
        denom = float(np.mean(Tk ** 2))
        if denom < 1e-12:
            rows.append({"fold": int(k), "n": int(mask.sum()),
                         "theta_k": float("nan"), "denom_k": denom,
                         "max_abs_T": float(np.max(np.abs(Tk)))})
            continue
        theta_k = float(np.mean(Tk * Yk) / denom)
        rows.append({"fold": int(k), "n": int(mask.sum()),
                     "theta_k": theta_k, "denom_k": denom,
                     "max_abs_T": float(np.max(np.abs(Tk)))})
    return pd.DataFrame(rows)


def diagnose_one(city, seed=0, k_folds=5):
    print(f"\n=========================  {city}  =========================")
    data = _build_features(city, use_cache=True, drop_crime=False)
    if data is None:
        print(f"[{city}] no data"); return None
    T, conf, Y, coords, meta = data
    n = len(Y)

    scaler = StandardScaler().fit(conf)
    conf_s = scaler.transform(conf).astype(np.float64, copy=False)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    splits = list(kf.split(np.arange(n)))

    res = _dml_core(T, conf_s, Y, learner_fn=_ridge_oof_predict, seed=seed,
                   k_folds=k_folds, splits=splits, return_residuals=True)
    theta, se_if, Y_resid, T_resid, fold_ids = res

    diag = compute_influence_and_leverage(T_resid, Y_resid, top_k=5)
    IC = diag["IC"]
    denom = diag["denom_T_resid_sq"]
    kappa = 1.0 / max(denom, 1e-12)

    ic_abs_sorted = np.sort(np.abs(IC))[::-1]
    total_var = float((IC ** 2).sum())
    share = lambda frac: float((ic_abs_sorted[:max(1, int(frac * n))] ** 2).sum() / max(total_var, 1e-30))
    skew_ic = float(stats.skew(IC, bias=False))
    kurt_ic = float(stats.kurtosis(IC, bias=False, fisher=True))

    sens = theta_minus_top_k(T, conf, Y, k_list=(1, 2, 5, 10), k_folds=k_folds, seed=seed)
    try:
        huber = trimmed_dml_huber(T, conf, Y, c=1.345, k_folds=k_folds, seed=seed)
    except Exception as e:
        print(f"  Huber failed: {e}", file=sys.stderr)
        huber = {"theta_huber": float("nan"), "se_huber": float("nan"),
                "theta_ols": theta, "n_clipped": -1, "sigma_hat": float("nan")}

    folds_df = fold_jackknife_theta(Y_resid, T_resid, fold_ids)
    theta_fold_sd = float(folds_df["theta_k"].dropna().std(ddof=1)) if folds_df["theta_k"].notna().sum() > 1 else float("nan")
    theta_fold_max_abs = float(folds_df["theta_k"].dropna().abs().max()) if folds_df["theta_k"].notna().any() else float("nan")
    denom_min = float(folds_df["denom_k"].min())

    summary = {
        "city": city, "n": int(n), "k_folds": k_folds,
        "theta_hat": float(theta), "se_IF": float(se_if),
        "denom_T_resid_sq": float(denom), "kappa_DML_proxy": float(kappa),
        "max_h_leverage": float(diag["h"].max()),
        "top1pct_IC_share": share(0.01),
        "top5pct_IC_share": share(0.05),
        "top10pct_IC_share": share(0.10),
        "top1_IC_share_of_var": float(np.max(IC ** 2) / max(total_var, 1e-30)),
        "ic_skew": skew_ic, "ic_excess_kurtosis": kurt_ic,
        "theta_drop1": float(sens.iloc[1]["theta"]),
        "theta_drop5": float(sens.iloc[3]["theta"]),
        "theta_drop10": float(sens.iloc[4]["theta"]),
        "theta_huber": float(huber["theta_huber"]),
        "se_huber": float(huber.get("se_huber", float("nan"))),
        "theta_huber_minus_ols": float(huber["theta_huber"] - theta),
        "n_huber_clipped": int(huber["n_clipped"]),
        "fold_theta_sd": theta_fold_sd,
        "fold_theta_max_abs": theta_fold_max_abs,
        "fold_denom_min": denom_min,
        "fold_thetas": folds_df["theta_k"].tolist(),
        "fold_denoms": folds_df["denom_k"].tolist(),
    }
    print(f"  theta_hat={theta:+.4f}  se_IF={se_if:.4f}  kappa_DML~{kappa:7.2f}")
    print(f"  max_h={diag['h'].max():.4f} (unif={1/n:.4f})  top1_IC_share_of_var={summary['top1_IC_share_of_var']:.3f}")
    print(f"  top-1%={summary['top1pct_IC_share']:.3f}  top-5%={summary['top5pct_IC_share']:.3f}  top-10%={summary['top10pct_IC_share']:.3f}")
    print(f"  IC skew={skew_ic:+.2f}  excess kurtosis={kurt_ic:+.2f}")
    print(f"  theta_{{-1}}={summary['theta_drop1']:+.4f}  theta_{{-5}}={summary['theta_drop5']:+.4f}  theta_{{-10}}={summary['theta_drop10']:+.4f}")
    print(f"  Huber theta={summary['theta_huber']:+.4f}  (gap from OLS={summary['theta_huber_minus_ols']:+.4f}, clipped={summary['n_huber_clipped']}/{n})")
    print(f"  fold thetas: {[f'{x:+.3f}' if not math.isnan(x) else 'nan' for x in summary['fold_thetas']]}  sd={theta_fold_sd:.3f}")
    print(f"  fold E[T^2] min={denom_min:.4f}")
    return summary


def main():
    cities = sys.argv[1:] if len(sys.argv) > 1 else CITIES_12
    t0 = time.time()
    rows = []
    for c in cities:
        try:
            r = diagnose_one(c)
            if r is not None:
                rows.append(r)
        except Exception as e:
            import traceback
            print(f"\n[{c}] FAILED: {e}", file=sys.stderr)
            traceback.print_exc()
    if not rows:
        print("No cities succeeded.", file=sys.stderr); return 1
    df = pd.DataFrame(rows)
    out_csv = REPO_ROOT / "results" / "leverage_all12.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    out_json = REPO_ROOT / "results" / "leverage_all12.json"
    out_json.write_text(json.dumps(rows, indent=2, default=float))

    print("\n" + "=" * 88)
    print("CONSOLIDATED LEVERAGE TABLE")
    print("=" * 88)
    cols_show = ["city", "n", "theta_hat", "se_IF", "kappa_DML_proxy",
                "max_h_leverage", "top1_IC_share_of_var", "top5pct_IC_share",
                "ic_excess_kurtosis", "theta_drop1", "theta_drop10",
                "theta_huber_minus_ols", "fold_theta_sd"]
    pd.options.display.float_format = lambda x: f"{x:8.3f}"
    print(df[cols_show].to_string(index=False))
    print(f"\nWrote {out_csv} and {out_json} in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
