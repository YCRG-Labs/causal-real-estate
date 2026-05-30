"""
Fast bootstrap DML for the PC1-on-log-price headline.

Optimizations vs the canonical pipeline (preserves full-pipeline correctness):
  1. Cheap Bootstrap (Lam 2022, JASA): B=50 replications with t-pivot CI
     instead of B=1000 percentile bootstrap. ~50x fewer reps, same O(n^-1)
     coverage error, intervals ~3% wider.
  2. Parallel loky workers with inner_max_num_threads=1 + LightGBM num_threads=1
     (avoids the documented oversubscription that kills the naive joblib path).
  3. n_rep=1 single 5-fold split per bootstrap rep (each Efron resample is its
     own split-randomness draw; per Bach et al. 2024 Remark 2.2).
  4. Randomized SVD for PC1 only (n_components=1, n_iter=7). ~10-30x faster
     than full PCA(n_components=50) when we only need the leading direction.
  5. LightGBM tuned for small n=348: num_leaves=8, min_data_in_leaf=30,
     max_bin=63, force_col_wise=True. ~3-5x per fit at this scale.

Usage:
    python fast_bootstrap_dml.py                       # all three cities, B=50
    python fast_bootstrap_dml.py sf nyc                # subset
    python fast_bootstrap_dml.py --B 200               # paranoid B
    python fast_bootstrap_dml.py --serial              # disable parallel for debugging
    python fast_bootstrap_dml.py sf --spatial_hac      # Conley + Salerno-Wu-McCormick jackknife HAC
    python fast_bootstrap_dml.py sf --buffered_kfold   # Emmenegger spatial-buffered cross-fit

References:
  Lam (2022, JASA) "A Cheap Bootstrap Method for Fast Inference" arXiv:2202.00090
  Bach et al. (2024) "Bootstrap consistency for general DML" arXiv:2604.17239
  Ohlendorff, Munch, Sorensen & Gerds (2025) "Cheap Subsampling bootstrap
    confidence intervals for fast and robust inference" arXiv:2501.10289.
    (Sub-sampling variant of Lam's cheap bootstrap, valid under
    cross-validation-induced duplicates.  Replaces the older
    Christensen-van-der-Laan attribution.)
  Conley (1999) JoE 92:1-45 — spatial HAC sandwich variance.
  Bester, Conley & Hansen (2011) JoE — Bartlett-kernel sandwich.
  Salerno, Wu & McCormick (2026) "Spatially Robust Inference with
    Predicted and Missing-at-Random Labels" arXiv:2603.11368
    (V_JK = V_off + V_between).
  Emmenegger, Spohn, Elmer & Buehlmann (2025) "Treatment Effect
    Estimation with Observational Network Data using ML" arXiv:2206.14591
    (spatial-buffered K-fold).
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _silence  # noqa: F401
import os
# CRITICAL: set thread caps BEFORE any numerical library imports. OpenMP reads
# OMP_NUM_THREADS once on first call and ignores later changes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys
import json
import time
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
from scipy import stats
from joblib import Parallel, delayed, parallel_config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from causal_inference import load_analysis_data, get_features_and_target
from spatial_se import (
    spatial_hac_se,
    salerno_jackknife_hac,
    buffered_kfold,
    bandwidth_sensitivity_sweep,
)


def _pc1_randomized(T, seed=0, weights=None):
    """Leading principal component via randomized SVD.

    weights=None  -> ordinary PC1 (each row weight 1).
    weights=w     -> weighted PC1 with mean-centering and unit-variance
                     scaling done under the weighted measure. This is the
                     Bayesian / fractional-weighted bootstrap PCA step.
    """
    if weights is None:
        Tc = T - T.mean(axis=0, keepdims=True)
        U, s, Vt = randomized_svd(Tc, n_components=1, n_iter=7,
                                  n_oversamples=10, random_state=seed)
        pc1 = U[:, 0] * s[0]
        sd = pc1.std()
        return pc1 / sd if sd > 0 else pc1

    w = np.asarray(weights, dtype=np.float64)
    mu = np.average(T, axis=0, weights=w)
    Tc = T - mu
    Tw = Tc * np.sqrt(w)[:, None]
    U, s, Vt = randomized_svd(Tw, n_components=1, n_iter=7, n_oversamples=10,
                              random_state=seed)
    v = Vt[0]
    pc1 = Tc @ v
    mean = np.average(pc1, weights=w)
    var = np.average((pc1 - mean) ** 2, weights=w)
    sd = np.sqrt(var) if var > 0 else 1.0
    return (pc1 - mean) / sd


def _make_lgbm():
    """Tuned LightGBM regressor for n≈280 training, d≈32 features."""
    import lightgbm as lgb
    return lgb.LGBMRegressor(
        n_estimators=200,
        num_leaves=8,            # default 31 over-leafs at n=280
        min_data_in_leaf=30,
        max_bin=63,              # default 255; small n + few features
        learning_rate=0.05,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        force_col_wise=True,
        n_jobs=1,                # critical: defer threading to outer joblib
        verbose=-1,
        random_state=42,
    )


def _pc1_crossfit(T, k_folds, seed=0):
    """Cross-fit PC1: for each fold, estimate the leading direction on the
    TRAIN rows and project the held-out rows onto it, sign-aligned to the
    full-sample direction so folds don't flip, standardized by the train
    projection's mean/sd. Out-of-fold treatment, fixes the generated-regressor
    undercoverage at small n."""
    n = T.shape[0]
    Tc = T - T.mean(axis=0, keepdims=True)
    _, _, Vt_full = randomized_svd(Tc, n_components=1, n_iter=7,
                                   n_oversamples=10, random_state=seed)
    ref = Vt_full[0]
    pc1 = np.zeros(n)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    for tr, te in kf.split(np.arange(n)):
        Tc_tr = T[tr] - T[tr].mean(axis=0, keepdims=True)
        _, _, Vt_tr = randomized_svd(Tc_tr, n_components=1, n_iter=7,
                                     n_oversamples=10, random_state=seed)
        v = Vt_tr[0]
        if np.dot(v, ref) < 0:
            v = -v
        proj_tr = (T[tr] - T[tr].mean(axis=0)) @ v
        mu = float(proj_tr.mean())
        sd = float(proj_tr.std()) or 1.0
        pc1[te] = ((T[te] - T[tr].mean(axis=0)) @ v - mu) / sd
    return pc1


def _dml_core_fast(T, conf, Y, k_folds=5, seed=0,
                   cross_fit_pca=False, weights=None,
                   return_residuals=False, splits=None):
    """Single DML estimate. Returns (theta, se_if) or None.

    weights: optional sample weights (Dirichlet for Bayesian bootstrap). When
    provided, pc1 uses weighted PCA and the partialling-out is weighted.
    return_residuals: if True, also return (Y_resid, T_resid, fold_ids).
    splits: optional list of (train_idx, eval_idx) tuples; if provided
        overrides the default random KFold (used for buffered spatial CV).
    """
    n = len(Y)
    if cross_fit_pca and weights is None:
        pc1 = _pc1_crossfit(T, k_folds, seed=seed)
    else:
        pc1 = _pc1_randomized(T, seed=seed, weights=weights)

    conf_s = StandardScaler().fit_transform(conf)
    if splits is None:
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        splits_iter = list(kf.split(np.arange(n)))
    else:
        splits_iter = list(splits)
    Y_resid = np.zeros(n)
    T_resid = np.zeros(n)
    fold_ids = np.full(n, -1, dtype=int)
    for k, (tr, te) in enumerate(splits_iter):
        m_y = _make_lgbm()
        m_t = _make_lgbm()
        if weights is None:
            m_y.fit(conf_s[tr], Y[tr])
            m_t.fit(conf_s[tr], pc1[tr])
        else:
            m_y.fit(conf_s[tr], Y[tr], sample_weight=weights[tr])
            m_t.fit(conf_s[tr], pc1[tr], sample_weight=weights[tr])
        Y_resid[te] = Y[te] - m_y.predict(conf_s[te])
        T_resid[te] = pc1[te] - m_t.predict(conf_s[te])
        fold_ids[te] = k

    if weights is None:
        denom = float(np.mean(T_resid ** 2))
        if denom < 1e-12:
            return None
        theta = float(np.mean(T_resid * Y_resid)) / denom
        se_if = float(np.sqrt(
            np.var((Y_resid - theta * T_resid) * T_resid / denom, ddof=1) / n
        ))
    else:
        denom = float(np.average(T_resid ** 2, weights=weights))
        if denom < 1e-12:
            return None
        theta = float(np.average(T_resid * Y_resid, weights=weights)) / denom
        se_if = None
    if return_residuals:
        return theta, se_if, Y_resid, T_resid, fold_ids
    return theta, se_if


def _boot_rep(T, conf, Y, rng_seed, mode="efron", cross_fit_pca=False):
    """One cheap bootstrap replicate. mode='efron' resamples row indices with
    replacement; mode='dirichlet' uses Dirichlet(1,...,1) weights (Rubin 1981
    Bayesian bootstrap, fractional-weighted variant per Greifer fwb)."""
    rng = np.random.default_rng(rng_seed)
    n = len(Y)
    if mode == "efron":
        idx = rng.integers(0, n, n)
        res = _dml_core_fast(T[idx], conf[idx], Y[idx], seed=rng_seed,
                             cross_fit_pca=cross_fit_pca, weights=None)
    elif mode == "dirichlet":
        w = rng.dirichlet(np.ones(n)) * n  # E[w_i]=1, sum=n
        res = _dml_core_fast(T, conf, Y, seed=rng_seed,
                             cross_fit_pca=False, weights=w)
    else:
        raise ValueError(f"unknown bootstrap mode {mode}")
    return res[0] if res is not None else np.nan


def fast_dml(city, B=50, k_folds=5, parallel=True, mode="efron",
             cross_fit_pca=False, drop_crime=False,
             spatial_hac=False, buffered_kfold_flag=False,
             buffer_quantile=0.05, bandwidth_quantile=0.10):
    spec_tag = (f"mode={mode} cross_fit_pca={cross_fit_pca} "
                f"drop_crime={drop_crime} spatial_hac={spatial_hac} "
                f"buffered_kfold={buffered_kfold_flag}")
    t0 = time.time()
    res = load_analysis_data(city)
    if res is None:
        print(f"[{city}] no data"); return None
    emb_df, parcels = res
    data = get_features_and_target(emb_df, parcels,
                                   drop_mismatched_crime=drop_crime)
    if data is None:
        print(f"[{city}] no price target"); return None
    T, conf, Y, meta = data
    T = np.asarray(T, dtype=np.float64)
    conf = np.asarray(conf, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = len(Y)

    coords = None
    if spatial_hac or buffered_kfold_flag:
        lat = np.asarray(meta.get("lat", []), dtype=np.float64)
        lon = np.asarray(meta.get("lon", []), dtype=np.float64)
        if lat.size == n and lon.size == n and np.isfinite(lat).all() and np.isfinite(lon).all():
            coords = np.column_stack([lat, lon])
        else:
            print(f"[{city}] spatial flags requested but lat/lon missing or non-finite; "
                  "spatial diagnostics will be skipped.")
            spatial_hac = False
            buffered_kfold_flag = False

    print(f"\n[{city}] n={n}  text_dim={T.shape[1]}  confounders={conf.shape[1]}  "
          f"rich={meta['has_rich_confounders']}  spec=({spec_tag})")

    splits_for_point = None
    if buffered_kfold_flag and coords is not None:
        splits_for_point = buffered_kfold(
            coords, k=k_folds, buffer_quantile=buffer_quantile, seed=0,
        )
        n_train_per_fold = [len(tr) for tr, _ in splits_for_point]
        print(f"  buffered K-fold (q_b={buffer_quantile:.2f}): n_train per fold = "
              f"{n_train_per_fold}")

    t1 = time.time()
    main = _dml_core_fast(T, conf, Y, k_folds=k_folds, seed=0,
                          cross_fit_pca=cross_fit_pca, weights=None,
                          return_residuals=(spatial_hac or False),
                          splits=splits_for_point)
    if main is None:
        print(f"[{city}] treatment fully explained by confounders"); return None
    if spatial_hac:
        theta_hat, se_if, Y_resid_main, T_resid_main, fold_ids_main = main
    else:
        theta_hat, se_if = main
        Y_resid_main = T_resid_main = fold_ids_main = None
    t_point = time.time() - t1

    se_str = f"{se_if:.4f}" if se_if is not None else "n/a"
    print(f"  point: θ̂ = {theta_hat:.4f}  (IF SE {se_str}, {t_point:.2f}s)")

    t2 = time.time()
    seeds = list(range(1, B + 1))
    if parallel:
        with parallel_config(backend="loky", inner_max_num_threads=1):
            boots = Parallel(n_jobs=-1, batch_size=1, verbose=0)(
                delayed(_boot_rep)(T, conf, Y, s, mode, cross_fit_pca)
                for s in seeds
            )
    else:
        boots = [_boot_rep(T, conf, Y, s, mode, cross_fit_pca) for s in seeds]
    boots = np.array([b for b in boots if not np.isnan(b)])
    t_boot = time.time() - t2

    # Cheap Bootstrap t-pivot CI (Lam 2022): centered at θ̂, divisor B not B-1
    se_cb = float(np.sqrt(np.mean((boots - theta_hat) ** 2)))
    t_q = stats.t.ppf(0.975, df=len(boots))
    ci_low = theta_hat - t_q * se_cb
    ci_high = theta_hat + t_q * se_cb
    mde = (1.96 + 0.84) * se_cb

    # ------------------- spatial HAC inference (#28) ----------------------
    spatial_hac_block = None
    if spatial_hac and Y_resid_main is not None and coords is not None:
        denom_main = float(np.mean(T_resid_main ** 2))
        if denom_main > 1e-12:
            # IF score psi_i = (Y_tilde - theta * T_tilde) * T_tilde / E[T_tilde^2]
            psi = ((Y_resid_main - theta_hat * T_resid_main)
                   * T_resid_main) / denom_main
            # Conley HAC SE on the scaled scores (Var(theta) = (1/n^2) sum w_ij psi_i psi_j)
            se_hac = spatial_hac_se(psi, coords, bandwidth_quantile=bandwidth_quantile)
            jk = salerno_jackknife_hac(psi, fold_ids_main, coords,
                                       bandwidth_quantile=bandwidth_quantile)
            sweep = bandwidth_sensitivity_sweep(psi, coords, fold_ids=fold_ids_main)
            spatial_hac_block = {
                "se_conley_hac": float(se_hac),
                "se_jackknife_hac": float(jk["se"]),
                "v_jk_decomposition": {
                    "v_jk": jk["v_jk"], "v_off": jk["v_off"],
                    "v_between": jk["v_between"], "v_diag": jk["v_diag"],
                },
                "bandwidth_quantile": float(bandwidth_quantile),
                "h_n": float(jk["h_n"]),
                "bandwidth_sweep": {
                    str(q): float(s) for q, s in sweep.items()
                },
            }
            print(f"  spatial HAC inference (Conley 1999, Salerno-Wu-McCormick 2026):")
            print(f"    SE_Conley_HAC      = {se_hac:.4f}   (bandwidth_q={bandwidth_quantile})")
            print(f"    SE_Jackknife_HAC   = {jk['se']:.4f}   "
                  f"(v_off={jk['v_off']:.2e}, v_between={jk['v_between']:.2e})")
            print(f"    bandwidth sweep q in {{0.05, 0.10, 0.15, 0.20}}: "
                  f"max SE = {sweep['max']:.4f}")

    elapsed = time.time() - t0
    print(f"  cheap bootstrap (B={len(boots)}, t_{len(boots)} = {t_q:.3f}):")
    print(f"    SE_cb = {se_cb:.4f}  ({t_boot:.1f}s for B reps, {elapsed:.1f}s total)")
    print(f"    95% CI = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"    MDE = ±{mde:.4f}  ({100*(math.exp(mde)-1):.1f}% in price)")
    sig = "EXCLUDES ZERO" if (ci_low > 0 or ci_high < 0) else "contains zero"
    print(f"    {sig}")

    out_row = {
        "city": city,
        "n": int(n),
        "text_dim": int(T.shape[1]),
        "n_confounders": int(conf.shape[1]),
        "has_rich_confounders": bool(meta["has_rich_confounders"]),
        "mode": mode,
        "cross_fit_pca": cross_fit_pca,
        "drop_crime": drop_crime,
        "buffered_kfold": bool(buffered_kfold_flag),
        "B": len(boots),
        "theta": theta_hat,
        "se_if": se_if,
        "se_cb": se_cb,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "mde": mde,
        "t_pivot": float(t_q),
        "elapsed_sec": round(elapsed, 1),
    }
    if spatial_hac_block is not None:
        out_row["spatial_hac_se"] = spatial_hac_block["se_jackknife_hac"]
        out_row["spatial_hac"] = spatial_hac_block
        out_row["bandwidth_sweep"] = spatial_hac_block["bandwidth_sweep"]
        out_row["v_jk_decomposition"] = spatial_hac_block["v_jk_decomposition"]
    return out_row


def main():
    argv = sys.argv[1:]
    B = 50
    parallel = True
    mode = "efron"
    cross_fit_pca = False
    drop_crime = False
    spatial_hac = False
    buffered_kfold_flag = False
    buffer_quantile = 0.05
    bandwidth_quantile = 0.10
    out_suffix = ""
    cities = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--B":
            B = int(argv[i + 1]); i += 2
        elif a == "--serial":
            parallel = False; i += 1
        elif a == "--dirichlet":
            mode = "dirichlet"; out_suffix = "_dirichlet"; i += 1
        elif a == "--cross-fit-pca":
            cross_fit_pca = True; out_suffix += "_cfpca"; i += 1
        elif a == "--no-crime":
            drop_crime = True; out_suffix += "_nocrime"; i += 1
        elif a == "--spatial_hac":
            spatial_hac = True; out_suffix += "_hac"; i += 1
        elif a == "--buffered_kfold":
            buffered_kfold_flag = True; out_suffix += "_buf"; i += 1
        elif a == "--buffer_quantile":
            buffer_quantile = float(argv[i + 1]); i += 2
        elif a == "--bandwidth_quantile":
            bandwidth_quantile = float(argv[i + 1]); i += 2
        elif a == "--suffix":
            out_suffix = argv[i + 1]; i += 2
        elif a.startswith("--"):
            i += 1
        else:
            cities.append(a); i += 1
    if not cities:
        cities = ["sf", "nyc", "boston"]
    print(f"Cheap Bootstrap DML | cities={cities} | B={B} | parallel={parallel} "
          f"| mode={mode} | cross_fit_pca={cross_fit_pca} | "
          f"drop_crime={drop_crime} | spatial_hac={spatial_hac} | "
          f"buffered_kfold={buffered_kfold_flag}\n")

    rows = []
    for c in cities:
        r = fast_dml(c, B=B, parallel=parallel, mode=mode,
                     cross_fit_pca=cross_fit_pca, drop_crime=drop_crime,
                     spatial_hac=spatial_hac,
                     buffered_kfold_flag=buffered_kfold_flag,
                     buffer_quantile=buffer_quantile,
                     bandwidth_quantile=bandwidth_quantile)
        if r is not None:
            rows.append(r)

    print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
    print(pd.DataFrame(rows).to_string(index=False))
    out = Path(f"results/fast_bootstrap_dml{out_suffix}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
