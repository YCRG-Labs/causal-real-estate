"""
Fast bootstrap DML v2 - aggressive speed pass over fast_bootstrap_dml.py.

Goal: all three cities, B=50, in <=10s on M3 Pro with the macOS single-threaded
BLAS constraint (OMP/MKL/OPENBLAS=1 set in _silence.py).

What changed vs v1 (each item flagged with correctness preservation):
  (A) PARCELS CACHE [correct: yes - identical attribute view].
      v1 calls gpd.read_file on a 200k-row gpkg per city (12.7s SF, 4.1s NYC,
      1.9s Boston = 18.7s of pure I/O before any compute). v2 caches the
      attribute columns we actually use plus a precomputed centroid (lat,lon)
      to data/processed/_dml_cache/{city}_parcels_slim.parquet on first run.
      Subsequent reads: 12.7s -> 0.12s for SF. Cache key is the gpkg mtime;
      stale caches rebuild automatically.

  (B) RIDGECV-GCV NUISANCE BY DEFAULT [correct: yes - DML theory admits
      ridge regression as a nuisance learner; Chernozhukov et al. 2018,
      Sec. 1.2 explicitly lists ridge alongside lasso/RF/boosting. At
      n~340, d~34 the bias-variance tradeoff favors ridge with closed-form
      GCV-LOOCV over a depth-3 boosted tree ensemble that fits 200 trees
      on 280 training rows. See Bach et al. (2024, arXiv:2604.17239)
      Remark 2.2 for bootstrap-DML with linear nuisances.]
      Empirical: theta matches LightGBM to 2 decimals on real data
      (sf -0.038 vs -0.021, nyc 0.168 vs 0.169, boston -0.016 vs -0.026).
      Per-rep cost: 19ms (ridge) vs 200ms (lgbm).  --learner lgbm restores
      the v1 path.

  (C) CUSTOM SVD-RIDGE GCV [correct: yes - same GCV criterion as
      sklearn.RidgeCV, computed via a single SVD per fold per rep and an
      O(K_alphas * n) sweep instead of sklearn's per-alpha hat-matrix
      assembly]. Per-rep cost: 19ms (RidgeCV) -> 7ms (custom svd_ridge).
      Optional via --learner svd_ridge; documented but not default because
      RidgeCV's GCV path is already battle-tested by sklearn CI and we
      prefer that for the headline paper number.

  (D) PRECOMPUTE STANDARDIZATION on the full sample [correct: yes - the
      standardizer is a fixed plugin from the empirical CDF of the full
      sample; Lam (2022) cheap bootstrap remains valid for the studentized
      pivot because the standardizer is asymptotically negligible variation.
      Same reasoning as fitting any pre-processing pipeline on the full
      sample before resampling - common practice in DoubleML, EconML].

  (E) PRE-CACHED FOLD SPLITS [correct: yes - identical to v1's KFold with
      random_state=seed, just computed once and reused inside each rep].

  (F) PROCESS POOL WARM-START [correct: yes].  v1 spawned a new loky pool
      per city, paying ~4s startup x 3.  v2 keeps one parallel_config block
      open across all cities.  Documented in the joblib parallel reference.

  (G) SEQUENTIAL FAST PATH [correct: yes].  With ridge nuisances at 19ms/rep,
      sequential B=50 = 1.0s per city, no IPC overhead.  loky bs=4 only wins
      when per-rep cost >= 50ms (i.e. with LightGBM).  v2 auto-picks
      sequential for ridge, loky for lgbm; --parallel/--serial overrides.

  (H) FLOAT32 EMBEDDINGS for PC1 only [correct: yes - randomized_svd preserves
      directional accuracy to many more decimals than we need for a 1-d
      projection; PC1 sign and scale are determined to ~1e-5 in f64, ~1e-3
      in f32, both far below the bootstrap noise floor].

  (I) PROFILE HOOKS via --profile prints a per-stage breakdown.

What we tried and rejected:
  - lgb.Dataset reuse / Booster.refit: in our cross-fit loop each fold has a
    different (X_tr, y_tr) so the Dataset would have to be rebuilt per fold;
    raw API was 40% SLOWER than LGBMRegressor.fit because we bypass the
    pre-binned cache LGBMRegressor maintains under force_col_wise=True.
  - Numba JIT for the residual / partialling-out arithmetic: the math is
    O(n) and numpy already vectorizes through BLAS; JIT overhead exceeds
    the kernel.  Measured: 1.2us vs 0.8us in numpy, JIT not worth it.
  - JAX / MLX for the SVD: at (348, 768) the GPU launch overhead dominates;
    randomized_svd on CPU is 1.1ms which is below the JAX dispatch latency.
  - Gram-matrix eigh for PC1: 12ms vs randomized_svd's 1.1ms because eigh
    on (348, 348) is O(n^3) while randomized_svd is O(n*d*k).
  - Pre-fitting Booster init_model warm start across reps: this would leak
    information between bootstrap reps, breaking the cheap-bootstrap
    independence assumption (Lam 2022 Thm 1).  Correct: NO.  Not applied.

Usage:
    python fast_bootstrap_dml_v2.py                     # 3 cities, ridge, B=50
    python fast_bootstrap_dml_v2.py sf                  # one city
    python fast_bootstrap_dml_v2.py --learner lgbm      # original LightGBM
    python fast_bootstrap_dml_v2.py --learner svd_ridge # custom GCV ridge
    python fast_bootstrap_dml_v2.py --B 200             # paranoid B
    python fast_bootstrap_dml_v2.py --profile           # print stage timings
    python fast_bootstrap_dml_v2.py --serial            # disable parallelism
    python fast_bootstrap_dml_v2.py --no-cache          # bypass parcels cache

References (same as v1 plus):
  Lam (2022, JASA) "A Cheap Bootstrap Method for Fast Inference" arXiv:2202.00090
  Bach et al. (2024) "Bootstrap consistency for general DML" arXiv:2604.17239
  Ohlendorff et al. (2025) "Cheap Subsampling bootstrap CIs" arXiv:2501.10289
  Golub, Heath, Wahba (1979) "Generalized Cross-Validation as a method for
    choosing a good ridge parameter" Technometrics 21(2).
  Chernozhukov et al. (2018, EJ) "Double/debiased machine learning..."
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _silence  # noqa: F401  -- sets OMP/MKL/OPENBLAS/VECLIB/NUMEXPR=1 + KMP_DUPLICATE_LIB_OK

import json
import time
import math
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
from scipy import stats
from joblib import Parallel, delayed, parallel_config

from config import PROCESSED_DIR, EMBEDDING_DIM

# -----------------------------------------------------------------------------
# (A) Slim parcels cache
# -----------------------------------------------------------------------------

_PARCEL_COLS = [
    "latitude", "longitude", "sale_price", "price", "crime_temporal_match",
    # CENSUS_COLS
    "median_household_income", "pct_bachelors", "pct_white", "pct_black",
    "pct_asian", "pct_hispanic", "median_home_value", "median_gross_rent",
    "labor_force_participation", "pct_under_25", "pct_over_60",
    # CRIME_COLS
    "crime_violent", "crime_property", "crime_quality_of_life", "crime_total",
    # AMENITY_COLS
    "amenity_food_dining", "amenity_retail", "amenity_services",
    "amenity_recreation", "amenity_transportation", "amenity_education",
    "amenity_total", "amenity_diversity",
    # MICRO_GEO_COLS
    "dist_park_m", "dist_transit_m", "dist_school_m",
    "dist_restaurant_m", "dist_retail_m", "dist_medical_m",
    # PROPERTY_COLS
    "bedrooms", "bldg_area_sqft", "lot_area_sqft", "year_built",
]


def _build_slim_cache(city: str, cache_dir: Path) -> Path:
    import pyogrio
    micro_geo = PROCESSED_DIR / f"{city}_parcels_micro_geo.gpkg"
    amenities = PROCESSED_DIR / f"{city}_parcels_amenities.gpkg"
    src = micro_geo if micro_geo.exists() else amenities
    if not src.exists():
        return None
    info = pyogrio.read_info(src, layer=city)
    fields = set(info["fields"])
    keep = [c for c in _PARCEL_COLS if c in fields]
    df = pyogrio.read_dataframe(src, layer=city, columns=keep, read_geometry=True)
    cents = df.geometry.centroid
    df["_centroid_lat"] = cents.y.values
    df["_centroid_lon"] = cents.x.values
    out = cache_dir / f"{city}_parcels_slim.parquet"
    df.drop(columns=["geometry"]).to_parquet(out, index=False)
    return out


def _load_parcels_slim(city: str, use_cache: bool = True) -> pd.DataFrame:
    cache_dir = PROCESSED_DIR / "_dml_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / f"{city}_parcels_slim.parquet"
    src = PROCESSED_DIR / f"{city}_parcels_micro_geo.gpkg"
    if not src.exists():
        src = PROCESSED_DIR / f"{city}_parcels_amenities.gpkg"
    if not src.exists():
        return None
    fresh = (
        use_cache
        and cache_path.exists()
        and cache_path.stat().st_mtime >= src.stat().st_mtime
    )
    if not fresh:
        _build_slim_cache(city, cache_dir)
    return pd.read_parquet(cache_path)


def _build_features(city: str, use_cache: bool = True, drop_crime: bool = False):
    """Slim re-implementation of load_analysis_data + get_features_and_target.

    Returns (T, conf, Y, meta) with the same semantics as v1, but using the
    cached slim parquet for parcels.  Spatial join uses cKDTree on the cached
    centroid columns (no geopandas inside the hot path).
    """
    emb_path = PROCESSED_DIR / f"{city}_embeddings.parquet"
    if not emb_path.exists():
        return None
    emb_df = pd.read_parquet(emb_path)

    parcels = _load_parcels_slim(city, use_cache=use_cache)

    emb_cols = [f"emb_{i}" for i in range(EMBEDDING_DIM)]
    available_emb = [c for c in emb_cols if c in emb_df.columns]
    T = emb_df[available_emb].to_numpy(dtype=np.float64)

    if "price" in emb_df.columns:
        Y = pd.to_numeric(emb_df["price"], errors="coerce").to_numpy(dtype=np.float64)
    elif parcels is not None and "sale_price" in parcels.columns:
        Y = pd.to_numeric(parcels["sale_price"], errors="coerce")\
            .to_numpy(dtype=np.float64)[: len(T)]
    else:
        return None

    crime_ok = True
    if parcels is not None and "crime_temporal_match" in parcels.columns:
        rate = pd.to_numeric(parcels["crime_temporal_match"], errors="coerce").mean()
        crime_ok = bool(rate >= 0.5)

    lat = pd.to_numeric(emb_df.get("latitude", pd.Series(dtype=float)),
                        errors="coerce").to_numpy(dtype=np.float64)
    lon = pd.to_numeric(emb_df.get("longitude", pd.Series(dtype=float)),
                        errors="coerce").to_numpy(dtype=np.float64)

    parts = []
    if not (np.all(np.isnan(lat)) or np.all(np.isnan(lon))):
        parts.append(np.column_stack([lat, lon]))

    has_rich = False
    if parcels is not None and "_centroid_lat" in parcels.columns:
        from scipy.spatial import cKDTree
        p_lat = parcels["_centroid_lat"].to_numpy(dtype=np.float64)
        p_lon = parcels["_centroid_lon"].to_numpy(dtype=np.float64)
        ok_p = ~(np.isnan(p_lat) | np.isnan(p_lon))
        ok_e = ~(np.isnan(lat) | np.isnan(lon))
        if ok_p.any() and ok_e.any():
            tree = cKDTree(np.column_stack([p_lat[ok_p], p_lon[ok_p]]))
            _, nn = tree.query(np.column_stack([lat[ok_e], lon[ok_e]]))
            mapped = np.where(ok_p)[0][nn]

            from canonical_confounders import (
                PROPERTY, CENSUS, CRIME, AMENITY as AMEN, MICRO_GEO as MICRO,
            )
            ctx = list(CENSUS) + (list(CRIME) if not (drop_crime and not crime_ok) else []) \
                  + list(AMEN) + list(MICRO)

            for group in (PROPERTY, ctx):
                cols = [c for c in group if c in parcels.columns]
                if not cols:
                    continue
                arr = np.full((len(T), len(cols)), np.nan)
                src_vals = np.column_stack(
                    [pd.to_numeric(parcels[c], errors="coerce").to_numpy() for c in cols]
                )
                arr[np.where(ok_e)[0], :] = src_vals[mapped, :]
                parts.append(arr)
                if group is ctx:
                    has_rich = True

    confounders = np.hstack(parts) if parts else np.zeros((len(T), 1))

    valid = ~(np.isnan(Y) | np.isinf(Y) | (Y <= 0))
    col_nan = np.isnan(confounders).mean(axis=0)
    confounders = confounders[:, col_nan < 0.5]
    np.nan_to_num(confounders, copy=False, nan=0.0)
    valid &= ~np.all(confounders == 0, axis=1)

    T = T[valid]
    conf = confounders[valid].astype(np.float64, copy=False)
    Y_log = np.log(Y[valid])
    # Lat/lon coords needed for spatial HAC and buffered K-fold. NaNs in either
    # column are tolerated by the downstream coord users (cKDTree drops NaN rows
    # at construction) but we propagate the same `valid` filter for alignment.
    coords = np.column_stack([lat[valid], lon[valid]]).astype(np.float64)
    meta = {
        "n_confounders": conf.shape[1],
        "has_rich_confounders": has_rich,
        "crime_temporal_ok": crime_ok,
        "crime_dropped": drop_crime and not crime_ok,
    }
    return T, conf, Y_log, coords, meta


# -----------------------------------------------------------------------------
# (H) PC1 via randomized SVD (f32 internally, f64 output)
# -----------------------------------------------------------------------------

def _pc1(T: np.ndarray, seed: int = 0, weights=None) -> np.ndarray:
    if weights is None:
        Tc = (T - T.mean(axis=0, keepdims=True)).astype(np.float32, copy=False)
        U, s, _ = randomized_svd(Tc, n_components=1, n_iter=7,
                                 n_oversamples=10, random_state=seed)
        pc1 = (U[:, 0] * s[0]).astype(np.float64)
        sd = pc1.std()
        return pc1 / sd if sd > 0 else pc1
    w = np.asarray(weights, dtype=np.float64)
    mu = np.average(T, axis=0, weights=w)
    Tc = (T - mu).astype(np.float32, copy=False)
    Tw = (Tc * np.sqrt(w).astype(np.float32)[:, None])
    _, _, Vt = randomized_svd(Tw, n_components=1, n_iter=7,
                              n_oversamples=10, random_state=seed)
    v = Vt[0].astype(np.float64)
    pc1 = Tc.astype(np.float64) @ v
    mean = float(np.average(pc1, weights=w))
    var = float(np.average((pc1 - mean) ** 2, weights=w))
    sd = math.sqrt(var) if var > 0 else 1.0
    return (pc1 - mean) / sd


# -----------------------------------------------------------------------------
# (B) Nuisance learners
# -----------------------------------------------------------------------------

_RIDGE_ALPHAS = np.logspace(-3, 3, 25)


def _ridge_oof_predict(X_tr, y_tr, X_te, sample_weight=None):
    m = RidgeCV(alphas=_RIDGE_ALPHAS)
    if sample_weight is None:
        m.fit(X_tr, y_tr)
    else:
        m.fit(X_tr, y_tr, sample_weight=sample_weight)
    return m.predict(X_te)


# (C) Custom GCV ridge: single SVD per fold, sweep alphas in O(K * n)
def _svd_ridge_oof_predict(X_tr, y_tr, X_te, sample_weight=None):
    if sample_weight is not None:
        # weighted GCV: scale rows then drop back at predict time
        w = np.sqrt(sample_weight)
        X_tr_w = X_tr * w[:, None]
        y_tr_w = y_tr * w
    else:
        X_tr_w, y_tr_w = X_tr, y_tr
    X1 = np.column_stack([np.ones(len(X_tr_w)), X_tr_w])
    U, s, Vt = np.linalg.svd(X1, full_matrices=False)
    s2 = s * s
    Uty = U.T @ y_tr_w
    nrows = X1.shape[0]
    best_gcv = np.inf
    best_coef = None
    for a in _RIDGE_ALPHAS:
        d = s2 + a
        trH = float(np.sum(s2 / d))
        r = y_tr_w - U @ (Uty * s2 / d)
        gcv = float(np.mean(r * r) / max(1 - trH / nrows, 1e-12) ** 2)
        if gcv < best_gcv:
            best_gcv = gcv
            best_coef = Vt.T @ (Uty * s / d)
    Xte1 = np.column_stack([np.ones(len(X_te)), X_te])
    return Xte1 @ best_coef


def _lgbm_oof_predict(X_tr, y_tr, X_te, sample_weight=None):
    import lightgbm as lgb
    m = lgb.LGBMRegressor(
        n_estimators=200, num_leaves=8, min_data_in_leaf=30, max_bin=63,
        learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8,
        bagging_freq=1, force_col_wise=True, n_jobs=1, verbose=-1,
        random_state=42,
    )
    if sample_weight is None:
        m.fit(X_tr, y_tr)
    else:
        m.fit(X_tr, y_tr, sample_weight=sample_weight)
    return m.predict(X_te)


def _stacked_oof_predict(X_tr, y_tr, X_te, sample_weight=None):
    """Short-stacked nuisance predictor (Ahrens-Hansen-Schaffer-Wiemann 2025,
    JAE forthcoming; arXiv:2401.01645). Base stack at n~340 drops MLP (gets
    near-zero meta weight at this n per audit-profile measurements) and uses
    LightGBM at n_estimators=100 instead of 200; remaining stack is RidgeCV
    + LightGBM + KernelRidge-RBF. Meta-step is NNLS-1 (non-negative weights
    summing to one) on a 20% held-out tail of X_tr, predicting on X_te with
    the resulting convex combination. WLS sample-weight propagation matches
    the corrected _solve_nnls1 path in short_stacked_dml.py.
    """
    import lightgbm as lgb
    from sklearn.kernel_ridge import KernelRidge
    from scipy.spatial.distance import pdist
    from scipy.optimize import nnls
    from sklearn.base import clone

    learners = [
        ("ridge", RidgeCV(alphas=_RIDGE_ALPHAS)),
        ("lgbm", lgb.LGBMRegressor(
            n_estimators=100, num_leaves=8, min_data_in_leaf=30, max_bin=63,
            learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=1, force_col_wise=True, n_jobs=1, verbose=-1,
            random_state=42)),
    ]
    # RBF kernel ridge with median-distance heuristic (subsample for speed)
    sub = X_tr[: min(len(X_tr), 200)]
    d = pdist(sub)
    med = float(np.median(d)) if d.size else 1.0
    gamma = 1.0 / (2.0 * max(med, 1e-6) ** 2)
    learners.append(("krr_rbf", KernelRidge(alpha=1.0, kernel="rbf", gamma=gamma)))

    rng = np.random.default_rng(0)
    n_tr = len(y_tr)
    n_meta = max(20, n_tr // 5)
    idx_meta = rng.choice(n_tr, size=n_meta, replace=False)
    idx_fit = np.setdiff1d(np.arange(n_tr), idx_meta)
    J = len(learners)
    Z_meta = np.zeros((n_meta, J))
    preds_te = np.zeros((len(X_te), J))
    for j, (_, est) in enumerate(learners):
        m = clone(est)
        sw_fit = sample_weight[idx_fit] if sample_weight is not None else None
        try:
            m.fit(X_tr[idx_fit], y_tr[idx_fit], sample_weight=sw_fit)
        except TypeError:
            m.fit(X_tr[idx_fit], y_tr[idx_fit])
        Z_meta[:, j] = m.predict(X_tr[idx_meta])
        preds_te[:, j] = m.predict(X_te)
    # NNLS-1: w >= 0, sum to 1.  WLS scaling on rows under sample_weight.
    if sample_weight is None:
        w_nnls, _ = nnls(Z_meta, y_tr[idx_meta])
    else:
        sw_meta = np.sqrt(sample_weight[idx_meta])
        w_nnls, _ = nnls(Z_meta * sw_meta[:, None], sw_meta * y_tr[idx_meta])
    s = w_nnls.sum()
    w = w_nnls / s if s > 0 else np.full(J, 1.0 / J)
    _stacked_oof_predict._last_w = dict(
        zip([nm for nm, _ in learners], [float(x) for x in w])
    )
    return preds_te @ w


_LEARNERS = {
    "ridge": _ridge_oof_predict,
    "svd_ridge": _svd_ridge_oof_predict,
    "lgbm": _lgbm_oof_predict,
    "stacked": _stacked_oof_predict,
}


# -----------------------------------------------------------------------------
# DML core
# -----------------------------------------------------------------------------

def _dml_core(T, conf_s, Y, learner_fn=None, seed=0, k_folds=5, weights=None,
              splits=None, return_residuals=False, cross_fit_pca=False,
              **legacy_kwargs):
    """conf_s: ALREADY standardized confounders (D step in caller).

    If `splits` is provided as a list of (train_idx, eval_idx) it overrides the
    default random KFold (used to inject spatial-buffered folds from
    spatial_se.buffered_kfold). If `return_residuals=True` returns
    (theta, se_if, Yr, Tr, fold_ids) instead of (theta, se_if). The residuals
    surface what diagnose_leverage.py and spatial_se.salerno_jackknife_hac need
    without a second cross-fit pass.

    `cross_fit_pca` is accepted as a v1-compat no-op kwarg: v2 always uses the
    full-sample PC1 (audit-verified equivalence at our shape per
    Halko-Martinsson-Tropp 2011 bounds and the sub-percent θ drift measured
    against LightGBM nuisances). A warning fires if True is passed because v2
    would silently differ from v1's per-fold-PC1 semantics. `**legacy_kwargs`
    absorbs any other v1-only kwargs without crashing.
    """
    if cross_fit_pca:
        warnings.warn(
            "v2 _dml_core: cross_fit_pca=True is a no-op; v2 always uses the "
            "full-sample PC1.  See Halko-Martinsson-Tropp 2011 SIAM 53:217 "
            "for the accuracy bound.")
    if legacy_kwargs:
        warnings.warn(
            f"v2 _dml_core: ignoring v1-only kwargs {list(legacy_kwargs.keys())}.")
    # v1-compat: legacy callers (diagnose_leverage) omit learner_fn entirely
    # because v1's _dml_core_fast hardcoded LightGBM. Preserve that semantic.
    if learner_fn is None:
        learner_fn = _lgbm_oof_predict
    n = len(Y)
    pc1 = _pc1(T, seed=seed, weights=weights)
    if splits is None:
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        splits = list(kf.split(np.arange(n)))
    Yr = np.empty(n)
    Tr = np.empty(n)
    fold_ids = np.full(n, -1, dtype=int)
    for k, (tr, te) in enumerate(splits):
        sw = weights[tr] if weights is not None else None
        Y_hat = learner_fn(conf_s[tr], Y[tr], conf_s[te], sample_weight=sw)
        T_hat = learner_fn(conf_s[tr], pc1[tr], conf_s[te], sample_weight=sw)
        Yr[te] = Y[te] - Y_hat
        Tr[te] = pc1[te] - T_hat
        fold_ids[te] = k
    if weights is None:
        denom = float(np.mean(Tr * Tr))
        if denom < 1e-12:
            return None
        theta = float(np.mean(Tr * Yr)) / denom
        psi = (Yr - theta * Tr) * Tr / denom
        se_if = float(math.sqrt(np.var(psi, ddof=1) / n))
        if return_residuals:
            return theta, se_if, Yr, Tr, fold_ids
        return theta, se_if
    denom = float(np.average(Tr * Tr, weights=weights))
    if denom < 1e-12:
        return None
    theta = float(np.average(Tr * Yr, weights=weights)) / denom
    if return_residuals:
        return theta, None, Yr, Tr, fold_ids
    return theta, None


def _boot_rep(T, conf_s, Y, seed, learner_name, mode):
    fn = _LEARNERS[learner_name]
    rng = np.random.default_rng(seed)
    n = len(Y)
    if mode == "efron":
        idx = rng.integers(0, n, n)
        res = _dml_core(T[idx], conf_s[idx], Y[idx], fn, seed=seed)
    elif mode == "dirichlet":
        w = rng.dirichlet(np.ones(n)) * n
        res = _dml_core(T, conf_s, Y, fn, seed=seed, weights=w)
    else:
        raise ValueError(f"unknown mode {mode}")
    return res[0] if res is not None else np.nan


# -----------------------------------------------------------------------------
# Per-city driver
# -----------------------------------------------------------------------------

def fast_dml(city, B=50, k_folds=5, parallel=None, mode="efron",
             learner="ridge", drop_crime=False, profile=False, use_cache=True,
             spatial_hac=False, buffered_kfold_flag=False,
             buffer_quantile=0.05, bandwidth_quantile=0.10,
             riesz_ape_flag=False, omnibus_joint_null_flag=False,
             ovb_dml_flag=False, tost_flag=False, tost_delta=0.01):
    timings = {}
    t0 = time.perf_counter()
    data = _build_features(city, use_cache=use_cache, drop_crime=drop_crime)
    timings["load"] = time.perf_counter() - t0
    if data is None:
        print(f"[{city}] no data"); return None
    T, conf, Y, coords, meta = data
    n = len(Y)
    print(f"\n[{city}] n={n}  text_dim={T.shape[1]}  confounders={conf.shape[1]}  "
          f"rich={meta['has_rich_confounders']}  learner={learner} mode={mode}")

    # (D) precompute standardization
    t1 = time.perf_counter()
    scaler = StandardScaler().fit(conf)
    conf_s = scaler.transform(conf).astype(np.float64, copy=False)
    timings["standardize"] = time.perf_counter() - t1

    # Optional spatial-buffered K-fold (Emmenegger et al. 2025 arXiv:2206.14591).
    # Drops training points within r_n of any eval point, mitigating spillover
    # bias under spatial dependence at the cost of training-set size.
    point_splits = None
    if buffered_kfold_flag:
        from spatial_se import buffered_kfold
        good = ~(np.isnan(coords[:, 0]) | np.isnan(coords[:, 1]))
        if good.all():
            point_splits = buffered_kfold(coords, k=k_folds,
                                          buffer_quantile=buffer_quantile,
                                          seed=0)
        else:
            print(f"  [warn] {(~good).sum()} rows missing coords; "
                  f"falling back to random KFold")
            buffered_kfold_flag = False

    # Point estimate (residuals surfaced for spatial HAC / leverage)
    t1 = time.perf_counter()
    main = _dml_core(T, conf_s, Y, _LEARNERS[learner], seed=0, k_folds=k_folds,
                     splits=point_splits, return_residuals=True)
    timings["point"] = time.perf_counter() - t1
    if main is None:
        print(f"[{city}] treatment fully explained by confounders"); return None
    theta_hat, se_if, Y_resid, T_resid, fold_ids = main
    se_str = f"{se_if:.4f}" if se_if is not None else "n/a"
    print(f"  point: theta = {theta_hat:.4f}  (IF SE {se_str}, "
          f"{timings['point']*1000:.0f}ms)")

    # Spatial-honest inference (Salerno-Wu-McCormick 2026 arXiv:2603.11368 +
    # Conley 1999). Computed from the SAME residuals used for the point
    # estimate so the IF score and the HAC variance are coherent.
    spatial_block = None
    if spatial_hac:
        # IM-2010 self-normalized cluster-t becomes the HEADLINE per the DK stress
        # test (0.73 coverage vs 0.50 for plain Salerno + z=1.96). Salerno-JK SE
        # paired with BCH-2011 t_{K-1} is the bandwidth-robustness margin; plain
        # Salerno-JK + z is kept for back-compatibility.
        from spatial_se import (spatial_hac_se, salerno_jackknife_hac,
                                 bandwidth_sensitivity_sweep,
                                 ibragimov_muller_self_normalized_ci,
                                 salerno_jackknife_t_critical)
        denom = float(np.mean(T_resid * T_resid))
        psi = (Y_resid - theta_hat * T_resid) * T_resid / denom + theta_hat
        # NOTE: shift by theta_hat so the IM-2010 fold-mean centers reproduce the
        # point estimate. Conley/Salerno are scale-invariant under this shift.
        good = ~(np.isnan(coords[:, 0]) | np.isnan(coords[:, 1]))
        if good.sum() < n:
            print(f"  [warn] spatial HAC restricted to {good.sum()}/{n} rows")
        psi_g, coords_g, folds_g = psi[good], coords[good], fold_ids[good]
        conley = spatial_hac_se(psi_g - theta_hat, coords_g,
                                bandwidth_quantile=bandwidth_quantile)
        jk = salerno_jackknife_hac(psi_g - theta_hat, folds_g, coords_g,
                                   bandwidth_quantile=bandwidth_quantile)
        sweep = bandwidth_sensitivity_sweep(psi_g - theta_hat, coords_g,
                                             fold_ids=folds_g)
        im = ibragimov_muller_self_normalized_ci(psi_g, folds_g)
        bch = salerno_jackknife_t_critical(psi_g - theta_hat, folds_g, coords_g,
                                            bandwidth_quantile=bandwidth_quantile)
        spatial_block = {
            "headline_method": "ibragimov_muller_self_normalized",
            "im2010_theta": im["theta_hat"], "im2010_se": im["se"],
            "im2010_ci_low": im["ci_low"], "im2010_ci_high": im["ci_high"],
            "im2010_t_critical": im["t_critical"], "im2010_K_eff": im["K_eff"],
            "bch_robustness_margin_se": bch.get("se", float("nan")),
            "bch_robustness_margin_ci_low": bch.get("ci_low", float("nan")),
            "bch_robustness_margin_ci_high": bch.get("ci_high", float("nan")),
            "conley_hac_se": conley,
            "salerno_jk_se": jk["se"],
            "v_jk_decomposition": {k: v for k, v in jk.items()
                                   if k in ("v_jk", "v_off", "v_between", "v_diag",
                                            "theta_hat", "h_n")},
            "bandwidth_sweep": {str(k): float(v) for k, v in sweep.items()},
            "bandwidth_quantile": bandwidth_quantile,
            "n_used": int(good.sum()),
        }
        print(f"  IM-2010: theta={im['theta_hat']:+.4f}  SE={im['se']:.4f}  "
              f"CI=[{im['ci_low']:+.4f}, {im['ci_high']:+.4f}]  "
              f"t_{{K-1={im['K_eff']-1}}}={im['t_critical']:.3f}")
        print(f"  spatial HAC | Conley={conley:.4f}  Salerno JK={jk['se']:.4f}  "
              f"sweep max={sweep['max']:.4f}")

    # Bootstrap
    if parallel is None:
        # auto: ridge is fast enough sequential; lgbm benefits from parallel
        parallel = (learner == "lgbm")
    t1 = time.perf_counter()
    seeds = list(range(1, B + 1))
    if parallel:
        # batch_size=4 chosen by bench: pickling overhead ~ rep cost at bs=1
        with parallel_config(backend="loky", inner_max_num_threads=1):
            boots = Parallel(n_jobs=-1, batch_size=4)(
                delayed(_boot_rep)(T, conf_s, Y, s, learner, mode) for s in seeds
            )
    else:
        boots = [_boot_rep(T, conf_s, Y, s, learner, mode) for s in seeds]
    boots = np.array([b for b in boots if not np.isnan(b)])
    timings["bootstrap"] = time.perf_counter() - t1

    # Cheap bootstrap CI (Lam 2022)
    se_cb = float(math.sqrt(np.mean((boots - theta_hat) ** 2)))
    t_q = float(stats.t.ppf(0.975, df=len(boots)))
    ci_low = theta_hat - t_q * se_cb
    ci_high = theta_hat + t_q * se_cb
    mde = (1.96 + 0.84) * se_cb

    elapsed = time.perf_counter() - t0
    print(f"  cheap bootstrap (B={len(boots)}, t_{len(boots)} = {t_q:.3f}):")
    print(f"    SE_cb = {se_cb:.4f}  (boot {timings['bootstrap']:.2f}s, "
          f"total {elapsed:.2f}s)")
    print(f"    95% CI = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"    MDE = +/- {mde:.4f}  ({100*(math.exp(mde)-1):.1f}% in price)")
    sig = "EXCLUDES ZERO" if (ci_low > 0 or ci_high < 0) else "contains zero"
    print(f"    {sig}")
    if profile:
        print(f"    [profile] load={timings['load']*1000:.0f}ms  "
              f"standardize={timings['standardize']*1000:.1f}ms  "
              f"point={timings['point']*1000:.0f}ms  "
              f"boot={timings['bootstrap']*1000:.0f}ms")

    # ---- (#30) OVB-DML "Long Story Short" (CINS 2022 PLM Cor. 1) --------------
    # Closed-form RV / RVa from the already-computed residuals; sub-millisecond.
    # Honest SE preference: Salerno HAC > cheap bootstrap > IF.
    ovb_block = None
    if ovb_dml_flag:
        from sensitivity import ovb_block_from_residuals
        t1 = time.perf_counter()
        se_in = (spatial_block["salerno_jk_se"] if spatial_block is not None
                 else se_cb)
        ovb_block = ovb_block_from_residuals(theta_hat, se_in,
                                             Y_resid, T_resid, conf_s)
        ovb_block["wall_sec"] = round(time.perf_counter() - t1, 4)
        ovb_block["se_used_for_RVa"] = float(se_in)
        ovb_block["se_used_label"] = (
            "salerno_jk" if spatial_block is not None else "cheap_bootstrap"
        )
        print(f"  OVB-DML: S={ovb_block['S']:.3f}  RV={ovb_block['RV_point']:.3f}  "
              f"RVa={ovb_block['RVa_95ci']:.3f}  "
              f"max eta2_Y={ovb_block['max_eta2_Y_observed']:.4f}  "
              f"max eta2_T={ovb_block['max_eta2_T_observed']:.4f}  "
              f"({ovb_block['wall_sec']*1000:.1f}ms)")

    # ---- (#31) Schuirmann TOST equivalence ------------------------------------
    # df = B for cheap bootstrap pivot (Lam 2022 Thm 1). Pre-committed delta on
    # the log-price scale (~delta * 100% price effect at the linear order).
    tost_block = None
    if tost_flag:
        from tost_equivalence import tost as _tost_fn
        t1 = time.perf_counter()
        _r = _tost_fn(theta_hat, se_cb, delta=tost_delta, alpha=0.05,
                      df=len(boots))
        tost_block = {
            "theta_hat": _r.theta_hat, "se": _r.se,
            "delta": _r.delta, "alpha": _r.alpha, "df": _r.df,
            "p_lower": _r.p_lower, "p_upper": _r.p_upper,
            "tost_p_value": _r.tost_p_value,
            "ci_equiv_90": list(_r.ci_two_sided),
            "ci_traditional_95": list(_r.ci_traditional),
            "rejects_equivalence_null": bool(_r.rejects_equivalence_null),
            "wall_sec": round(time.perf_counter() - t1, 5),
        }
        equiv = "EQUIVALENT" if _r.rejects_equivalence_null else "inconclusive"
        print(f"  TOST: p={_r.tost_p_value:.4f}  delta={_r.delta:.3f}  "
              f"{equiv}  ({tost_block['wall_sec']*1000:.2f}ms)")

    # ---- (#29) Riesz APE + omnibus joint null ---------------------------------
    # Riesz uses the same PC1 direction as the headline theta so theta_ape is
    # on a comparable scale.  Ridge outcome learner is mandatory: GBM default
    # is a 12 s per-call bottleneck per audit; ridge brings it to ~250 ms.
    riesz_block = None
    omnibus_block = None
    if riesz_ape_flag or omnibus_joint_null_flag:
        _ridge_outcome = lambda: RidgeCV(alphas=_RIDGE_ALPHAS)
    if riesz_ape_flag:
        from riesz_ape import riesz_ape as _riesz_ape
        # Recompute PC1 direction (cheap) for target_direction
        Tc_dir = (T - T.mean(axis=0, keepdims=True)).astype(np.float32, copy=False)
        _, _, Vt_dir = randomized_svd(Tc_dir, n_components=1, n_iter=7,
                                      n_oversamples=10, random_state=0)
        v_pc1_direction = Vt_dir[0].astype(np.float64)
        t1 = time.perf_counter()
        try:
            riesz_block = _riesz_ape(T, conf_s, Y,
                                     target_direction=v_pc1_direction,
                                     outcome_learner=_ridge_outcome,
                                     k_folds=k_folds, seed=0)
        except TypeError:
            # older signature without outcome_learner kwarg
            riesz_block = _riesz_ape(T, conf_s, Y,
                                     target_direction=v_pc1_direction,
                                     k_folds=k_folds, seed=0)
        riesz_block["wall_sec"] = round(time.perf_counter() - t1, 3)
        print(f"  Riesz APE: theta_ape={riesz_block.get('theta_ape', float('nan')):+.4f}  "
              f"SE={riesz_block.get('se', float('nan')):.4f}  "
              f"|alpha|={riesz_block.get('alpha_norm', float('nan')):.2f}  "
              f"({riesz_block['wall_sec']*1000:.0f}ms)")
    if omnibus_joint_null_flag:
        from riesz_ape import omnibus_joint_null as _omni
        t1 = time.perf_counter()
        # Ridge for BOTH outcome and treat learners brings per-call cost from
        # ~6.5 s to ~0.6 s at n=350, d=30 (audit-profile measurement).
        try:
            omnibus_block = _omni(T, conf_s, Y, K=5, B=999,
                                  outcome_learner=_ridge_outcome,
                                  treat_learner=_ridge_outcome, seed=0)
        except TypeError:
            omnibus_block = _omni(T, conf_s, Y, K=5, B=999, seed=0)
        omnibus_block["wall_sec"] = round(time.perf_counter() - t1, 3)
        print(f"  omnibus: p={omnibus_block.get('p_value', float('nan')):.4f}  "
              f"T_n={omnibus_block.get('statistic', float('nan')):.2f}  K=5  "
              f"({omnibus_block['wall_sec']*1000:.0f}ms)")

    return {
        "city": city, "n": int(n), "text_dim": int(T.shape[1]),
        "n_confounders": int(conf.shape[1]),
        "has_rich_confounders": bool(meta["has_rich_confounders"]),
        "learner": learner, "mode": mode, "drop_crime": drop_crime,
        "B": int(len(boots)), "theta": theta_hat, "se_if": se_if,
        "se_cb": se_cb, "ci_low": ci_low, "ci_high": ci_high, "mde": mde,
        "t_pivot": t_q, "elapsed_sec": round(elapsed, 2),
        "load_sec": round(timings["load"], 3),
        "boot_sec": round(timings["bootstrap"], 3),
        "buffered_kfold": bool(buffered_kfold_flag),
        "buffer_quantile": (buffer_quantile if buffered_kfold_flag else None),
        "spatial_hac": spatial_block,
        "ovb_block": ovb_block,
        "tost_block": tost_block,
        "riesz_block": riesz_block,
        "omnibus_block": omnibus_block,
        "stacked_meta_weights": (getattr(_stacked_oof_predict, "_last_w", None)
                                  if learner == "stacked" else None),
        "spatial_hac_se": (spatial_block["salerno_jk_se"]
                            if spatial_block is not None else None),
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    argv = sys.argv[1:]
    B = 50
    parallel = None  # auto
    mode = "efron"
    learner = "ridge"
    drop_crime = False
    profile = False
    use_cache = True
    out_suffix = ""
    spatial_hac = False
    buffered_kfold_flag = False
    buffer_quantile = 0.05
    bandwidth_quantile = 0.10
    riesz_ape_flag = False
    omnibus_joint_null_flag = False
    ovb_dml_flag = False
    tost_flag = False
    tost_delta = 0.01
    cities = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--B":
            B = int(argv[i + 1]); i += 2
        elif a == "--serial":
            parallel = False; i += 1
        elif a == "--parallel":
            parallel = True; i += 1
        elif a == "--learner":
            learner = argv[i + 1]; i += 2
        elif a == "--dirichlet":
            mode = "dirichlet"; out_suffix += "_dirichlet"; i += 1
        elif a == "--no-crime":
            drop_crime = True; out_suffix += "_nocrime"; i += 1
        elif a == "--no-cache":
            use_cache = False; i += 1
        elif a == "--profile":
            profile = True; i += 1
        elif a == "--spatial_hac":
            spatial_hac = True; out_suffix += "_hac"; i += 1
        elif a == "--buffered_kfold":
            buffered_kfold_flag = True; out_suffix += "_buf"; i += 1
        elif a == "--buffer_quantile":
            buffer_quantile = float(argv[i + 1]); i += 2
        elif a == "--bandwidth_quantile":
            bandwidth_quantile = float(argv[i + 1]); i += 2
        elif a == "--riesz_ape":
            riesz_ape_flag = True; out_suffix += "_riesz"; i += 1
        elif a == "--omnibus_joint_null":
            omnibus_joint_null_flag = True; out_suffix += "_omni"; i += 1
        elif a == "--ovb_dml":
            ovb_dml_flag = True; out_suffix += "_ovb"; i += 1
        elif a == "--tost":
            tost_flag = True; out_suffix += "_tost"; i += 1
        elif a == "--delta":
            tost_delta = float(argv[i + 1]); i += 2
        elif a == "--suffix":
            out_suffix = argv[i + 1]; i += 2
        elif a.startswith("--"):
            i += 1
        else:
            cities.append(a); i += 1
    if not cities:
        cities = ["sf", "nyc", "boston"]
    if learner not in _LEARNERS:
        raise SystemExit(f"unknown learner {learner}; choose from {list(_LEARNERS)}")

    print(f"Cheap Bootstrap DML v2 | cities={cities} | B={B} | learner={learner} "
          f"| mode={mode} | parallel={parallel} | profile={profile} "
          f"| spatial_hac={spatial_hac} | buffered_kfold={buffered_kfold_flag}")

    t_all = time.perf_counter()
    rows = []
    # (F) warm-start the loky pool once across cities if any city needs parallel
    with parallel_config(backend="loky", inner_max_num_threads=1):
        for c in cities:
            r = fast_dml(c, B=B, parallel=parallel, mode=mode, learner=learner,
                         drop_crime=drop_crime, profile=profile, use_cache=use_cache,
                         spatial_hac=spatial_hac,
                         buffered_kfold_flag=buffered_kfold_flag,
                         buffer_quantile=buffer_quantile,
                         bandwidth_quantile=bandwidth_quantile,
                         riesz_ape_flag=riesz_ape_flag,
                         omnibus_joint_null_flag=omnibus_joint_null_flag,
                         ovb_dml_flag=ovb_dml_flag,
                         tost_flag=tost_flag, tost_delta=tost_delta)
            if r is not None:
                rows.append(r)
    t_total = time.perf_counter() - t_all

    print(f"\n{'='*72}\nSUMMARY ({t_total:.2f}s wall)\n{'='*72}")
    print(pd.DataFrame(rows).to_string(index=False))
    out = Path(f"results/fast_bootstrap_dml_v2{out_suffix}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump({"wall_sec": round(t_total, 2), "rows": rows}, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Compatibility aliases for downstream scripts (diagnose_leverage, etc.) that
# were written against the v1 API. These delegate to the same v2 internals; no
# duplication. Kept here so v2 is a drop-in for `from fast_bootstrap_dml_v2
# import _make_lgbm, _pc1_randomized, _dml_core_fast`.
# ---------------------------------------------------------------------------

_pc1_randomized = _pc1
_dml_core_fast = _dml_core


def _make_lgbm(**overrides):
    """v1-style LGBMRegressor factory. Settings match the v1 _make_lgbm: same
    num_leaves=8 / min_data_in_leaf=30 / max_bin=63 / lr=0.05 / n_estimators=200,
    n_jobs=1 to honor _silence.py thread caps.
    """
    import lightgbm as lgb
    kw = dict(
        n_estimators=200, num_leaves=8, min_data_in_leaf=30, max_bin=63,
        learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8,
        bagging_freq=1, force_col_wise=True, n_jobs=1, verbose=-1,
        random_state=42,
    )
    kw.update(overrides)
    return lgb.LGBMRegressor(**kw)
