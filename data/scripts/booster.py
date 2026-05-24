"""Gradient-boosting factory: LightGBM when available, sklearn fallback.

LightGBM is 5-20x faster per fit than sklearn's GradientBoostingRegressor, and
uses all cores by default (sklearn GBR is single-threaded). On a 16-core Brev
box, this compounds to a 30-50x speedup on un-parallelized call sites.

To match sklearn's capacity at a given max_depth, we set num_leaves = 2**d - 1
(sklearn GBR depth-d trees max out at 2**d leaves; LightGBM with num_leaves=31
default would be deeper at d<=4). Otherwise hyperparameters pass through
unchanged.

If you're wrapping the booster inside a joblib.Parallel loop, pass n_jobs=1
to avoid double-parallelism (the outer joblib spawns workers; LightGBM inside
each worker would itself try to fan out to all cores).
"""
from __future__ import annotations

from typing import Optional

from sklearn.ensemble import GradientBoostingRegressor as _SkGBR

try:
    import lightgbm as _lgb  # type: ignore
    _HAS_LIGHTGBM = True
except Exception:
    _lgb = None  # type: ignore
    _HAS_LIGHTGBM = False


def make_regressor(
    n_estimators: int = 100,
    max_depth: Optional[int] = 3,
    learning_rate: float = 0.05,
    random_state: int = 42,
    subsample: Optional[float] = None,
    max_features: Optional[float] = None,
    n_jobs: int = -1,
    force_sklearn: bool = False,
):
    """Return a LightGBM regressor if available, else sklearn GBR fallback.

    Maps sklearn hyperparameters to LightGBM equivalents:
      - max_depth=d -> num_leaves=2**d-1, max_depth=d (matches sklearn capacity).
      - subsample passes through; subsample_freq=1 is set when subsample<1.
      - max_features (sklearn: per-split feature fraction) is mapped to
        LightGBM's feature_fraction_bynode (per-split). Approximate semantic
        match; results may drift slightly versus sklearn baseline.
      - n_jobs=-1 by default (all cores). Pass n_jobs=1 inside joblib.Parallel.
    """
    if force_sklearn or not _HAS_LIGHTGBM:
        sk_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "random_state": random_state,
        }
        if subsample is not None:
            sk_params["subsample"] = subsample
        if max_features is not None:
            sk_params["max_features"] = max_features
        return _SkGBR(**sk_params)

    if max_depth is not None and max_depth > 0:
        num_leaves = max(2, 2 ** int(max_depth) - 1)
        lgb_max_depth = int(max_depth)
    else:
        num_leaves = 31
        lgb_max_depth = -1

    lgb_params = {
        "n_estimators": n_estimators,
        "max_depth": lgb_max_depth,
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "random_state": random_state,
        "n_jobs": n_jobs,
        "verbose": -1,
    }
    if subsample is not None and subsample < 1.0:
        lgb_params["subsample"] = subsample
        lgb_params["subsample_freq"] = 1
    if max_features is not None and max_features < 1.0:
        # sklearn max_features = per-split feature fraction; LightGBM analog is
        # feature_fraction_bynode (sampled per node). Approximate match.
        lgb_params["feature_fraction_bynode"] = float(max_features)
    return _lgb.LGBMRegressor(**lgb_params)


def is_lightgbm_active() -> bool:
    """Return True iff LightGBM is the active backend (not sklearn fallback)."""
    return _HAS_LIGHTGBM
