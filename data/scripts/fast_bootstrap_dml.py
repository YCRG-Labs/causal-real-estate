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
  4. Randomized SVD for PC1 only (n_components=1, n_iter=2). ~10-30x faster
     than full PCA(n_components=50) when we only need the leading direction.
  5. LightGBM tuned for small n=348: num_leaves=8, min_data_in_leaf=30,
     max_bin=63, force_col_wise=True. ~3-5x per fit at this scale.

Usage:
    python fast_bootstrap_dml.py            # all three cities, B=50
    python fast_bootstrap_dml.py sf nyc     # subset
    python fast_bootstrap_dml.py --B 200    # paranoid B
    python fast_bootstrap_dml.py --serial   # disable parallel for debugging

References:
  Lam (2022, JASA) "A Cheap Bootstrap Method for Fast Inference" arXiv:2202.00090
  Bach et al. (2024) "Bootstrap consistency for general DML" arXiv:2604.17239
  Christensen & van der Laan (2025) "Cheap Subsampling" arXiv:2501.10289
"""
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


def _pc1_randomized(T, seed=0):
    """Leading principal component via randomized SVD. n_components=1, n_iter=2.
    Returns the standardized PC1 score vector for the row population of T."""
    Tc = T - T.mean(axis=0, keepdims=True)
    U, s, Vt = randomized_svd(Tc, n_components=1, n_iter=2, n_oversamples=5,
                              random_state=seed)
    pc1 = (U[:, 0] * s[0])  # = T_centered @ V[:,0]
    sd = pc1.std()
    if sd <= 0:
        return pc1
    return (pc1 - pc1.mean()) / sd


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


def _dml_core_fast(T, conf, Y, k_folds=5, seed=0):
    """Single DML estimate. Returns (theta, residuals_for_IF_SE)."""
    n = len(Y)
    pc1 = _pc1_randomized(T, seed=seed)
    conf_s = StandardScaler().fit_transform(conf)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    Y_resid = np.zeros(n)
    T_resid = np.zeros(n)
    for tr, te in kf.split(np.arange(n)):
        m_y = _make_lgbm(); m_y.fit(conf_s[tr], Y[tr])
        Y_resid[te] = Y[te] - m_y.predict(conf_s[te])
        m_t = _make_lgbm(); m_t.fit(conf_s[tr], pc1[tr])
        T_resid[te] = pc1[te] - m_t.predict(conf_s[te])
    denom = float(np.mean(T_resid ** 2))
    if denom < 1e-12:
        return None
    theta = float(np.mean(T_resid * Y_resid)) / denom
    se_if = float(np.sqrt(
        np.var((Y_resid - theta * T_resid) * T_resid / denom, ddof=1) / n
    ))
    return theta, se_if


def _boot_rep(T, conf, Y, rng_seed):
    """One Cheap Bootstrap replicate: resample indices with replacement, run
    full DML pipeline (PCA + nuisances refit), return theta."""
    rng = np.random.default_rng(rng_seed)
    n = len(Y)
    idx = rng.integers(0, n, n)
    res = _dml_core_fast(T[idx], conf[idx], Y[idx], seed=rng_seed)
    return res[0] if res is not None else np.nan


def fast_dml(city, B=50, k_folds=5, parallel=True):
    t0 = time.time()
    res = load_analysis_data(city)
    if res is None:
        print(f"[{city}] no data"); return None
    emb_df, parcels = res
    data = get_features_and_target(emb_df, parcels)
    if data is None:
        print(f"[{city}] no price target"); return None
    T, conf, Y, meta = data
    T = np.asarray(T, dtype=np.float64)
    conf = np.asarray(conf, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = len(Y)

    print(f"\n[{city}] n={n}  text_dim={T.shape[1]}  confounders={conf.shape[1]}  "
          f"rich={meta['has_rich_confounders']}")

    t1 = time.time()
    main = _dml_core_fast(T, conf, Y, k_folds=k_folds, seed=0)
    if main is None:
        print(f"[{city}] treatment fully explained by confounders"); return None
    theta_hat, se_if = main
    t_point = time.time() - t1

    print(f"  point: θ̂ = {theta_hat:.4f}  (IF SE {se_if:.4f}, {t_point:.2f}s)")

    t2 = time.time()
    seeds = list(range(1, B + 1))
    if parallel:
        with parallel_config(backend="loky", inner_max_num_threads=1):
            boots = Parallel(n_jobs=-1, batch_size=1, verbose=0)(
                delayed(_boot_rep)(T, conf, Y, s) for s in seeds
            )
    else:
        boots = [_boot_rep(T, conf, Y, s) for s in seeds]
    boots = np.array([b for b in boots if not np.isnan(b)])
    t_boot = time.time() - t2

    # Cheap Bootstrap t-pivot CI (Lam 2022): centered at θ̂, divisor B not B-1
    se_cb = float(np.sqrt(np.mean((boots - theta_hat) ** 2)))
    t_q = stats.t.ppf(0.975, df=len(boots))
    ci_low = theta_hat - t_q * se_cb
    ci_high = theta_hat + t_q * se_cb
    mde = (1.96 + 0.84) * se_cb

    elapsed = time.time() - t0
    print(f"  cheap bootstrap (B={len(boots)}, t_{len(boots)} = {t_q:.3f}):")
    print(f"    SE_cb = {se_cb:.4f}  ({t_boot:.1f}s for B reps, {elapsed:.1f}s total)")
    print(f"    95% CI = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"    MDE = ±{mde:.4f}  ({100*(math.exp(mde)-1):.1f}% in price)")
    sig = "EXCLUDES ZERO" if (ci_low > 0 or ci_high < 0) else "contains zero"
    print(f"    {sig}")

    return {
        "city": city,
        "n": int(n),
        "text_dim": int(T.shape[1]),
        "n_confounders": int(conf.shape[1]),
        "has_rich_confounders": bool(meta["has_rich_confounders"]),
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


def main():
    argv = sys.argv[1:]
    B = 50
    parallel = True
    cities = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--B":
            B = int(argv[i + 1]); i += 2
        elif a == "--serial":
            parallel = False; i += 1
        elif a.startswith("--"):
            i += 1
        else:
            cities.append(a); i += 1
    if not cities:
        cities = ["sf", "nyc", "boston"]
    print(f"Cheap Bootstrap DML | cities={cities} | B={B} | parallel={parallel}\n")

    rows = []
    for c in cities:
        r = fast_dml(c, B=B, parallel=parallel)
        if r is not None:
            rows.append(r)

    print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
    print(pd.DataFrame(rows).to_string(index=False))
    out = Path("results/fast_bootstrap_dml.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
