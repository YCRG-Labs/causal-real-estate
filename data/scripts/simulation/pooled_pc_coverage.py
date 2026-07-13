"""Prop 3 confirmation: does a pooled/fixed PC direction restore IF-SE coverage.

Three arms on one DGP, identical GBR learner and 5-fold residualization, only
the treatment direction differs:

  insample : v = PC1 of the market's own E (re-estimated every replicate) --
             reproduces the generated-regressor undercoverage of the DML arm.
  pooled   : v = PC1 of a fixed pooled draw of size n_pool = 12*N, held constant
             across replicates -- the paper's actual construction.
  oracle   : v = generator's population direction (the n_pool -> inf limit).

Coverage is assessed against each arm's own population estimand theta(v), so the
comparison is apples to apples. Prop 3(b) predicts pooled/oracle recover ~0.95
at the effect sizes where insample sags to ~0.80.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("BOOSTER_NJOBS", "1")

import numpy as np
from joblib import Parallel, delayed, parallel_config
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from booster import make_regressor
from simulation.dgp import (
    calibrate_beta_direct,
    fit_generator,
    load_real_pairs,
    sample_scm0,
    sample_scm1,
)

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results" / "simulation_pooled"


def _align(v, ref):
    return -v if float(np.dot(v, ref)) < 0 else v


def _pc1(E, ref, seed=42):
    v = PCA(n_components=1, random_state=seed).fit(E).components_[0]
    return _align(v, ref)


def _treatment(E, v):
    t = E @ v
    sd = t.std()
    return (t - t.mean()) / (sd if sd > 0 else 1.0)


def _dml_core_fixed(t, W, Y, k_folds=5, seed=42):
    """theta + IF-SE for a pre-built 1-D treatment t, residualizing on W."""
    n = len(Y)
    Ws = StandardScaler().fit_transform(W)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    Yr = np.zeros(n)
    Tr = np.zeros(n)
    for tr, te in kf.split(np.arange(n)):
        my = make_regressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                            random_state=42)
        my.fit(Ws[tr], Y[tr])
        Yr[te] = Y[te] - my.predict(Ws[te])
        mt = make_regressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                            random_state=42)
        mt.fit(Ws[tr], t[tr])
        Tr[te] = t[te] - mt.predict(Ws[te])
    denom = float(np.mean(Tr ** 2))
    if denom < 1e-12:
        return None
    theta = float(np.mean(Tr * Yr)) / denom
    psi = (Yr - theta * Tr) * Tr / denom
    se = float(np.sqrt(float(np.var(psi, ddof=1)) / n))
    return theta, se


def _direction(mode, E, ref, v_pool):
    if mode == "insample":
        return _pc1(E, ref)
    if mode == "pooled":
        return v_pool
    return ref  # oracle


def _draw(gen, N, beta, dgp, seed, n_W=5):
    rng = np.random.default_rng(seed)
    if dgp == "scm0":
        E, _, W, Y = sample_scm0(gen, None, N, n_W=n_W, rng=rng)
    else:
        E, _, W, Y = sample_scm1(gen, None, N, beta_direct=beta, n_W=n_W, rng=rng)
    return E, W, Y


def _one_rep(gen, N, beta, dgp, mode, ref, v_pool, seed):
    E, W, Y = _draw(gen, N, beta, dgp, seed)
    v = _direction(mode, E, ref, v_pool)
    t = _treatment(E, v)
    out = _dml_core_fixed(t, W, Y)
    if out is None:
        return None
    theta, se = out
    return theta, se, theta - 1.96 * se, theta + 1.96 * se


def _truth_draw(gen, beta, dgp, mode, ref, v_pool, n_truth, seed):
    E, W, Y = _draw(gen, n_truth, beta, dgp, seed)
    v = _direction(mode, E, ref, v_pool)
    out = _dml_core_fixed(_treatment(E, v), W, Y)
    return None if out is None else out[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_reps", type=int, default=300)
    ap.add_argument("--N", type=int, nargs="*", default=[500, 2000])
    ap.add_argument("--effects", type=float, nargs="*", default=[0.05, 0.10, 0.15, 0.20])
    ap.add_argument("--modes", type=str, nargs="*", default=["insample", "pooled", "oracle"])
    ap.add_argument("--pool_mult", type=int, default=12)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=20260712)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/4] loading real (E,z) pairs + fitting generator...", flush=True)
    real_E, real_z = load_real_pairs()
    gen = fit_generator(real_E, real_z, low_rank=10, min_bin_n=10)
    ref = gen.pc1_direction.copy()

    print("[2/4] calibrating beta_direct...", flush=True)
    betas = {e: calibrate_beta_direct(gen, e, n_pop=20000) for e in args.effects}
    for e, b in betas.items():
        print(f"      eta={e:.2f} -> beta_direct={b:.4f}", flush=True)

    print("[3/4] estimating fixed pooled directions (n_pool = mult*N)...", flush=True)
    v_pool_by_N = {}
    for N in args.N:
        n_pool = args.pool_mult * N
        Ep, _, _ = _draw(gen, n_pool, 0.0, "scm0", args.seed + 7 * N)
        v_pool_by_N[N] = _pc1(Ep, ref)
        cos_ins = float(abs(np.dot(v_pool_by_N[N], ref)))
        print(f"      N={N}: n_pool={n_pool}, |cos(v_pool, oracle)|={cos_ins:.4f}", flush=True)

    dgps = [(f"scm1_{e:.2f}", betas[e], e) for e in args.effects]

    print("[3a/4] calibrating per-arm truths (parallel)...", flush=True)
    n_truth, n_draws = 8000, 4
    truth_jobs = []
    for N in args.N:
        for dgp, beta, e in dgps:
            for mode in args.modes:
                for d in range(n_draws):
                    truth_jobs.append((mode, N, dgp, beta, 99 + 1000 * d + 7 * N))
    with parallel_config(backend="loky", inner_max_num_threads=1):
        tvals = Parallel(n_jobs=args.n_jobs)(
            delayed(_truth_draw)(gen, beta, dgp, mode, ref, v_pool_by_N[N], n_truth, s)
            for (mode, N, dgp, beta, s) in truth_jobs
        )
    truths = {}
    from collections import defaultdict
    acc = defaultdict(list)
    for (mode, N, dgp, beta, s), tv in zip(truth_jobs, tvals):
        if tv is not None:
            acc[f"{mode}|N={N}|{dgp}"].append(tv)
    for k, vs in acc.items():
        truths[k] = float(np.mean(vs))
        print(f"      truth[{k}] = {truths[k]:+.5f}", flush=True)

    print(f"[4/4] running {args.n_reps} reps/cell...", flush=True)
    rng_master = np.random.default_rng(args.seed)
    rows = []
    with parallel_config(backend="loky", inner_max_num_threads=1):
        with Parallel(n_jobs=args.n_jobs) as par:
            for N in args.N:
                for dgp, beta, e in dgps:
                    for mode in args.modes:
                        seeds = rng_master.integers(0, 2**31 - 1, size=args.n_reps).tolist()
                        res = par(
                            delayed(_one_rep)(gen, N, beta, dgp, mode, ref,
                                              v_pool_by_N[N], int(s))
                            for s in seeds
                        )
                        res = [r for r in res if r is not None]
                        arr = np.array(res)
                        th, se, lo, hi = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
                        truth = truths[f"{mode}|N={N}|{dgp}"]
                        cov = float(np.mean((lo <= truth) & (truth <= hi)))
                        sd = float(th.std(ddof=1))
                        avg_se = float(se.mean())
                        rows.append({
                            "mode": mode, "N": N, "dgp": dgp, "theta_true": e,
                            "truth": truth, "n_ok": len(res),
                            "bias": float(th.mean() - truth), "sd": sd,
                            "avg_se": avg_se, "sd_over_se": sd / avg_se,
                            "coverage": cov,
                        })
                        print(f"  {mode:9} N={N:<5} {dgp:11} "
                              f"cov={cov:.3f} sd/se={sd/avg_se:.3f}", flush=True)

    with open(RESULTS_DIR / "pooled_coverage.json", "w") as f:
        json.dump(rows, f, indent=2)
    import csv
    with open(RESULTS_DIR / "pooled_coverage.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {RESULTS_DIR/'pooled_coverage.csv'}", flush=True)


if __name__ == "__main__":
    main()
