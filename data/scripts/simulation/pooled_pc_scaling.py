"""Prop 3 deep test: the n_m/n_pool rate law + Murphy-Topel term separation +
bootstrap recovery to nominal.

Reuses the validated helpers in pooled_pc_coverage. Arms on one DGP, one learner:

  insample        : PC re-estimated on the market's own N rows every replicate.
                    Carries the same-sample overfitting correlation (MT C-term)
                    AND the direction variance V1 = O(1/N).
  pooled:mult      : PC fixed on an INDEPENDENT draw of size n_pool = mult*N.
                    No same-sample C-term; V1 = O(1/(mult*N)).
  oracle           : generator's true direction. V_gen = 0. Residual = baseline
                    DML-with-ML-nuisances undercoverage.
  boot:mult        : pooled:mult direction, but percentile bootstrap over the
                    market sample (refit nuisances each resample) instead of IF-SE.

Decomposition delivered:
  excess(oracle)                       = baseline DML undercoverage
  excess(pooled:mult) - excess(oracle) = V1 component, predicted ~ 1/mult
  excess(insample) - excess(pooled:1)  = overfitting C-term
  coverage(boot:12)                    -> should approach 0.95 (two-ingredient fix)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("BOOSTER_NJOBS", "1")

import numpy as np
from joblib import Parallel, delayed, parallel_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulation.pooled_pc_coverage import (
    _dml_core_fixed,
    _draw,
    _pc1,
    _treatment,
)
from simulation.dgp import calibrate_beta_direct, fit_generator, load_real_pairs

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results" / "simulation_pooled"


def _dir_for(mode, E, ref, v_by_mult):
    if mode == "insample":
        return _pc1(E, ref)
    if mode == "oracle":
        return ref
    mult = int(mode.split(":")[1])
    return v_by_mult[mult]


def _if_rep(gen, N, beta, dgp, mode, ref, v_by_mult, seed):
    E, W, Y = _draw(gen, N, beta, dgp, seed)
    v = _dir_for(mode, E, ref, v_by_mult)
    out = _dml_core_fixed(_treatment(E, v), W, Y)
    if out is None:
        return None
    theta, se = out
    return theta, se, theta - 1.96 * se, theta + 1.96 * se


def _boot_rep(gen, N, beta, dgp, v, seed, B=120):
    """Percentile bootstrap over the market sample with a FIXED direction v."""
    E, W, Y = _draw(gen, N, beta, dgp, seed)
    t = _treatment(E, v)
    base = _dml_core_fixed(t, W, Y)
    if base is None:
        return None
    theta_hat = base[0]
    rng = np.random.default_rng(seed + 1)
    boots = []
    for _ in range(B):
        idx = rng.integers(0, N, N)
        out = _dml_core_fixed(_treatment(E[idx], v), W[idx], Y[idx])
        if out is not None:
            boots.append(out[0])
    if len(boots) < 20:
        return None
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return theta_hat, float(np.std(boots, ddof=1)), float(lo), float(hi)


def _truth_draw(gen, beta, dgp, mode, ref, v_by_mult, n_truth, seed):
    E, W, Y = _draw(gen, n_truth, beta, dgp, seed)
    v = _dir_for(mode, E, ref, v_by_mult)
    out = _dml_core_fixed(_treatment(E, v), W, Y)
    return None if out is None else out[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_reps", type=int, default=800)
    ap.add_argument("--boot_reps", type=int, default=250)
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--effects", type=float, nargs="*", default=[0.10, 0.15, 0.20])
    ap.add_argument("--mults", type=int, nargs="*", default=[1, 2, 4, 8, 16])
    ap.add_argument("--boot_mult", type=int, default=12)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=20260712)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    N = args.N
    print(f"[1/5] generator...", flush=True)
    real_E, real_z = load_real_pairs()
    gen = fit_generator(real_E, real_z, low_rank=10, min_bin_n=10)
    ref = gen.pc1_direction.copy()

    print(f"[2/5] beta_direct...", flush=True)
    betas = {e: calibrate_beta_direct(gen, e, n_pop=20000) for e in args.effects}

    print(f"[3/5] fixed pooled directions across mults...", flush=True)
    all_mults = sorted(set(args.mults) | {args.boot_mult})
    v_by_mult = {}
    for m in all_mults:
        Ep, _, _ = _draw(gen, m * N, 0.0, "scm0", args.seed + 7 * m)
        v_by_mult[m] = _pc1(Ep, ref)
        print(f"      mult={m:>2} n_pool={m*N:>6} |cos(v,oracle)|={abs(np.dot(v_by_mult[m], ref)):.4f}", flush=True)

    if_modes = ["insample"] + [f"pooled:{m}" for m in args.mults] + ["oracle"]
    truth_modes = if_modes + [f"pooled:{args.boot_mult}"]
    dgps = [(f"scm1_{e:.2f}", betas[e], e) for e in args.effects]

    print(f"[4/5] IF-SE truths + reps ({args.n_reps}/cell)...", flush=True)
    # truths
    n_truth, n_draws = 8000, 4
    tjobs = [(mode, dgp, beta, e, 99 + 1000 * d)
             for dgp, beta, e in dgps for mode in truth_modes for d in range(n_draws)]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        tvals = Parallel(n_jobs=args.n_jobs)(
            delayed(_truth_draw)(gen, beta, dgp, mode, ref, v_by_mult, n_truth, s)
            for (mode, dgp, beta, e, s) in tjobs)
    acc = defaultdict(list)
    for (mode, dgp, beta, e, s), tv in zip(tjobs, tvals):
        if tv is not None:
            acc[f"{mode}|{dgp}"].append(tv)
    truths = {k: float(np.mean(v)) for k, v in acc.items()}

    rows = []
    rng_master = np.random.default_rng(args.seed)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        with Parallel(n_jobs=args.n_jobs) as par:
            for dgp, beta, e in dgps:
                for mode in if_modes:
                    seeds = rng_master.integers(0, 2**31 - 1, size=args.n_reps).tolist()
                    res = [r for r in par(
                        delayed(_if_rep)(gen, N, beta, dgp, mode, ref, v_by_mult, int(s))
                        for s in seeds) if r is not None]
                    arr = np.array(res)
                    th, se, lo, hi = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
                    truth = truths[f"{mode}|{dgp}"]
                    cov = float(np.mean((lo <= truth) & (truth <= hi)))
                    sd, avg_se = float(th.std(ddof=1)), float(se.mean())
                    rows.append({"arm": mode, "N": N, "theta_true": e, "ci": "if",
                                 "truth": truth, "n_ok": len(res),
                                 "bias": float(th.mean() - truth), "sd": sd,
                                 "avg_se": avg_se, "sd_over_se": sd / avg_se,
                                 "excess_var": (sd / avg_se) ** 2 - 1, "coverage": cov})
                    print(f"  {mode:11} theta={e:.2f} cov={cov:.3f} excess={((sd/avg_se)**2-1):+.3f}", flush=True)

    def _flush_csv(rws):
        with open(RESULTS_DIR / "scaling.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rws[0].keys()))
            w.writeheader(); w.writerows(rws)
    _flush_csv(rows)
    print(f"  [saved IF sweep to scaling.csv]", flush=True)

    print(f"[5/5] bootstrap arm (pooled:{args.boot_mult} + insample), {args.boot_reps} reps...", flush=True)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        with Parallel(n_jobs=args.n_jobs) as par:
            for dgp, beta, e in dgps:
                for label, v in [(f"boot:{args.boot_mult}", v_by_mult[args.boot_mult])]:
                    seeds = rng_master.integers(0, 2**31 - 1, size=args.boot_reps).tolist()
                    res = [r for r in par(
                        delayed(_boot_rep)(gen, N, beta, dgp, v, int(s)) for s in seeds)
                        if r is not None]
                    arr = np.array(res)
                    th, se, lo, hi = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
                    truth = truths[f"pooled:{args.boot_mult}|{dgp}"]
                    cov = float(np.mean((lo <= truth) & (truth <= hi)))
                    rows.append({"arm": label, "N": N, "theta_true": e, "ci": "boot",
                                 "truth": truth, "n_ok": len(res),
                                 "bias": float(th.mean() - truth), "sd": float(th.std(ddof=1)),
                                 "avg_se": float(se.mean()),
                                 "sd_over_se": float(th.std(ddof=1)) / float(se.mean()),
                                 "excess_var": float("nan"), "coverage": cov})
                    print(f"  {label:11} theta={e:.2f} cov={cov:.3f}", flush=True)

    _flush_csv(rows)
    with open(RESULTS_DIR / "scaling.json", "w") as f:
        json.dump({"rows": rows, "mults": args.mults, "boot_mult": args.boot_mult,
                   "N": N, "n_reps": args.n_reps}, f, indent=2)
    print(f"\nwrote {RESULTS_DIR/'scaling.csv'}", flush=True)


if __name__ == "__main__":
    main()
