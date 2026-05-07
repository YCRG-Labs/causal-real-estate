"""Monte Carlo orchestration for the JBES-style simulation validation.

For each cell in the cross-product
    {DR, DML, Adversarial, Randomization}
  x {N=500, 2000, 10000}
  x {SCM_0, SCM_1(0.01), SCM_1(0.05), SCM_1(0.10)}
run R replicates and record bias, SD, RMSE, 95% CI coverage, mean CI length,
and (for SCM_1 cells) power = P(reject H0: theta=0 at alpha=0.05).

Truth per estimator under SCM_1 is calibrated at population scale (one big
draw) so that the bias is measured against each estimator's native estimand
rather than against the structural beta_direct (which only equals theta for
the DML pipeline). The targets are stored alongside each cell.

Smoke vs. full grid:
  default (no flags):              N=[500],         R=20    (~1 min)
  --n_reps 200:                    N=[500],         R=200
  --n_reps 1000 --full_grid:       N=[500,2000,10000], R=1000  (full JBES grid)

Outputs:
  results/simulation/coverage_table.csv
  results/simulation/power_table.csv
  results/simulation/raw_replicates.csv
  results/simulation/truths.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulation.dgp import (  # noqa: E402
    GaussianMixtureGenerator,
    calibrate_beta_direct,
    fit_generator,
    load_real_pairs,
    sample_scm0,
    sample_scm1,
)
from simulation.estimators import ESTIMATORS, EstimateResult  # noqa: E402

DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parents[3] / "results" / "simulation"
)

EFFECT_SIZES = (0.01, 0.05, 0.10)


@dataclass
class CellSpec:
    estimator: str
    N: int
    dgp: str            # "scm0" | "scm1_0.01" | "scm1_0.05" | "scm1_0.10"
    beta_direct: float  # 0.0 for SCM_0


def _cell_label(spec: CellSpec) -> str:
    return f"{spec.estimator}|N={spec.N}|{spec.dgp}"


def _draw_one(
    spec: CellSpec, gen: GaussianMixtureGenerator, seed: int, n_W: int = 5
) -> dict:
    """Draw one replicate for `spec` and run the wrapped estimator."""
    rng = np.random.default_rng(seed)
    if spec.dgp == "scm0":
        E, _, W, Y = sample_scm0(gen, None, spec.N, n_W=n_W, rng=rng)
    else:
        E, _, W, Y = sample_scm1(
            gen, None, spec.N, beta_direct=spec.beta_direct, n_W=n_W, rng=rng,
        )
    fn = ESTIMATORS[spec.estimator]
    try:
        res = fn(E, W, Y)
    except Exception as e:
        return {
            "cell": _cell_label(spec), "rep_seed": seed,
            "theta": float("nan"), "se": float("nan"),
            "ci_low": float("nan"), "ci_high": float("nan"),
            "error": str(e)[:200],
        }
    return {
        "cell": _cell_label(spec),
        "estimator": spec.estimator,
        "N": spec.N,
        "dgp": spec.dgp,
        "beta_direct": spec.beta_direct,
        "rep_seed": seed,
        "theta": res.theta,
        "se": res.se,
        "ci_low": res.ci_low,
        "ci_high": res.ci_high,
    }


# ---------------------------------------------------------------------------
# Truth calibration: each estimator has a different estimand.
# We pin the per-estimator "truth" to the sample-mean theta over a single
# very large draw (n_truth_pop). This is the population-scale estimand
# under the same DGP each replicate uses.
# ---------------------------------------------------------------------------

def calibrate_truths(
    gen: GaussianMixtureGenerator,
    estimators: list[str],
    dgps: list[tuple[str, float]],
    n_truth_pop: int,
    seed: int = 20260429,
    n_W: int = 5,
) -> dict[str, float]:
    """For each (estimator, dgp), one big-N draw -> theta_truth.

    For SCM_0 we *enforce* truth = 0 (this is by construction of the DGP).
    For SCM_1(eta) we run one big draw and use the estimator's theta on it.
    """
    truths: dict[str, float] = {}
    for est in estimators:
        for dgp_name, beta in dgps:
            key = f"{est}|{dgp_name}"
            if dgp_name == "scm0":
                truths[key] = 0.0
                continue
            rng = np.random.default_rng(seed)
            E, _, W, Y = sample_scm1(
                gen, None, n_truth_pop, beta_direct=beta, n_W=n_W, rng=rng,
            )
            try:
                r = ESTIMATORS[est](E, W, Y)
                truths[key] = float(r.theta)
            except Exception as e:
                print(f"  truth calibration FAILED for {key}: {e}", flush=True)
                truths[key] = float("nan")
            print(f"  truth[{key}] = {truths[key]:+.5f}", flush=True)
    return truths


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_cell(
    df: pd.DataFrame, truth: float, alpha: float = 0.05
) -> dict:
    """Compute bias / SD / RMSE / coverage / mean CI length / power."""
    th = df["theta"].to_numpy(dtype=float)
    se = df["se"].to_numpy(dtype=float)
    lo = df["ci_low"].to_numpy(dtype=float)
    hi = df["ci_high"].to_numpy(dtype=float)
    ok = np.isfinite(th) & np.isfinite(se) & np.isfinite(lo) & np.isfinite(hi)
    th, se, lo, hi = th[ok], se[ok], lo[ok], hi[ok]
    n_ok = len(th)
    if n_ok == 0:
        return {
            "n_reps_ok": 0, "bias": np.nan, "sd": np.nan, "rmse": np.nan,
            "coverage": np.nan, "mean_ci_length": np.nan, "power": np.nan,
            "truth": truth,
        }
    bias = float(th.mean() - truth)
    sd = float(th.std(ddof=1)) if n_ok > 1 else float("nan")
    rmse = float(np.sqrt(np.mean((th - truth) ** 2)))
    coverage = float(np.mean((lo <= truth) & (truth <= hi)))
    ci_len = float(np.mean(hi - lo))
    # Power = P(|theta_hat / SE| > 1.96)
    z = np.abs(np.divide(th, se, out=np.full_like(th, np.nan), where=se > 0))
    power = float(np.mean(z > 1.96))
    return {
        "n_reps_ok": n_ok,
        "bias": bias, "sd": sd, "rmse": rmse,
        "coverage": coverage, "mean_ci_length": ci_len, "power": power,
        "truth": truth,
    }


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def run(
    n_reps: int,
    Ns: list[int],
    estimators: list[str],
    out_dir: Path,
    n_jobs: int = -1,
    n_truth_pop: int = 10000,
    seed: int = 20260429,
    n_W: int = 5,
    parquet_path: Path | None = None,
    n_subsample: int | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading real (E, z) pairs...", flush=True)
    real_E, real_z = load_real_pairs(parquet_path, n_subsample=n_subsample, seed=seed)
    print(f"      n_real={len(real_z)}, dim={real_E.shape[1]}, "
          f"n_zips={len(np.unique(real_z))}", flush=True)

    print(f"[2/4] Fitting GaussianMixture generator...", flush=True)
    gen = fit_generator(real_E, real_z, low_rank=10, min_bin_n=10)

    # Calibrate beta_direct at population scale for each effect size.
    print(f"[3/4] Calibrating beta_direct for effect sizes "
          f"{EFFECT_SIZES}...", flush=True)
    betas = {}
    for eta in EFFECT_SIZES:
        betas[eta] = calibrate_beta_direct(gen, eta, n_pop=20000)
        print(f"      target eta={eta:.2f} -> beta_direct={betas[eta]:.4f}", flush=True)

    dgps: list[tuple[str, float]] = [("scm0", 0.0)]
    for eta in EFFECT_SIZES:
        dgps.append((f"scm1_{eta:.2f}", betas[eta]))

    print(f"[3a/4] Calibrating per-estimator truths (n_pop={n_truth_pop})...", flush=True)
    truths = calibrate_truths(gen, estimators, dgps, n_truth_pop=n_truth_pop, n_W=n_W)
    with open(out_dir / "truths.json", "w") as f:
        json.dump({k: float(v) for k, v in truths.items()}, f, indent=2)

    cells: list[CellSpec] = []
    for est in estimators:
        for N in Ns:
            for (dgp_name, beta) in dgps:
                cells.append(CellSpec(estimator=est, N=N, dgp=dgp_name, beta_direct=beta))

    total_jobs = len(cells) * n_reps
    print(f"[4/4] Running {len(cells)} cells x {n_reps} reps = {total_jobs} jobs",
          flush=True)
    t0 = time.time()

    rng_master = np.random.default_rng(seed)

    rows: list[dict] = []
    for ci, spec in enumerate(cells, start=1):
        seeds = rng_master.integers(0, 2**31 - 1, size=n_reps).tolist()
        cell_t0 = time.time()
        cell_rows = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(_draw_one)(spec, gen, int(s), n_W) for s in seeds
        )
        rows.extend(cell_rows)
        dt = time.time() - cell_t0
        print(f"  [{ci}/{len(cells)}] {_cell_label(spec):<35} "
              f"R={n_reps}  {dt:6.1f}s "
              f"({dt / max(n_reps,1)*1000:.0f} ms/rep)", flush=True)

    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "raw_replicates.csv", index=False)

    # Aggregate per cell.
    summary_rows = []
    for spec in cells:
        sub = raw[
            (raw["estimator"] == spec.estimator)
            & (raw["N"] == spec.N)
            & (raw["dgp"] == spec.dgp)
        ]
        truth = truths.get(f"{spec.estimator}|{spec.dgp}", 0.0)
        agg = aggregate_cell(sub, truth=truth)
        summary_rows.append({
            "estimator": spec.estimator,
            "N": spec.N,
            "dgp": spec.dgp,
            "beta_direct": spec.beta_direct,
            **agg,
        })
    summary = pd.DataFrame(summary_rows)

    # Coverage table and power table.
    coverage_cols = ["estimator", "N", "dgp", "beta_direct", "truth",
                     "n_reps_ok", "bias", "sd", "rmse",
                     "coverage", "mean_ci_length"]
    power_cols = ["estimator", "N", "dgp", "beta_direct", "truth",
                  "n_reps_ok", "power"]
    summary[coverage_cols].to_csv(out_dir / "coverage_table.csv", index=False)
    summary[power_cols].to_csv(out_dir / "power_table.csv", index=False)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min. Wrote:", flush=True)
    print(f"  {out_dir/'coverage_table.csv'}", flush=True)
    print(f"  {out_dir/'power_table.csv'}", flush=True)
    print(f"  {out_dir/'raw_replicates.csv'}", flush=True)
    print(f"  {out_dir/'truths.json'}", flush=True)

    # Quick console preview.
    print("\nCoverage table preview:")
    print(summary[coverage_cols].to_string(index=False))


def main():
    ap = argparse.ArgumentParser(
        description="JBES-style Monte Carlo validation of causal estimators."
    )
    ap.add_argument("--n_reps", type=int, default=20,
                    help="Monte Carlo replicates per cell (default 20 = smoke).")
    ap.add_argument("--full_grid", action="store_true",
                    help="Sweep N=[500,2000,10000] (otherwise N=[500] only).")
    ap.add_argument("--N", type=int, nargs="*", default=None,
                    help="Override the N grid (e.g. --N 500 2000).")
    ap.add_argument("--estimators", type=str, nargs="*",
                    default=list(ESTIMATORS.keys()),
                    help="Subset of estimators to run.")
    ap.add_argument("--n_jobs", type=int, default=-1,
                    help="joblib n_jobs (-1 = all cores).")
    ap.add_argument("--n_truth_pop", type=int, default=10000,
                    help="N for population-scale truth calibration.")
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--parquet", type=Path, default=None,
                    help="Override path to real (E,z) parquet.")
    ap.add_argument("--n_subsample", type=int, default=None,
                    help="Subsample real rows for fast generator fit.")
    ap.add_argument("--seed", type=int, default=20260429)
    args = ap.parse_args()

    if args.N is not None:
        Ns = args.N
    elif args.full_grid:
        Ns = [500, 2000, 10000]
    else:
        Ns = [500]

    bad = [e for e in args.estimators if e not in ESTIMATORS]
    if bad:
        ap.error(f"Unknown estimator(s): {bad}. Available: {list(ESTIMATORS)}")

    run(
        n_reps=args.n_reps,
        Ns=Ns,
        estimators=args.estimators,
        out_dir=args.out_dir,
        n_jobs=args.n_jobs,
        n_truth_pop=args.n_truth_pop,
        seed=args.seed,
        parquet_path=args.parquet,
        n_subsample=args.n_subsample,
    )


if __name__ == "__main__":
    main()
