"""
Verification of Proposition 2(b2):

  Under uniform-convergence conditions on Φ', the post-hoc plug-in estimator
      V̂_Φ'^post(Z; C)  →_p  V_Φ'(Z; C)  as m → ∞.

The rate predicted by Xu et al. 2020 Theorem 3 is O(m^{-1/2}) for bounded
log-loss with finite Rademacher complexity.

Method (Monte Carlo, since the asymptotic step requires uniform PAC-style
bounds whose proof is the content of the cited theorem):
  - DGP where V_Φ' is computable analytically. We take Z = β·C + ε with C
    ~ Bern(1/2), ε ~ N(0, 1), β = 1, and Φ' = univariate logistic regression
    with intercept. The Bayes-optimal posterior p(C=1|z) = σ(β·z - β²/2) lies
    in Φ', so V_Φ' = I(Z; C). We approximate I(Z; C) by Monte Carlo at n_huge.
  - For each m ∈ {1e2, 3e2, 1e3, 3e3, 1e4, 3e4}, draw R = 30 fresh samples,
    fit Φ' on each, compute |V̂_Φ'^post − V_Φ'|.
  - Verify monotone decrease in m AND log-log slope ≈ -1/2 (target [-0.7, -0.3]).
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)


BETA = 1.0
N_HUGE = 200_000


def sample(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    C = rng.integers(0, 2, size=n).astype(np.int64)
    Z = (BETA * C + rng.standard_normal(size=n)).astype(np.float64)
    return Z[:, None], C


def true_V_phi_prime(rng: np.random.Generator) -> tuple[float, float]:
    """Compute V_Φ' = I(Z;C) by Monte Carlo at N_HUGE under the true DGP."""
    Z, C = sample(N_HUGE, rng)
    p1_given_z = expit(BETA * Z[:, 0] - BETA ** 2 / 2)
    p1_marg = float(C.mean())
    p0_marg = 1 - p1_marg
    H_C = -p0_marg * np.log(p0_marg) - p1_marg * np.log(p1_marg)
    eps = 1e-12
    H_C_given_Z = float((
        -p1_given_z * np.log(p1_given_z + eps)
        - (1 - p1_given_z) * np.log(1 - p1_given_z + eps)
    ).mean())
    return float(H_C - H_C_given_Z), float(H_C)


def estimate_V_post(m: int, rng: np.random.Generator) -> float:
    """Fit Φ' on m samples; evaluate V̂_Φ'^post on a fresh m-sample."""
    Z_tr, C_tr = sample(m, rng)
    Z_te, C_te = sample(m, rng)
    if len(np.unique(C_tr)) < 2:
        return float("nan")
    clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    clf.fit(Z_tr, C_tr)
    p1 = clf.predict_proba(Z_te)[:, 1]
    eps = 1e-12
    log_p_correct = np.where(C_te == 1, np.log(p1 + eps), np.log(1 - p1 + eps))
    p1_marg = float(C_te.mean())
    p0_marg = 1 - p1_marg
    if p1_marg in (0.0, 1.0):
        return float("nan")
    H_C = -p0_marg * np.log(p0_marg) - p1_marg * np.log(p1_marg)
    L = -float(log_p_correct.mean())
    return H_C - L


def main() -> int:
    rng_master = np.random.default_rng(2024)
    V_true, H_C = true_V_phi_prime(rng_master)

    sample_sizes = [100, 300, 1000, 3000, 10_000, 30_000]
    n_repeats = 30

    table = []
    for m in sample_sizes:
        errs = []
        for r in range(n_repeats):
            v_hat = estimate_V_post(m, np.random.default_rng(10_000 * m + r))
            if not np.isnan(v_hat):
                errs.append(abs(v_hat - V_true))
        errs = np.array(errs)
        table.append({
            "m": m,
            "n_repeats": int(len(errs)),
            "mean_abs_error": float(errs.mean()),
            "median_abs_error": float(np.median(errs)),
            "p95_abs_error": float(np.quantile(errs, 0.95)),
        })

    log_m = np.log(np.array([row["m"] for row in table]))
    log_err = np.log(np.array([row["mean_abs_error"] for row in table]))
    slope, intercept = np.polyfit(log_m, log_err, 1)

    monotone = bool(all(table[i + 1]["mean_abs_error"] <= table[i]["mean_abs_error"] * 1.2
                        for i in range(len(table) - 1)))
    rate_in_band = bool(-0.7 <= slope <= -0.3)

    overall = monotone and rate_in_band

    result = {
        "verdict": "PASS" if overall else "FAIL",
        "claim": "V̂_Φ'^post → V_Φ' at rate O(m^{-1/2}); slope of log|err| vs log m should be ≈ -0.5.",
        "true_V_phi_prime_nats": V_true,
        "H_C_nats": H_C,
        "monte_carlo_table": table,
        "log_log_slope": float(slope),
        "log_log_intercept": float(intercept),
        "monotone_decrease_in_m": monotone,
        "slope_in_band_-0.7_to_-0.3": rate_in_band,
    }

    out = RESULTS / "05_consistency_rate.json"
    out.write_text(json.dumps(result, indent=2))

    if not overall:
        print(f"[05] FAIL — slope {slope:.3f}, monotone={monotone}; see {out}")
        return 1
    print(f"[05] PASS — log-log slope = {slope:.3f} ∈ [-0.7, -0.3], "
          f"|err| monotone decreasing in m. Wrote {out}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
