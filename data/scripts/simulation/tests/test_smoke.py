"""Smoke test for the simulation pipeline.

Fits the GaussianMixtureGenerator on a 200-row subsample of real SF
embeddings, draws single SCM_0 / SCM_1(0.10) replicates at N=500, and
confirms the DML estimator behaves sanely:

  - SCM_0  : 95% CI for theta should contain 0
  - SCM_1  : theta_hat should be within +/- 0.05 of population-calibrated truth

Should complete in well under 60 seconds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simulation.dgp import (  # noqa: E402
    calibrate_beta_direct,
    fit_generator,
    load_real_pairs,
    sample_scm0,
    sample_scm1,
)
from simulation.estimators import (  # noqa: E402
    adversarial_estimator,
    dml_estimator,
    dr_estimator,
    randomization_estimator,
)


@pytest.fixture(scope="module")
def gen():
    real_E, real_z = load_real_pairs(n_subsample=200, seed=1)
    return fit_generator(real_E, real_z, low_rank=10, min_bin_n=10)


def test_generator_shapes(gen):
    rng = np.random.default_rng(0)
    z = gen.sample_z(50, rng)
    E = gen.sample_E(z, rng)
    assert E.shape == (50, gen.embedding_dim)
    assert np.isfinite(E).all()
    assert gen.pc1_direction.shape == (gen.embedding_dim,)


def test_dml_scm0_contains_zero(gen):
    """Under SCM_0 the DML 95% CI should contain 0."""
    rng = np.random.default_rng(7)
    E, _, W, Y = sample_scm0(gen, None, n=500, rng=rng)
    res = dml_estimator(E, W, Y)
    print(f"  SCM_0  DML  theta={res.theta:+.4f}  "
          f"CI=[{res.ci_low:+.4f}, {res.ci_high:+.4f}]")
    assert np.isfinite(res.theta) and np.isfinite(res.se)
    assert res.ci_low <= 0 <= res.ci_high, (
        f"95% CI [{res.ci_low:+.4f}, {res.ci_high:+.4f}] should contain 0 under SCM_0"
    )


def test_dml_scm1_recovers_truth(gen):
    """Under SCM_1(0.10), DML theta should be within +/- 0.05 of calibrated truth."""
    eta = 0.10
    beta = calibrate_beta_direct(gen, eta, n_pop=5000)

    # Population-scale truth on the DML estimand.
    rng_truth = np.random.default_rng(2026)
    E, _, W, Y = sample_scm1(gen, None, n=5000, beta_direct=beta, rng=rng_truth)
    truth_res = dml_estimator(E, W, Y)
    truth = truth_res.theta
    print(f"  SCM_1(0.10) calibrated beta={beta:.4f}, truth_theta={truth:+.4f}")

    # Replicate at N=500
    rng = np.random.default_rng(11)
    E, _, W, Y = sample_scm1(gen, None, n=500, beta_direct=beta, rng=rng)
    res = dml_estimator(E, W, Y)
    print(f"  SCM_1(0.10) DML  theta={res.theta:+.4f}  "
          f"CI=[{res.ci_low:+.4f}, {res.ci_high:+.4f}]  truth={truth:+.4f}")

    assert np.isfinite(res.theta)
    # Smoke tolerance: allow +/- 0.05 around the population truth (this is
    # one replicate at N=500; we expect order-of-MC-SE deviation).
    assert abs(res.theta - truth) <= 0.05, (
        f"theta_hat={res.theta:+.4f} too far from truth={truth:+.4f} "
        f"(diff={res.theta - truth:+.4f}); smoke tolerance is +/- 0.05."
    )


def test_other_estimators_run(gen):
    """All four estimators should run on a single SCM_0 draw without crashing."""
    rng = np.random.default_rng(101)
    E, _, W, Y = sample_scm0(gen, None, n=500, rng=rng)

    res_dr = dr_estimator(E, W, Y)
    print(f"  DR  theta={res_dr.theta:+.4f}  CI=[{res_dr.ci_low:+.4f}, {res_dr.ci_high:+.4f}]")
    assert np.isfinite(res_dr.theta)

    res_adv = adversarial_estimator(E, W, Y, epochs=20)
    print(f"  Adv theta={res_adv.theta:+.4f}  CI=[{res_adv.ci_low:+.4f}, {res_adv.ci_high:+.4f}]")
    assert np.isfinite(res_adv.theta)

    res_rand = randomization_estimator(E, W, Y, n_perm=10)
    print(f"  Rand delta_R2={res_rand.theta:+.4f}  CI=[{res_rand.ci_low:+.4f}, {res_rand.ci_high:+.4f}]")
    assert np.isfinite(res_rand.theta)
