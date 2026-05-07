"""Data-generating processes for the simulation validation.

Two structural causal models:

  SCM_0:  Y = alpha * z_score + W @ beta_W + eps
          (no direct E -> Y term; text embedding is a non-causal proxy of z)

  SCM_1:  Y = alpha * z_score + W @ beta_W + beta_direct * proj(E) + eps
          where proj(E) is a fixed scalar projection of the text embedding
          (PC1 by default) and beta_direct is calibrated so that the
          population variance Var(beta_direct * proj(E)) equals a target
          fraction of Var(Y) (1%, 5%, or 10%).

`E` is sampled from a per-zip-bin Gaussian-mixture generator fitted on real
(E, z) pairs from `release/data/sf/embeddings_mpnet.parquet`. The dossier
(research/simulation/research_notes.md) prefers a 2-layer RealNVP normalizing
flow but the GaussianMixture fallback is what we use here -- per-bin means
plus low-rank covariances captures the dominant variability without the
nflows install dependency.

`W` (the confounder block, mimicking census/crime/amenity features that load
on location) is generated as nonlinear transforms of `z` plus IID noise so
DML / DR have meaningful adjustment work to do (without that, every estimator
collapses to OLS-on-text and the simulation tests nothing).

Refs:
  Athey, Imbens, Metzger & Munro 2024, JoE 240(2):105076 -- generator-based MC
  Knaus, Lechner & Strittmatter 2021, EJ 24(1):134-161 -- empirical MC
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Default location of the real (E, z) pairs we fit the generator on.
DEFAULT_REAL_PARQUET = (
    Path(__file__).resolve().parents[3]
    / "release" / "data" / "sf" / "embeddings_mpnet.parquet"
)


@dataclass
class _BinParams:
    """Per-bin Gaussian parameters for low-rank N(mu, U U^T + sigma^2 I)."""
    mu: np.ndarray            # (d,)
    U: np.ndarray             # (d, k) low-rank loadings
    sigma2: float             # isotropic residual variance
    n_train: int              # training rows from this bin


@dataclass
class GaussianMixtureGenerator:
    """Per-zip-bin Gaussian generator on text embeddings.

    For each bin z:
      E | z = N(mu_z, U_z U_z^T + sigma_z^2 I)

    Sampling marginal-on-z is achieved by drawing z from the empirical bin
    distribution `bin_freq` (or from a supplied population). Used in place of
    a normalizing flow for transparency and dependency-light deployment.
    """
    bins: np.ndarray                                # (B,) sorted bin labels
    params: dict[int, _BinParams] = field(default_factory=dict)
    bin_freq: np.ndarray | None = None              # (B,) marginal P(z)
    embedding_dim: int = 0
    low_rank: int = 10
    population_mean: np.ndarray | None = None       # (d,) global mean
    population_cov_diag: np.ndarray | None = None   # (d,) global var (fallback)
    pc1_direction: np.ndarray | None = None         # (d,) population PC1 (unit)

    def sample_E(self, z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample one embedding per supplied bin label z[i]."""
        n = len(z)
        d = self.embedding_dim
        E = np.empty((n, d), dtype=np.float64)
        for b in self.bins:
            idx = np.where(z == b)[0]
            if len(idx) == 0:
                continue
            p = self.params.get(int(b))
            if p is None:
                # Fallback to global params for unseen bin
                E[idx] = (
                    self.population_mean
                    + rng.standard_normal((len(idx), d))
                    * np.sqrt(self.population_cov_diag)
                )
                continue
            k = p.U.shape[1]
            f = rng.standard_normal((len(idx), k))     # latent factors
            eps = rng.standard_normal((len(idx), d)) * np.sqrt(p.sigma2)
            E[idx] = p.mu + f @ p.U.T + eps
        return E

    def sample_z(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.bin_freq is None:
            # Uniform fallback if marginal not set
            return rng.choice(self.bins, size=n, replace=True)
        return rng.choice(self.bins, size=n, replace=True, p=self.bin_freq)


def fit_generator(
    real_E: np.ndarray, real_z: np.ndarray, low_rank: int = 10, min_bin_n: int = 10
) -> GaussianMixtureGenerator:
    """Fit per-bin low-rank Gaussian on real (E, z) pairs.

    Bins with fewer than `min_bin_n` rows fall back to the global population
    mean + diagonal covariance to avoid degenerate per-bin fits.
    """
    real_E = np.asarray(real_E, dtype=np.float64)
    real_z = np.asarray(real_z)
    d = real_E.shape[1]
    bins, counts = np.unique(real_z, return_counts=True)
    freq = counts.astype(float) / counts.sum()

    pop_mean = real_E.mean(axis=0)
    pop_cov_diag = real_E.var(axis=0, ddof=1) + 1e-8

    # Population PC1 direction (unit vector) -- used for SCM_1 projection.
    centered = real_E - pop_mean
    # SVD on centered data; first right-singular vector = first PC direction.
    # Cap rank to avoid huge SVD when d is large but n is small.
    rank_cap = min(centered.shape[0], centered.shape[1], 32)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    pc1 = Vt[0]
    pc1 = pc1 / (np.linalg.norm(pc1) + 1e-12)

    params: dict[int, _BinParams] = {}
    for b in bins:
        idx = np.where(real_z == b)[0]
        n_b = len(idx)
        E_b = real_E[idx]
        if n_b < min_bin_n:
            # Use population mean and diagonal var as proxy
            params[int(b)] = _BinParams(
                mu=pop_mean.copy(),
                U=np.zeros((d, 0)),
                sigma2=float(pop_cov_diag.mean()),
                n_train=int(n_b),
            )
            continue
        mu_b = E_b.mean(axis=0)
        Eb_c = E_b - mu_b
        # Low-rank truncated SVD: U = V_k * sqrt(s_k^2 / (n-1))
        rk = max(1, min(low_rank, n_b - 1, d))
        u, s, vt = np.linalg.svd(Eb_c, full_matrices=False)
        # Top-k components scaled to standard-deviation loadings.
        s_k = s[:rk]
        Vk = vt[:rk]                                     # (rk, d)
        loadings = (Vk.T * (s_k / np.sqrt(max(n_b - 1, 1))))  # (d, rk)
        # Residual variance: average over remaining singular values^2 / (n-1)
        if len(s) > rk:
            resid = float(np.sum(s[rk:] ** 2) / max(n_b - 1, 1) / d)
        else:
            resid = 1e-6
        resid = max(resid, 1e-8)
        params[int(b)] = _BinParams(
            mu=mu_b, U=loadings, sigma2=resid, n_train=int(n_b)
        )

    return GaussianMixtureGenerator(
        bins=bins,
        params=params,
        bin_freq=freq,
        embedding_dim=d,
        low_rank=low_rank,
        population_mean=pop_mean,
        population_cov_diag=pop_cov_diag,
        pc1_direction=pc1,
    )


# ---------------------------------------------------------------------------
# Confounder (W) generator: nonlinear transforms of z + noise
# ---------------------------------------------------------------------------

def _z_to_score(z: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Map raw bin labels to a deterministic real-valued z-score in [-1, 1].

    We need a numeric handle on `z` to drive the structural equation. Sorting
    the bin labels and rescaling to [-1, 1] gives a deterministic, monotone
    embedding that doesn't rely on any auxiliary feature.
    """
    sorted_bins = np.sort(np.unique(bins))
    rank = {int(b): i for i, b in enumerate(sorted_bins)}
    B = max(len(sorted_bins) - 1, 1)
    out = np.array([rank[int(b)] / B for b in z], dtype=np.float64)
    return 2.0 * out - 1.0   # rescale to [-1, 1]


def _generate_W(
    z_score: np.ndarray, n_dim: int, rng: np.random.Generator
) -> np.ndarray:
    """Build a confounder block W that loads on z (so DML must adjust).

    Mimics the structure of census + amenity features that vary with location.
    Each column is a different smooth-or-trig function of z plus IID noise.
    """
    n = len(z_score)
    cols = [
        z_score + 0.5 * rng.standard_normal(n),           # ~ income
        np.sin(np.pi * z_score) + 0.4 * rng.standard_normal(n),
        np.cos(2.0 * z_score) + 0.4 * rng.standard_normal(n),
        z_score ** 2 - 0.5 + 0.5 * rng.standard_normal(n),
        np.tanh(2.0 * z_score) + 0.3 * rng.standard_normal(n),
    ]
    while len(cols) < n_dim:
        # Pad with random linear combos + noise (noise-only columns)
        cols.append(rng.standard_normal(n))
    return np.column_stack(cols[:n_dim])


# ---------------------------------------------------------------------------
# SCM_0 and SCM_1 samplers
# ---------------------------------------------------------------------------

DEFAULT_ALPHA = 0.6      # location -> log-price slope
DEFAULT_BETA_W = 0.4     # confounder -> log-price slope on first W column
DEFAULT_NOISE_SD = 0.3   # epsilon SD for log-price


def sample_scm0(
    gen: GaussianMixtureGenerator,
    z_population: np.ndarray | None,
    n: int,
    n_W: int = 5,
    rng: np.random.Generator | None = None,
    alpha: float = DEFAULT_ALPHA,
    beta_W: float = DEFAULT_BETA_W,
    noise_sd: float = DEFAULT_NOISE_SD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample (E, z, W, Y) from SCM_0.  No direct E -> Y term.

    Y = alpha * z_score + beta_W * W[:, 0] + eps
    """
    if rng is None:
        rng = np.random.default_rng()
    if z_population is None:
        z = gen.sample_z(n, rng)
    else:
        z = rng.choice(z_population, size=n, replace=True)
    z_score = _z_to_score(z, gen.bins)
    W = _generate_W(z_score, n_W, rng)
    E = gen.sample_E(z, rng)
    Y = (
        alpha * z_score
        + beta_W * W[:, 0]
        + noise_sd * rng.standard_normal(n)
    )
    return E, z, W, Y


def calibrate_beta_direct(
    gen: GaussianMixtureGenerator,
    target_var_share: float,
    n_pop: int = 20000,
    rng: np.random.Generator | None = None,
    alpha: float = DEFAULT_ALPHA,
    beta_W: float = DEFAULT_BETA_W,
    noise_sd: float = DEFAULT_NOISE_SD,
    n_W: int = 5,
) -> float:
    """Compute beta_direct so that Var(beta_direct * proj(E)) / Var(Y) = target.

    Done once per cell at population level (n_pop large) so that the
    same beta_direct is used across all replicates of one cell.
    """
    if rng is None:
        rng = np.random.default_rng(20260429)
    E, z, W, _ = sample_scm0(
        gen, None, n_pop, n_W=n_W, rng=rng,
        alpha=alpha, beta_W=beta_W, noise_sd=noise_sd,
    )
    Y0 = (
        alpha * _z_to_score(z, gen.bins)
        + beta_W * W[:, 0]
        + noise_sd * rng.standard_normal(n_pop)
    )
    var_Y0 = float(np.var(Y0, ddof=1))
    proj = E @ gen.pc1_direction
    var_proj = float(np.var(proj, ddof=1))
    if var_proj <= 1e-12:
        return 0.0
    # Variance contribution v of (b * proj) is b^2 * Var(proj). To make
    # v / Var(Y_full) = target, with Var(Y_full) = Var(Y0) + v, solve for b:
    #   b^2 = target / (1 - target) * Var(Y0) / Var(proj)
    b2 = (target_var_share / (1.0 - target_var_share)) * var_Y0 / var_proj
    return float(np.sqrt(max(b2, 0.0)))


def sample_scm1(
    gen: GaussianMixtureGenerator,
    z_population: np.ndarray | None,
    n: int,
    beta_direct: float,
    n_W: int = 5,
    rng: np.random.Generator | None = None,
    alpha: float = DEFAULT_ALPHA,
    beta_W: float = DEFAULT_BETA_W,
    noise_sd: float = DEFAULT_NOISE_SD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample (E, z, W, Y) from SCM_1 with calibrated direct effect.

    Y = alpha * z_score + beta_W * W[:,0] + beta_direct * proj(E) + eps
    where proj(E) := E @ pc1_direction.
    """
    if rng is None:
        rng = np.random.default_rng()
    if z_population is None:
        z = gen.sample_z(n, rng)
    else:
        z = rng.choice(z_population, size=n, replace=True)
    z_score = _z_to_score(z, gen.bins)
    W = _generate_W(z_score, n_W, rng)
    E = gen.sample_E(z, rng)
    proj = E @ gen.pc1_direction
    Y = (
        alpha * z_score
        + beta_W * W[:, 0]
        + beta_direct * proj
        + noise_sd * rng.standard_normal(n)
    )
    return E, z, W, Y


def load_real_pairs(
    parquet_path: Path | str | None = None, n_subsample: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the (E, z) pairs we fit the generator on."""
    p = Path(parquet_path) if parquet_path is not None else DEFAULT_REAL_PARQUET
    df = pd.read_parquet(p)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise ValueError(f"No emb_* columns found in {p}")
    if "zip" not in df.columns:
        raise ValueError(f"No 'zip' column in {p}")
    if n_subsample is not None and n_subsample < len(df):
        df = df.sample(n=n_subsample, random_state=seed).reset_index(drop=True)
    E = df[emb_cols].to_numpy(dtype=np.float64)
    z = df["zip"].to_numpy()
    return E, z
