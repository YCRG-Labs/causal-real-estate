"""Uniform-interface wrappers around the four causal estimators.

Each wrapper returns an `EstimateResult(theta, se, ci_low, ci_high, extras)`
so the Monte Carlo loop can compare apples to apples. Three wrappers thinly
adapt the heavy implementations in `causal_inference.py`; the adversarial
wrapper uses a fast simplified version (single-head linear discriminator,
50 epochs) suitable for thousands of replicates -- the production multi-head
+ frozen-probe pipeline in causal_inference is too slow for simulation.

Returned theta is on the estimator's native scale (DML: per-sigma of PC1;
DR: ATE between binarized halves on the log-Y outcome; Adversarial: linear
coefficient of PC1 on the deconfounded residual; Randomization: delta R^2
between original and permuted-confounder fits). The simulation orchestrator
calibrates the "truth" per estimator at population scale so that bias is
measured against the right estimand.
"""
from __future__ import annotations

import contextlib
import io
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from causal_inference import (  # noqa: E402
    doubly_robust_estimation,
    dml_continuous_treatment,
)


@dataclass
class EstimateResult:
    estimator: str
    theta: float
    se: float
    ci_low: float
    ci_high: float
    extras: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "extras"}
        d.update({f"extra_{k}": v for k, v in self.extras.items()})
        return d


@contextlib.contextmanager
def _silent():
    """Swallow stdout/stderr from the chatty causal_inference functions."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield


# ---------------------------------------------------------------------------
# (1) Doubly-robust estimator (binarized PC1-norm treatment)
# ---------------------------------------------------------------------------

def dr_estimator(T: np.ndarray, W: np.ndarray, Y: np.ndarray) -> EstimateResult:
    """Wrap `causal_inference.doubly_robust_estimation`.

    Returns IF-CI rather than bootstrap CI because the IF interval is what we
    want for coverage / power calculations (matches the asymptotic theory the
    simulation is testing). Bootstrap interval is recorded in extras for
    debugging.
    """
    with _silent():
        dr_effect, boot_ci, extras = doubly_robust_estimation(T, W, Y)
    if_lo, if_hi = extras["if_ci"]
    return EstimateResult(
        estimator="DR",
        theta=float(dr_effect),
        se=float(extras["if_se"]),
        ci_low=float(if_lo),
        ci_high=float(if_hi),
        extras={
            "boot_ci_low": float(boot_ci[0]),
            "boot_ci_high": float(boot_ci[1]),
            "mde": float(extras["mde"]),
        },
    )


# ---------------------------------------------------------------------------
# (2) DML continuous-treatment estimator (PC1)
# ---------------------------------------------------------------------------

def dml_estimator(T: np.ndarray, W: np.ndarray, Y: np.ndarray) -> EstimateResult:
    """Wrap `causal_inference.dml_continuous_treatment`."""
    with _silent():
        out = dml_continuous_treatment(T, W, Y)
    if out is None:
        # Treatment fully explained by W -- emit a NaN result so the cell
        # records the failure rather than crashing the loop.
        return EstimateResult(
            estimator="DML",
            theta=float("nan"), se=float("nan"),
            ci_low=float("nan"), ci_high=float("nan"),
            extras={"failed": True},
        )
    lo, hi = out["ci"]
    return EstimateResult(
        estimator="DML",
        theta=float(out["theta"]),
        se=float(out["se"]),
        ci_low=float(lo),
        ci_high=float(hi),
        extras={"mde": float(out["mde"])},
    )


# ---------------------------------------------------------------------------
# (3) Adversarial deconfounding (simplified, fast)
# ---------------------------------------------------------------------------

class _SmallEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class _LinearHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class _GradReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def adversarial_estimator(
    T: np.ndarray,
    W: np.ndarray,
    Y: np.ndarray,
    n_pca: int = 50,
    epochs: int = 50,
    lr: float = 1e-3,
    seed: int = 42,
) -> EstimateResult:
    """Simplified adversarial deconfounding for simulation.

    Pipeline:
      1. PCA(T) -> T_pca, take pc1 = T_pca[:,0] z-scored
      2. Encode confounders W with 2-layer MLP -> z_repr
      3. Train predictor head Y ~ z_repr while a discriminator head
         tries to predict pc1 from z_repr (gradient-reversal removes
         pc1 signal from z_repr)
      4. Fit a final OLS of Y on [pc1, z_repr_detached]; the coefficient
         on pc1 is theta. SE from OLS HC0 (heteroskedasticity-robust).

    The full causal_inference adversarial routine uses a multi-head
    discriminator + frozen probe + 150 epochs. We strip that to a single
    linear head + 50 epochs because each simulation cell runs hundreds of
    replicates and we need each to complete in seconds.

    Refs:
      Ganin et al. 2016 JMLR -- domain-adversarial training
      Ravfogel et al. 2020 ACL -- nullspace projection
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(Y)
    n_pca_eff = min(n_pca, T.shape[1], max(n - 1, 1))
    pca = PCA(n_components=n_pca_eff, random_state=seed)
    T_pca = pca.fit_transform(T)
    pc1 = T_pca[:, 0]
    pc1 = (pc1 - pc1.mean()) / (pc1.std() if pc1.std() > 0 else 1.0)

    W_s = StandardScaler().fit_transform(W)

    W_t = torch.FloatTensor(W_s)
    Y_t = torch.FloatTensor(Y - Y.mean())
    pc1_t = torch.FloatTensor(pc1)

    encoder = _SmallEncoder(W_s.shape[1], hidden_dim=64, output_dim=32)
    predictor = _LinearHead(32, 1)
    discriminator = _LinearHead(32, 1)

    opt_main = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    opt_disc = torch.optim.Adam(discriminator.parameters(), lr=lr, weight_decay=1e-4)
    mse = nn.MSELoss()

    for ep in range(epochs):
        # Adversarial schedule: ramp lambda from 0.1 -> 1.0 over training.
        lam = 0.1 + 0.9 * min(ep / max(epochs - 1, 1), 1.0)

        encoder.train(); predictor.train(); discriminator.train()
        z = encoder(W_t)
        y_pred = predictor(z).squeeze(-1)
        pred_loss = mse(y_pred, Y_t)
        z_rev = _GradReversal.apply(z, lam)
        d_pred = discriminator(z_rev).squeeze(-1)
        disc_loss = mse(d_pred, pc1_t)
        total = pred_loss + disc_loss
        opt_main.zero_grad()
        total.backward()
        opt_main.step()

        z_det = encoder(W_t).detach()
        d_pred2 = discriminator(z_det).squeeze(-1)
        disc_loss2 = mse(d_pred2, pc1_t)
        opt_disc.zero_grad()
        disc_loss2.backward()
        opt_disc.step()

    encoder.eval()
    with torch.no_grad():
        z_repr = encoder(W_t).numpy()

    # Residualize Y on z_repr (linear), then regress on pc1 alone.
    # This is the "deconfounded" pipeline: the encoder removed pc1's signal
    # from z_repr, so what's left in (Y - alpha . z_repr) attributable to pc1
    # is the direct effect.
    X_aug = np.column_stack([np.ones(n), z_repr])
    beta_y, *_ = np.linalg.lstsq(X_aug, Y - Y.mean(), rcond=None)
    Y_hat = X_aug @ beta_y
    Y_resid = (Y - Y.mean()) - Y_hat

    # Regress residuals on pc1.
    X_pc1 = np.column_stack([np.ones(n), pc1])
    coef, *_ = np.linalg.lstsq(X_pc1, Y_resid, rcond=None)
    theta = float(coef[1])

    # HC0 heteroskedasticity-robust SE for theta.
    eps = Y_resid - X_pc1 @ coef
    XtX_inv = np.linalg.pinv(X_pc1.T @ X_pc1)
    meat = X_pc1.T @ (X_pc1 * (eps ** 2)[:, None])
    cov = XtX_inv @ meat @ XtX_inv
    se = float(np.sqrt(max(cov[1, 1], 0.0)))
    ci_lo = theta - 1.96 * se
    ci_hi = theta + 1.96 * se

    return EstimateResult(
        estimator="Adversarial",
        theta=theta,
        se=se,
        ci_low=float(ci_lo),
        ci_high=float(ci_hi),
        extras={"epochs": epochs, "encoder_dim": 32},
    )


# ---------------------------------------------------------------------------
# (4) Randomization test
# ---------------------------------------------------------------------------

def randomization_estimator(
    T: np.ndarray,
    W: np.ndarray,
    Y: np.ndarray,
    n_perm: int = 100,
    n_pca: int = 30,
    seed: int = 42,
) -> EstimateResult:
    """Lightweight randomization variant for simulation.

    Statistic: delta_R2 = R2_original - mean(R2_permuted_W).
    SE: across-permutation SD of (R2_orig - R2_perm) / sqrt(n_perm).

    The full randomization_test in causal_inference.py uses 200-tree GBRs;
    that's too slow for simulation. We use a smaller GBR (50 trees, depth 3)
    -- the variance of the bias estimate dominates over the smoothness loss
    from a smaller learner.
    """
    rng = np.random.RandomState(seed)
    n = len(Y)
    n_pca_eff = min(n_pca, T.shape[1], max(n - 1, 1))
    pca = PCA(n_components=n_pca_eff, random_state=seed)
    T_pca = pca.fit_transform(T)
    T_s = StandardScaler().fit_transform(T_pca)

    feats_orig = np.hstack([T_s, W])
    train_n = int(n * 0.7)
    perm_idx = rng.permutation(n)
    tr, te = perm_idx[:train_n], perm_idx[train_n:]

    def _fit_score(X_tr, y_tr, X_te, y_te, rng_seed: int) -> float:
        m = GradientBoostingRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=rng_seed,
        )
        m.fit(X_tr, y_tr)
        return float(m.score(X_te, y_te))

    r2_orig = _fit_score(feats_orig[tr], Y[tr], feats_orig[te], Y[te], seed)

    r2_perms = np.empty(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(n)
        feats_p = np.hstack([T_s, W[perm]])
        r2_perms[k] = _fit_score(feats_p[tr], Y[tr], feats_p[te], Y[te], seed + k + 1)

    delta = r2_orig - float(r2_perms.mean())
    # SD across permutations -> SE of delta under the random-permutation null
    se = float(r2_perms.std(ddof=1)) if n_perm > 1 else float("nan")
    p_value = float(np.mean(r2_perms >= r2_orig))

    return EstimateResult(
        estimator="Randomization",
        theta=float(delta),
        se=se,
        ci_low=float(delta - 1.96 * se),
        ci_high=float(delta + 1.96 * se),
        extras={
            "r2_original": float(r2_orig),
            "r2_permuted_mean": float(r2_perms.mean()),
            "p_value": p_value,
            "n_perm": int(n_perm),
        },
    )


# Public registry for the orchestrator to iterate over.
ESTIMATORS = {
    "DR": dr_estimator,
    "DML": dml_estimator,
    "Adversarial": adversarial_estimator,
    "Randomization": randomization_estimator,
}
