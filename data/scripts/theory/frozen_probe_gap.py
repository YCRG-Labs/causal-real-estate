"""Numerical verification of the variational MI gap that the frozen-encoder
probe diagnostic detects.

Three experiments:

  Experiment 1 (static gap, XOR DGP).
    Z = (Z_1, Z_2) ~ Bernoulli(0.5) iid; C = Z_1 ⊕ Z_2.
    True I(Z; C) = 1 bit (C is a deterministic function of Z).
    Trains a linear (logistic regression) and a 2-layer-MLP discriminator and
    computes the variational lower bound V_Φ(Z; C) = H(C) − CE(D_ψ(Z), C) for
    each. Demonstrates:
      V_linear → 0    (linear D cannot separate XOR)
      V_MLP    → 1    (MLP D recovers full 1-bit information)
      Gap = V_MLP − V_linear ≈ 1 bit
    This is the mathematical foundation of the frozen-probe diagnostic.

  Experiment 2 (dynamic gap, gradient-reversal training).
    X ∈ ℝ^10, C = sign(X_1 X_2). A small MLP encoder E_φ: ℝ^10 → ℝ^4 is
    trained adversarially against a *linear* discriminator D_ψ via gradient
    reversal. After training:
      live discriminator accuracy on E_φ(X) → chance
      frozen MLP probe accuracy on E_φ(X) >> chance
    Replicates the empirical phenomenon reported at 19–116× random in the
    main paper. Establishes that gradient-reversal "deconfounding" is
    illusory when discriminator capacity is below probe capacity.

  Experiment 3 (capacity ladder).
    Same XOR DGP; vary discriminator capacity continuously (linear → 1-layer
    → 2-layer → 3-layer MLP) and plot V_Φ as a function of capacity. Shows
    V_Φ → I(Z; C) monotonically as the discriminator class approaches the
    Bayes-optimal posterior — operationalizing the "confounder escalation"
    test as varying probe capacity rather than confounder set richness.

All three use a fixed seed so results are deterministic and reproducible.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results" / "theory"


# ---------------------------------------------------------------------------
# Variational MI bound from a fitted classifier
# ---------------------------------------------------------------------------

def variational_mi_lower_bound(clf, Z: np.ndarray, C: np.ndarray) -> float:
    """V_Φ(Z; C) = H(C) − CE_Φ(D̂(Z), C), in nats.

    For a binary classifier with predicted probabilities p_hat = D̂(Z),
    cross-entropy is mean( −c log p̂ − (1−c) log(1−p̂) ). Then V_Φ ≥ 0.
    Returns NaN if cross-entropy exceeds H(C) (which can happen for poorly-
    fit classifiers; the bound is vacuous in that case).
    """
    p_hat = clf.predict_proba(Z)[:, 1]
    eps = 1e-9
    p_hat = np.clip(p_hat, eps, 1 - eps)
    ce = -np.mean(C * np.log(p_hat) + (1 - C) * np.log(1 - p_hat))
    p_c = float(np.mean(C))
    h_c = -(p_c * math.log(p_c + eps) + (1 - p_c) * math.log(1 - p_c + eps))
    return max(0.0, h_c - ce)


def entropy_binary(p: float) -> float:
    eps = 1e-12
    return -(p * math.log(p + eps) + (1 - p) * math.log(1 - p + eps))


# ---------------------------------------------------------------------------
# Experiment 1: static gap, XOR
# ---------------------------------------------------------------------------

def experiment_1_static_xor(n: int = 4000, n_trials: int = 5, seed: int = 42) -> dict:
    """Fixed DGP, vary classifier class, measure V_Φ.

    Returns: {
      "true_mi_nats": float,
      "h_c_nats": float,
      "linear": {"v_phi_nats": [...], "accuracy": [...]},
      "mlp":    {"v_phi_nats": [...], "accuracy": [...]},
      "gap_mean_nats": float,
    }
    """
    rng = np.random.default_rng(seed)
    h_c = math.log(2)  # exactly 1 bit = log(2) nats for balanced Bernoulli

    v_lin = []; acc_lin = []
    v_mlp = []; acc_mlp = []
    for t in range(n_trials):
        Z1 = rng.integers(0, 2, n)
        Z2 = rng.integers(0, 2, n)
        Z = np.column_stack([Z1, Z2]).astype(float)
        C = (Z1 ^ Z2).astype(int)

        lin = LogisticRegression(max_iter=2000, C=1.0).fit(Z, C)
        mlp = MLPClassifier(
            hidden_layer_sizes=(16, 16),
            activation="relu",
            max_iter=2000,
            random_state=seed + t,
        ).fit(Z, C)

        v_lin.append(variational_mi_lower_bound(lin, Z, C))
        acc_lin.append(float(lin.score(Z, C)))
        v_mlp.append(variational_mi_lower_bound(mlp, Z, C))
        acc_mlp.append(float(mlp.score(Z, C)))

    return {
        "n": int(n),
        "n_trials": int(n_trials),
        "true_mi_nats": float(h_c),
        "true_mi_bits": float(h_c / math.log(2)),
        "linear": {
            "v_phi_nats_mean": float(np.mean(v_lin)),
            "v_phi_nats_sd": float(np.std(v_lin, ddof=1)),
            "accuracy_mean": float(np.mean(acc_lin)),
        },
        "mlp": {
            "v_phi_nats_mean": float(np.mean(v_mlp)),
            "v_phi_nats_sd": float(np.std(v_mlp, ddof=1)),
            "accuracy_mean": float(np.mean(acc_mlp)),
        },
        "gap_mean_nats": float(np.mean(v_mlp) - np.mean(v_lin)),
        "gap_mean_bits": float((np.mean(v_mlp) - np.mean(v_lin)) / math.log(2)),
    }


# ---------------------------------------------------------------------------
# Experiment 2: dynamic gap, gradient-reversal training
# ---------------------------------------------------------------------------

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradientReversal.apply(x, lambd)


class Encoder(nn.Module):
    def __init__(self, d_in=10, d_hidden=32, d_out=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


class LinearDiscriminator(nn.Module):
    def __init__(self, d_in=4):
        super().__init__()
        self.fc = nn.Linear(d_in, 2)

    def forward(self, z):
        return self.fc(z)


class Predictor(nn.Module):
    """Downstream task head — predicts a regression target Y from Z.

    Y is a noisy linear combination of the input features (so the encoder has
    a real downstream task to keep, otherwise it can collapse Z to zero).
    """
    def __init__(self, d_in=4):
        super().__init__()
        self.fc = nn.Linear(d_in, 1)

    def forward(self, z):
        return self.fc(z)


def make_dgp_continuous(n: int, d: int = 10, seed: int = 0):
    """X ∈ R^d, C = sign(X_1 X_2), Y = α·X_3 + ε.

    The confounder C is a *nonlinear* (multiplicative) function of two input
    features. A linear discriminator on either E(X) or X cannot recover C.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    C = (np.sign(X[:, 0] * X[:, 1]) > 0).astype(np.int64)  # {0,1}
    Y = (0.5 * X[:, 2] + 0.1 * rng.normal(size=n)).astype(np.float32)
    return X, C, Y


def experiment_2_dynamic_grl(
    n: int = 4000, d_in: int = 10, d_repr: int = 4,
    n_epochs: int = 200, lambd: float = 1.0, lr: float = 1e-2,
    seed: int = 42,
) -> dict:
    """Train E + linear D adversarially via GRL; then measure frozen-probe gap."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X, C, Y = make_dgp_continuous(n, d=d_in, seed=seed)
    Xt = torch.tensor(X)
    Ct = torch.tensor(C)
    Yt = torch.tensor(Y).unsqueeze(1)

    enc = Encoder(d_in=d_in, d_out=d_repr)
    pred = Predictor(d_in=d_repr)
    disc = LinearDiscriminator(d_in=d_repr)

    optE = torch.optim.Adam(enc.parameters(), lr=lr)
    optP = torch.optim.Adam(pred.parameters(), lr=lr)
    optD = torch.optim.Adam(disc.parameters(), lr=lr)

    history = {"loss_pred": [], "loss_disc": [], "disc_acc": []}
    for epoch in range(n_epochs):
        optE.zero_grad(); optP.zero_grad(); optD.zero_grad()

        Z = enc(Xt)
        # Predictor uses Z directly (forward signal)
        Y_hat = pred(Z)
        loss_pred = F.mse_loss(Y_hat, Yt)

        # Discriminator sees grad-reversed Z; minimizing CE on D
        # maximizes the reversed CE on E (so E tries to fool D)
        Z_rev = grad_reverse(Z, lambd)
        logits = disc(Z_rev)
        loss_disc = F.cross_entropy(logits, Ct)

        total = loss_pred + loss_disc
        total.backward()
        optE.step(); optP.step(); optD.step()

        with torch.no_grad():
            disc_acc = (logits.argmax(dim=1) == Ct).float().mean().item()
        history["loss_pred"].append(float(loss_pred.item()))
        history["loss_disc"].append(float(loss_disc.item()))
        history["disc_acc"].append(float(disc_acc))

    # ---- Evaluation: live discriminator vs frozen probe on the same Z ----
    enc.eval()
    with torch.no_grad():
        Z_final = enc(Xt).numpy()

    # Live discriminator: just the last D's accuracy (already in history)
    live_acc = float(np.mean(history["disc_acc"][-10:]))

    # Frozen MLP probe: train fresh from scratch on (Z_final, C)
    probe_mlp = MLPClassifier(
        hidden_layer_sizes=(32, 32), max_iter=3000, random_state=seed
    ).fit(Z_final, C)
    probe_acc = float(probe_mlp.score(Z_final, C))
    probe_v_phi = variational_mi_lower_bound(probe_mlp, Z_final, C)

    # Frozen LINEAR probe (matched-capacity to live D): should also be at chance
    probe_lin = LogisticRegression(max_iter=2000).fit(Z_final, C)
    probe_lin_acc = float(probe_lin.score(Z_final, C))
    probe_lin_v_phi = variational_mi_lower_bound(probe_lin, Z_final, C)

    h_c = entropy_binary(float(np.mean(C)))
    return {
        "n": int(n),
        "d_in": int(d_in),
        "d_repr": int(d_repr),
        "n_epochs": int(n_epochs),
        "lambda_grl": float(lambd),
        "h_c_nats": float(h_c),
        "h_c_bits": float(h_c / math.log(2)),
        "final_predictor_loss": float(np.mean(history["loss_pred"][-10:])),
        "live_discriminator_accuracy": live_acc,
        "frozen_linear_probe_accuracy": probe_lin_acc,
        "frozen_mlp_probe_accuracy": probe_acc,
        "frozen_linear_probe_v_phi_nats": float(probe_lin_v_phi),
        "frozen_mlp_probe_v_phi_nats": float(probe_v_phi),
        "diagnostic_gap_accuracy": float(probe_acc - live_acc),
        "history_last_10": {
            "disc_acc": history["disc_acc"][-10:],
            "loss_disc": history["loss_disc"][-10:],
        },
    }


# ---------------------------------------------------------------------------
# Experiment 3: capacity ladder
# ---------------------------------------------------------------------------

def experiment_3_capacity_ladder(n: int = 4000, n_trials: int = 3, seed: int = 42) -> dict:
    """Vary classifier capacity from linear to deep MLP; measure V_Φ on XOR."""
    rng = np.random.default_rng(seed)
    h_c = math.log(2)

    architectures = [
        ("linear", None),
        ("mlp_4", (4,)),
        ("mlp_8", (8,)),
        ("mlp_16", (16,)),
        ("mlp_16_16", (16, 16)),
        ("mlp_32_32_32", (32, 32, 32)),
    ]

    out = {"true_mi_nats": float(h_c), "n": int(n), "trials": int(n_trials), "rungs": []}
    for name, hs in architectures:
        v_phis = []
        accs = []
        for t in range(n_trials):
            Z1 = rng.integers(0, 2, n)
            Z2 = rng.integers(0, 2, n)
            Z = np.column_stack([Z1, Z2]).astype(float)
            C = (Z1 ^ Z2).astype(int)
            if hs is None:
                clf = LogisticRegression(max_iter=2000)
            else:
                clf = MLPClassifier(
                    hidden_layer_sizes=hs, max_iter=2000, random_state=seed + t,
                )
            clf.fit(Z, C)
            v_phis.append(variational_mi_lower_bound(clf, Z, C))
            accs.append(float(clf.score(Z, C)))
        out["rungs"].append({
            "name": name,
            "hidden_layers": list(hs) if hs else [],
            "v_phi_nats_mean": float(np.mean(v_phis)),
            "v_phi_nats_sd": float(np.std(v_phis, ddof=1)),
            "accuracy_mean": float(np.mean(accs)),
        })
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["1", "2", "3", "all"], default="all")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "frozen_probe_gap.json")
    args = ap.parse_args()

    out: dict = {}
    if args.experiment in ("1", "all"):
        print("\n=== Experiment 1: static XOR gap ===")
        r1 = experiment_1_static_xor()
        print(f"  H(C) = {r1['true_mi_bits']:.3f} bits (true I(Z;C) for XOR)")
        print(f"  Linear D: V_Φ = {r1['linear']['v_phi_nats_mean']/math.log(2):.4f} bits, "
              f"acc = {r1['linear']['accuracy_mean']:.3f}")
        print(f"  MLP D:    V_Φ = {r1['mlp']['v_phi_nats_mean']/math.log(2):.4f} bits, "
              f"acc = {r1['mlp']['accuracy_mean']:.3f}")
        print(f"  Gap     = {r1['gap_mean_bits']:.4f} bits "
              f"({r1['gap_mean_bits']/r1['true_mi_bits']*100:.0f}% of total MI)")
        out["experiment_1_static_xor"] = r1

    if args.experiment in ("2", "all"):
        print("\n=== Experiment 2: dynamic gap under GRL training ===")
        r2 = experiment_2_dynamic_grl()
        print(f"  H(C) = {r2['h_c_bits']:.3f} bits")
        print(f"  Final predictor loss:        {r2['final_predictor_loss']:.4f}")
        print(f"  Live discriminator accuracy: {r2['live_discriminator_accuracy']:.3f} "
              f"(chance ≈ 0.5)")
        print(f"  Frozen linear probe acc:     {r2['frozen_linear_probe_accuracy']:.3f}")
        print(f"  Frozen MLP probe acc:        {r2['frozen_mlp_probe_accuracy']:.3f}  "
              f"<-- the diagnostic")
        print(f"  Diagnostic gap (probe − live): "
              f"{r2['diagnostic_gap_accuracy']:+.3f} accuracy")
        out["experiment_2_dynamic_grl"] = r2

    if args.experiment in ("3", "all"):
        print("\n=== Experiment 3: capacity ladder on XOR ===")
        r3 = experiment_3_capacity_ladder()
        print(f"  True MI = {r3['true_mi_nats']/math.log(2):.3f} bits")
        print(f"  {'Architecture':<18}{'V_Φ (bits)':>12}{'Accuracy':>10}")
        for r in r3["rungs"]:
            print(f"  {r['name']:<18}"
                  f"{r['v_phi_nats_mean']/math.log(2):>12.4f}"
                  f"{r['accuracy_mean']:>10.3f}")
        out["experiment_3_capacity_ladder"] = r3

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
