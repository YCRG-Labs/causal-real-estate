"""Widom finite-sample regime diagnostic for Lemma 6 of dk_full_proof_lemmas.tex.

Lemma 6 predicts: at the BBP cutoff index r = floor(lambda^{-d/(2nu+d)}), the
eigengap delta_r = lambda_r(S_n) - lambda_{r+1}(S_n) scales as lambda^{d/(2nu+d)}.
For Matern-2.5, d=2 this exponent is 2/7 = 0.2857. The empirical check at
n=348, rho=0.15 returns slope ~0.80, and even n=2000 returns ~0.70. A JBES
referee will ask which rate is operative at our sample size.

This script maps out the (n, rho) regime so we can answer that question
honestly. Three diagnostic outputs:

  (i)  Eigengap and tail-bias slopes across rho in {0.01, 0.03, 0.05, 0.10,
       0.15, 0.25} and n in {348, 1000, 2000}. Used to identify where the
       polynomial (asymptotic) regime sets in.
  (ii) log mu_k vs log k for k = 1..min(n, 200) at each rho. Visualizes the
       transition from exponential (pre-asymptotic / RKHS-dominated) to
       polynomial (Widom / Mercer-asymptotic) decay.
  (iii) Effective rank (1/rho)^d compared against the smallest index where
       the empirical log-log slope of mu_k matches the predicted -(2nu+d)/d
       = -3.5.

References:
  Widom, H. (1963) Trans. AMS 109, 278-295.
  Stein, M. L. (1999) Interpolation of Spatial Data, Springer, sec. 2.7.
  Rasmussen, C. E. and Williams, C. K. I. (2006) GP for ML, Ch. 4.
  Kanagawa, Hennig, Sejdinovic, Sriperumbudur (2018) arXiv:1807.02582.
  Bach, F. (2018) "Spectrum of kernel matrices II," francisbach.com.

Run: python data/scripts/diagnostics/diag_widom_rho_sweep.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
try:
    import _silence
except Exception:
    pass

import numpy as np
from pathlib import Path
from sklearn.gaussian_process.kernels import Matern

D, NU = 2, 2.5
RHOS = (0.01, 0.03, 0.05, 0.10, 0.15, 0.25)
NS = (348, 1000, 2000)
LAMBDAS = np.logspace(-4, -1, 10)
PRED_GAP = D / (2 * NU + D)
PRED_BIAS = 2 * NU / (2 * NU + D)
PRED_MU_SLOPE = -(2 * NU + D) / D

ROOT = Path(__file__).resolve().parents[3]
OUT_JSON = ROOT / "results" / "diagnostics" / "widom_rho_sweep.json"
OUT_PNG = ROOT / "results" / "diagnostics" / "widom_rho_sweep.png"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)


def spectrum(n: int, rho: float, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coords = rng.uniform(size=(n, D))
    K = Matern(length_scale=rho, nu=NU)(coords) + 1e-10 * np.eye(n)
    return np.linalg.eigvalsh(K)[::-1]


def slopes(mu: np.ndarray) -> tuple[float, float]:
    n = len(mu)
    gaps, tails = [], []
    for lam in LAMBDAS:
        s = mu / (mu + lam)
        r = max(1, min(int(np.floor(lam ** (-PRED_GAP))), n - 1))
        gaps.append(s[r - 1] - s[r])
        tails.append(float(np.sum(s[r:] * mu[r:])))
    lg = np.log(LAMBDAS)
    sg = float(np.polyfit(lg, np.log(np.array(gaps)), 1)[0])
    st = float(np.polyfit(lg, np.log(np.array(tails)), 1)[0])
    return sg, st


def asymptotic_index(mu: np.ndarray, tol: float = 0.5) -> int:
    """Smallest k>=5 where the local log-log slope of mu hits PRED_MU_SLOPE within tol."""
    n = len(mu)
    pos = mu > 1e-12
    mu = mu[pos]
    if len(mu) < 20:
        return -1
    lk = np.log(np.arange(1, len(mu) + 1))
    lmu = np.log(mu)
    win = 10
    for k in range(5, len(mu) - win):
        local = np.polyfit(lk[k:k + win], lmu[k:k + win], 1)[0]
        if abs(local - PRED_MU_SLOPE) < tol:
            return int(k + win // 2)
    return -1


print(f"d={D} nu={NU}; predicted: gap_exp={PRED_GAP:.4f}, bias_exp={PRED_BIAS:.4f}, mu_k slope={PRED_MU_SLOPE}")
print(f"sweep: rho in {RHOS}, n in {NS}")
print(f"lambdas = {LAMBDAS.round(5).tolist()}")
print()

results = {
    "predicted_gap_exp": PRED_GAP,
    "predicted_bias_exp": PRED_BIAS,
    "predicted_mu_loglog_slope": PRED_MU_SLOPE,
    "lambdas": LAMBDAS.tolist(),
    "ns": list(NS),
    "rhos": list(RHOS),
    "sweep": [],
}

spec_cache: dict[tuple[int, float], np.ndarray] = {}
print(f"{'n':>5}  {'rho':>6}  {'gap_slope':>10}  {'bias_slope':>10}  {'eff_rank':>9}  {'(1/rho)^d':>10}  {'k_asym':>7}")
print("-" * 78)
for n in NS:
    for rho in RHOS:
        mu = spectrum(n, rho)
        spec_cache[(n, rho)] = mu
        sg, st = slopes(mu)
        eff_rank_emp = int((mu > mu[0] * 1e-6).sum())
        eff_rank_theory = (1.0 / rho) ** D
        k_asym = asymptotic_index(mu)
        results["sweep"].append({
            "n": n,
            "rho": rho,
            "gap_slope": sg,
            "bias_slope": st,
            "eff_rank_empirical": eff_rank_emp,
            "eff_rank_theory_inv_rho_d": eff_rank_theory,
            "k_asymptotic_onset": k_asym,
            "mu_top20": mu[:20].tolist(),
        })
        print(f"{n:>5}  {rho:>6.3f}  {sg:>+10.4f}  {st:>+10.4f}  {eff_rank_emp:>9d}  {eff_rank_theory:>10.1f}  {k_asym:>7d}")

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nJSON -> {OUT_JSON}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    colors = {348: "C0", 1000: "C1", 2000: "C2"}
    for n in NS:
        rhos_n = [r["rho"] for r in results["sweep"] if r["n"] == n]
        gs_n = [r["gap_slope"] for r in results["sweep"] if r["n"] == n]
        bs_n = [r["bias_slope"] for r in results["sweep"] if r["n"] == n]
        ax[0].plot(rhos_n, gs_n, "o-", color=colors[n], label=f"gap n={n}")
        ax[0].plot(rhos_n, bs_n, "s--", color=colors[n], alpha=0.6, label=f"bias n={n}")
    ax[0].axhline(PRED_GAP, color="k", ls=":", label=f"asymp gap 2/7={PRED_GAP:.3f}")
    ax[0].axhline(PRED_BIAS, color="gray", ls=":", label=f"asymp bias 5/7={PRED_BIAS:.3f}")
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"length scale $\rho$")
    ax[0].set_ylabel("empirical log-log slope vs $\lambda$")
    ax[0].set_title("Eigengap and bias-tail slopes (Lemma 6)")
    ax[0].legend(fontsize=7, ncol=2, loc="best")
    ax[0].grid(alpha=0.3)

    k_max = 200
    for rho in RHOS:
        mu = spec_cache[(2000, rho)][:k_max]
        ks = np.arange(1, len(mu) + 1)
        pos = mu > 0
        ax[1].loglog(ks[pos], mu[pos], "-", label=f"rho={rho}")
        eff = (1.0 / rho) ** D
        if eff < k_max:
            ax[1].axvline(eff, color=ax[1].lines[-1].get_color(), ls=":", alpha=0.5)
    ref_k = np.arange(5, k_max)
    ref_mu = ref_k ** PRED_MU_SLOPE
    ref_mu = ref_mu * (spec_cache[(2000, 0.01)][20] / ref_mu[15])
    ax[1].loglog(ref_k, ref_mu, "k--", lw=1.2, label=f"Widom slope {PRED_MU_SLOPE}")
    ax[1].set_xlabel("eigenvalue index $k$")
    ax[1].set_ylabel(r"$\mu_k$ (n=2000)")
    ax[1].set_title("Spectrum transition: exponential -> polynomial")
    ax[1].legend(fontsize=7, ncol=2, loc="best")
    ax[1].grid(alpha=0.3, which="both")

    fig.suptitle(f"Widom finite-sample regime diagnostic; Matern nu={NU}, d={D}")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f"PNG  -> {OUT_PNG}")
except Exception as e:
    print(f"plot skipped: {e}")
