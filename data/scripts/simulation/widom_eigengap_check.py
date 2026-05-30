"""Empirical check of the Lemma 6 spectral predictions for kernel ridge.

Matern-2.5 kernel on uniform points in [0,1]^2 (d=2). Sobolev order
s = nu + d/2 = 3.5, so Mercer eigenvalues decay as mu_k ~ k^{-(2*nu+d)/d}
= k^{-3.5} (Stein 1999 sec. 2.7; Kanagawa et al. 2018 Prop. 4.4).

For S_n = K_n (K_n + lambda I)^{-1} the eigenvalues are s_k = mu_k / (mu_k + lambda).
Cutoff index r = floor(lambda^{-d/(2*nu+d)}) = floor(lambda^{-2/7}). Predictions:
  (a) eigengap exponent: delta_r = s_r - s_{r+1} ~ lambda^{d/(2*nu+d)} = lambda^{2/7}
  (b) bias exponent: integrated tail sum_{k>r} s_k mu_k ~ lambda^{2*nu/(2*nu+d)} = lambda^{5/7}

Honest n caveat: at the project-data size n=348 with length-scale rho=0.15,
the empirical kernel-matrix spectrum has effective rank ~ (1/rho)^d = 44 and
the Mercer asymptotic rate k^{-3.5} only sets in once mu_k is well below
rho^{2*nu+d}. We therefore run the spectral verification at n=2000 (still
under 30 s on an M3 Pro) and report the n=348 slopes as a diagnostic
comparison so the reader can see the finite-sample drift.

References:
  Stein, M. L. (1999) "Interpolation of Spatial Data" Springer, sec. 2.7.
  Kanagawa, Hennig, Sejdinovic, Sriperumbudur (2018) "Gaussian Processes and
    Kernel Methods: A Review on Connections and Equivalences" arXiv:1807.02582.
  Widom, H. (1963) "Asymptotic behavior of the eigenvalues of certain integral
    equations" Trans. AMS 109, 278-295.
  Bach, F. (2018) "Unraveling spectral properties of kernel matrices"
    francisbach.com blog.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
try:
    import _silence  # noqa: F401
except Exception:
    pass

import numpy as np
from pathlib import Path
from sklearn.gaussian_process.kernels import Matern

D, NU, RHO = 2, 2.5, 0.15
LAMBDAS = np.logspace(-4, -1, 10)
PRED_GAP, PRED_BIAS = D / (2 * NU + D), 2 * NU / (2 * NU + D)   # 2/7, 5/7
OUT_DIR = Path(__file__).resolve().parents[3] / "results" / "simulation" / "widom"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run(n: int):
    rng = np.random.default_rng(42)
    coords = rng.uniform(size=(n, D))
    K = Matern(length_scale=RHO, nu=NU)(coords) + 1e-10 * np.eye(n)
    mu = np.linalg.eigvalsh(K)[::-1]
    gaps, tails = [], []
    for lam in LAMBDAS:
        s = mu / (mu + lam)
        r = max(1, min(int(np.floor(lam ** (-PRED_GAP))), n - 1))
        gaps.append(s[r - 1] - s[r])
        tails.append(float(np.sum(s[r:] * mu[r:])))
    lg = np.log(LAMBDAS)
    sg = float(np.polyfit(lg, np.log(np.array(gaps)), 1)[0])
    st = float(np.polyfit(lg, np.log(np.array(tails)), 1)[0])
    return mu, gaps, tails, sg, st


print(f"d={D} nu={NU} rho={RHO}; predicted exponents 2/7={PRED_GAP:.4f}, 5/7={PRED_BIAS:.4f}")
print(f"lambdas = {LAMBDAS.round(5).tolist()}")
results = {}
for n in (348, 2000):
    mu, gaps, tails, sg, st = run(n)
    rank = int((mu > 1e-8).sum())
    print(f"  n={n:>4}  rank(K)={rank:>4}  gap_slope={sg:+.4f}  bias_slope={st:+.4f}")
    results[n] = (gaps, tails, sg, st)
print("Note: empirical slopes deviate from 2/7 and 5/7 because mu_k ~ k^{-3.5}")
print("      asymptote requires k where mu_k << rho^{2*nu+d}; at rho=0.15, n=348")
print("      effective rank is ~ (1/rho)^d = 44 << index range needed.")

gaps, tails, sg, st = results[2000]
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for a, y, slope, pred, ttl in [
        (ax[0], gaps, sg, PRED_GAP, "eigengap delta_r"),
        (ax[1], tails, st, PRED_BIAS, "integrated bias sum s_k mu_k"),
    ]:
        a.loglog(LAMBDAS, y, "o-", label=f"empirical (slope={slope:+.3f})")
        ref = (np.array(y)[0] / LAMBDAS[0] ** pred) * LAMBDAS ** pred
        a.loglog(LAMBDAS, ref, "--", color="gray", label=f"ref slope={pred:+.3f}")
        a.set_xlabel(r"$\lambda$"); a.set_ylabel(ttl); a.legend(); a.grid(alpha=0.3)
    fig.suptitle(f"Widom / Lemma 6 spectral check (n=2000, nu={NU}, d={D}, rho={RHO})")
    fig.tight_layout(); fig.savefig(OUT_DIR / "widom_check.png", dpi=140); plt.close(fig)
    print(f"plot -> {OUT_DIR/'widom_check.png'}")
except Exception as e:
    print(f"plot skipped: {e}")
