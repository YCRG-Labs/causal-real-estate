"""
Symbolic and numerical verification of the estimation-side math introduced in the
v3 (12-city) paper, which the original verification suite (01-07) did not cover:

  (a) the Robinson partialling-out estimator solves its moment (eq. for theta-hat);
  (b) the partially-linear DML score is Neyman-orthogonal in the nuisances;
  (c) the influence-function / sandwich variance equals the reported SE formula;
  (d) the Gelbach (2016) order-invariant decomposition identity, numerically;
  (e) the Cinelli-Hazlett robustness value formula RV = 1/2(sqrt(f^4+4f^2)-f^2);
  (f) the counterfactual delta-method variance Var(theta*Delta) = Delta^2 Var(theta)
      + theta^2 Var(Delta);
  (g) random-effects (inverse-variance) pooling + the HKSJ variance factor.

SymPy for the symbolic identities, NumPy for the numerical ones.
"""

import json
import sys
from pathlib import Path

import numpy as np
import sympy as sp

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
checks = []


def record(name, ok, detail):
    checks.append({"check": name, "pass": bool(ok), "detail": detail})


# (a) Robinson estimator solves the moment sum_i (Ytil - theta Ttil) Ttil = 0
def check_robinson():
    th = sp.symbols("theta")
    Yt = sp.symbols("Yt1 Yt2 Yt3"); Tt = sp.symbols("Tt1 Tt2 Tt3")
    moment = sum((Yt[i] - th * Tt[i]) * Tt[i] for i in range(3))
    sol = sp.solve(moment, th)[0]
    formula = (sum(Tt[i] * Yt[i] for i in range(3))) / (sum(Tt[i] ** 2 for i in range(3)))
    ok = sp.simplify(sol - formula) == 0
    record("(a) Robinson estimator solves its moment",
           ok, "theta_hat = sum(Ttil*Ytil)/sum(Ttil^2) recovered symbolically")
    return th, Yt, Tt


# (b) Neyman orthogonality of the PLR score in the nuisances (l, m)
def check_neyman():
    # eps, nu are the structural error and treatment residual (mean-zero given W);
    # dl, dm are nuisance perturbation directions; th0 the truth.
    eps, nu, th0, dl, dm, r, l0, m0 = sp.symbols("eps nu theta0 dl dm r l0 m0")
    Y = l0 + th0 * nu + eps     # Y = l0(W) + th0*nu + eps,  eps mean-zero given W
    T = m0 + nu                 # T = m0(W) + nu,            nu  mean-zero given W
    l = l0 + r * dl
    m = m0 + r * dm
    psi = (Y - l - th0 * (T - m)) * (T - m)
    dpsi = sp.expand(sp.diff(psi, r).subs(r, 0))
    # Every term of d/dr psi|_0 is first order in the mean-zero residuals (nu, eps):
    # zeroing nu and eps annihilates it, so each term has expectation 0 under
    # E[nu|W] = E[eps|W] = 0 (and eps _||_ nu), giving Neyman orthogonality.
    residual = sp.expand(dpsi.subs({nu: 0, eps: 0}))
    ok = (residual == 0)
    record("(b) PLR score is Neyman-orthogonal",
           ok, f"d/dr psi|_0 = {dpsi}; setting nu=eps=0 leaves {residual}, so every "
               "term carries a mean-zero residual factor and E[d/dr psi]=0")


# (c) sandwich / IF variance equals the reported SE^2 formula
def check_sandwich():
    th = sp.symbols("theta")
    Yt = sp.symbols("Yt1 Yt2 Yt3"); Tt = sp.symbols("Tt1 Tt2 Tt3")
    psi = [(Yt[i] - th * Tt[i]) * Tt[i] for i in range(3)]
    jac = [sp.diff(p, th) for p in psi]
    # dpsi/dtheta = -Ttil^2
    jac_ok = all(sp.simplify(jac[i] + Tt[i] ** 2) == 0 for i in range(3))
    # sandwich Var = (sum dpsi/dtheta)^-2 * sum psi^2  ==  sum eps^2 Ttil^2 / (sum Ttil^2)^2
    bread = sum(jac)
    meat = sum(p ** 2 for p in psi)
    sandwich = meat / bread ** 2
    reported = sum(((Yt[i] - th * Tt[i]) ** 2 * Tt[i] ** 2) for i in range(3)) / \
        (sum(Tt[i] ** 2 for i in range(3))) ** 2
    ok = jac_ok and sp.simplify(sandwich - reported) == 0
    record("(c) IF/sandwich variance = reported SE^2 formula",
           ok, "dpsi/dtheta=-Ttil^2; sandwich = sum(resid^2 Ttil^2)/(sum Ttil^2)^2")


# (d) Gelbach order-invariant decomposition: theta_short - theta_long
#     = sum_g (full-model coef on g) * (coef on T from regressing g on [T, W])
def check_gelbach(seed=0):
    rng = np.random.default_rng(seed)
    n = 4000
    W = rng.normal(size=(n, 2))                       # base controls
    G = np.column_stack([0.6 * W[:, 0] + rng.normal(size=n),
                         0.4 * W[:, 1] + rng.normal(size=n)])  # geo block
    T = 0.5 * W[:, 0] + 0.3 * G[:, 0] - 0.2 * G[:, 1] + rng.normal(size=n)
    Y = 1.0 * T + 0.8 * W[:, 0] + 0.5 * G[:, 0] + 0.9 * G[:, 1] + rng.normal(size=n)

    def ols(X, y):
        Xc = np.column_stack([np.ones(len(y)), X])
        return np.linalg.lstsq(Xc, y, rcond=None)[0]

    th_short = ols(np.column_stack([T, W]), Y)[1]            # coef on T, no geo
    full = ols(np.column_stack([T, W, G]), Y)                # [const,T,W1,W2,G1,G2]
    th_long = full[1]
    delta = th_short - th_long
    # Gelbach contributions of the geo block
    contrib = 0.0
    for j in range(G.shape[1]):
        delta_g = full[1 + 1 + W.shape[1] + j]              # full-model coef on G_j
        gamma_g = ols(np.column_stack([T, W]), G[:, j])[1]  # coef on T in aux reg
        contrib += delta_g * gamma_g
    ok = abs(delta - contrib) < 1e-6
    record("(d) Gelbach decomposition identity (numerical)",
           ok, f"theta_short-theta_long = {delta:.6f}; sum of Gelbach "
               f"contributions = {contrib:.6f}; |diff|={abs(delta-contrib):.2e}")


# (e) Cinelli-Hazlett robustness value: RV = 1/2(sqrt(f^4+4f^2)-f^2) solves
#     RV^2/(1-RV) = f^2  (equal-strength confounder that kills the estimate)
def check_rv():
    f = sp.symbols("f", positive=True)
    RV = (sp.sqrt(f**4 + 4 * f**2) - f**2) / 2
    ok = sp.simplify(RV**2 / (1 - RV) - f**2) == 0
    # numerical spot-check at the SF partial-f (RV=0.14)
    fval = float(sp.sqrt(sp.Rational(14, 100)**2 / (1 - sp.Rational(14, 100))))
    rv_num = float(RV.subs(f, fval))
    record("(e) Cinelli-Hazlett robustness-value formula",
           ok and abs(rv_num - 0.14) < 1e-6,
           f"RV=1/2(sqrt(f^4+4f^2)-f^2) satisfies RV^2/(1-RV)=f^2; "
           f"back-solved RV at SF f={fval:.4f} is {rv_num:.4f}")


# (f) counterfactual delta-method variance
def check_delta_method():
    th, D = sp.symbols("theta Delta")
    vth, vD = sp.symbols("Var_theta Var_Delta", positive=True)
    delta = th * D
    grad = sp.Matrix([sp.diff(delta, th), sp.diff(delta, D)])   # = [Delta, theta]
    Sigma = sp.diag(vth, vD)                                    # theta, Delta indep
    var = (grad.T * Sigma * grad)[0]
    reported = D**2 * vth + th**2 * vD
    ok = sp.simplify(var - reported) == 0
    record("(f) counterfactual delta-method variance",
           ok, "grad(theta*Delta)=(Delta,theta); g'Sigma g = Delta^2 Var(theta) + "
               "theta^2 Var(Delta)")


# (g) random-effects inverse-variance pooling + HKSJ variance factor
def check_meta(seed=1):
    rng = np.random.default_rng(seed)
    k = 12
    theta = rng.normal(0.1, 0.15, size=k)
    s2 = (rng.uniform(0.02, 0.06, size=k)) ** 2
    tau2 = 0.02
    w = 1.0 / (s2 + tau2)
    pooled = np.sum(w * theta) / np.sum(w)
    # HKSJ: q = sum w (theta - pooled)^2 / (k-1); Var_HKSJ = q / sum w
    q = np.sum(w * (theta - pooled) ** 2) / (k - 1)
    var_hksj = q / np.sum(w)
    var_iv = 1.0 / np.sum(w)
    # checks: pooled is the inverse-variance weighted mean; HKSJ inflates by q
    ok = (abs(pooled - np.average(theta, weights=w)) < 1e-12
          and abs(var_hksj - q * var_iv) < 1e-12
          and var_hksj > 0)
    record("(g) RE inverse-variance pooling + HKSJ factor",
           ok, f"pooled = weighted mean; Var_HKSJ = q*Var_IV with q={q:.3f}")


def main():
    check_robinson()
    check_neyman()
    check_sandwich()
    check_gelbach()
    check_rv()
    check_delta_method()
    check_meta()
    out = RESULTS / "08_estimation_identities.json"
    allok = all(c["pass"] for c in checks)
    out.write_text(json.dumps({"verdict": "PASS" if allok else "FAIL",
                               "checks": checks}, indent=2))
    for c in checks:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['check']}")
    if not allok:
        print(f"[08] FAIL — see {out}")
        return 1
    print(f"[08] PASS — {len(checks)}/{len(checks)} estimation identities verified. Wrote {out}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
