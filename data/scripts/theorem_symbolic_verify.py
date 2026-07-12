"""Symbolic verification of the algebraic cores of Theorem 1 (identification) and
Theorem 2 (decomposition consistency) with SymPy.

Model. Orthonormal, unit-variance exogenous basis (eta, ls, ld, eps):
  eta  = location-orthogonal semantic variation in the treatment
  ls   = SMOOTH location channel (spanned by a coordinate basis)
  ld   = DISCRETE location channel (borough/neighborhood/school names; NOT spanned
         by a coordinate spline)
  eps  = outcome noise
Each observed variable is a linear combination, so covariance is the dot product of
coefficient vectors and residualizing on a set of basis directions is coordinate
deletion. Structural equations:
  U = a*ls + b*ld                       (unobserved spatial confounder, = h(location))
  T = c*ls + d*ld + eta                 (treatment: text encodes location + semantics)
  Y = theta*T + beta*U + eps            (outcome)

We verify:
  (T1a) With the ENRICHED basis L = {ls, ld}, the Robinson estimand recovers theta.
  (T1b) With the coordinate-only basis L = {ls}, it is biased by beta*b*d/(d^2+1),
        which is nonzero exactly when the discrete channel matters (b,d != 0). This
        is the necessity of the basis-richness assumption (Theorem 1, As. basis).
  (T2)  theta_base - theta_geo equals the Gelbach / omitted-variable-bias term for
        the location block, exactly (Theorem 2, linear case).
"""
from __future__ import annotations
import json
from pathlib import Path
import sympy as sp

OUT = Path(__file__).resolve().parents[2] / "results" / "theorem_symbolic_verify.json"

theta, beta, a, b, c, d, gs, gd = sp.symbols("theta beta a b c d gamma_s gamma_d", real=True)
# basis order: [eta, ls, ld, eps]
eta = sp.Matrix([1, 0, 0, 0])
ls  = sp.Matrix([0, 1, 0, 0])
ld  = sp.Matrix([0, 0, 1, 0])
eps = sp.Matrix([0, 0, 0, 1])

def cov(A, B):  # orthonormal unit-variance basis => Cov = dot product
    return (A.T * B)[0, 0]

def resid(A, drop):  # residual of A after projecting on span of coordinate directions `drop`
    A = A.copy()
    for i in drop:
        A[i] = 0
    return A

def robinson(Y, T, Lcoords):
    Ty, Tt = resid(Y, Lcoords), resid(T, Lcoords)
    return sp.simplify(cov(Ty, Tt) / cov(Tt, Tt))

# ---- Theorem 1: identification + basis-richness necessity ----
U = a*ls + b*ld
T = c*ls + d*ld + eta
Y = theta*T + beta*U + eps

th_enriched = robinson(Y, T, [1, 2])   # L = {ls, ld}  (coords 1,2)
th_coordonly = robinson(Y, T, [1])     # L = {ls}      (coord 1 only; misses ld)
bias = sp.simplify(th_coordonly - theta)
bias_target = sp.simplify(beta*b*d/(d**2 + 1))

# ---- Theorem 2: decomposition = Gelbach/OVB term (linear, W_0 empty) ----
Y2 = theta*T + gs*ls + gd*ld + eps
th_geo = robinson(Y2, T, [1, 2])       # full: control the location block
th_base = robinson(Y2, T, [])          # base: omit the location block
delta = sp.simplify(th_base - th_geo)
# Gelbach / OVB term: sum_k [coef of L_k on T] * [coef of L_k in full model]
VarT = cov(T, T)
ovb = sp.simplify((cov(ls, T)/VarT)*gs + (cov(ld, T)/VarT)*gd)

res = {
    "T1a_enriched_recovers_theta": bool(sp.simplify(th_enriched - theta) == 0),
    "T1a_value": str(th_enriched),
    "T1b_coordonly_biased": str(bias),
    "T1b_bias_equals_beta_b_d_over_d2p1": bool(sp.simplify(bias - bias_target) == 0),
    "T1b_bias_nonzero_iff_discrete_channel": "bias = 0  <=>  b*d = 0 (no discrete leakage or no discrete treatment content)",
    "T2_delta": str(delta),
    "T2_ovb": str(ovb),
    "T2_delta_equals_gelbach_ovb": bool(sp.simplify(delta - ovb) == 0),
}

print("=== Theorem 1 (identification) ===")
print(f"  enriched basis L={{ls,ld}}: Robinson estimand = {res['T1a_value']}  -> equals theta: {res['T1a_enriched_recovers_theta']}")
print(f"  coordinate-only L={{ls}}:   bias = {res['T1b_coordonly_biased']}")
print(f"  bias == beta*b*d/(d^2+1):  {res['T1b_bias_equals_beta_b_d_over_d2p1']}   ({res['T1b_bias_nonzero_iff_discrete_channel']})")
print("=== Theorem 2 (decomposition = Gelbach/OVB) ===")
print(f"  theta_base - theta_geo = {res['T2_delta']}")
print(f"  Gelbach/OVB term       = {res['T2_ovb']}")
print(f"  EQUAL (exact):           {res['T2_delta_equals_gelbach_ovb']}")

OUT.write_text(json.dumps(res, indent=2, default=str))
ok = (res["T1a_enriched_recovers_theta"] and res["T1b_bias_equals_beta_b_d_over_d2p1"]
      and res["T2_delta_equals_gelbach_ovb"])
print(f"\nALGEBRAIC CORES VERIFIED: {ok}")
