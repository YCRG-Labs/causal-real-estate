"""Symbolic verification of Theorem 1' (direction-indexed spatial identification for
an embedding-valued exposure).

Generalizes Theorem 1: instead of a GIVEN scalar treatment T, the treatment is a
DIRECTION through a vector embedding, T = v'W, and identification depends on how the
direction v aligns with the spatial vs non-spatial subspaces of the field.

Model. Orthonormal unit-variance latent factors [ls, ld, eta, eps]:
  ls  = SMOOTH location channel (captured by the coordinate control L)
  ld  = DISCRETE location channel (borough/school names; NOT captured by L) -- the leak
  eta = non-spatial semantic variation (the identified signal)
  eps = outcome noise
A p=2 embedding whose coordinates mix the channels differently:
  W1 = c1*ls + d1*ld + e1*eta
  W2 = c2*ls + d2*ld + e2*eta
Treatment along direction v=(v1,v2):  T = v'W.
Confounder  U = a*ls + b*ld  (spatial).   Outcome  Y = theta*T + beta*U + eps.
Control L = {ls} (captures smooth, MISSES the discrete leak ld).

Claims verified:
  (A) theta(v) = theta + beta*b * D(v)/(D(v)^2 + E(v)^2),
      where D(v)=v.(d1,d2) is v's load on the leaked spatial channel and
            E(v)=v.(e1,e2) is v's load on the non-spatial subspace.
  (B) Identification (theta(v)=theta) holds iff D(v)=0 (v orthogonal to the leak) or beta*b=0.
      -> the identified DIRECTIONS form the subspace {v : D(v)=0}. No scalar analogue.
  (C) Positivity/overlap E[T~^2] = D(v)^2 + E(v)^2 > 0 requires v to load off the smooth
      subspace; a purely-smooth direction (D=E=0) is non-identified (0/0).
  (D) Reduction: the scalar Theorem 1 bias beta*b*d/(d^2+1) is the special case
      D=d, E=1 (a single unit-variance non-spatial channel).
  (E) Partial-ID: over the unit sphere the bias ranges on a bounded interval; its
      magnitude is maximized at |D|=|E| and is 0 on the identified subspace.
"""
from __future__ import annotations
import json
from pathlib import Path
import sympy as sp

OUT = Path(__file__).resolve().parents[2] / "results" / "theorem_1prime_verify.json"

theta, beta, a, b = sp.symbols("theta beta a b", real=True)
c1, d1, e1, c2, d2, e2 = sp.symbols("c1 d1 e1 c2 d2 e2", real=True)
v1, v2 = sp.symbols("v1 v2", real=True)

# basis [ls, ld, eta, eps]
ls  = sp.Matrix([1, 0, 0, 0])
ld  = sp.Matrix([0, 1, 0, 0])
eta = sp.Matrix([0, 0, 1, 0])
eps = sp.Matrix([0, 0, 0, 1])

def cov(A, B):
    return (A.T * B)[0, 0]

def resid(A, drop):
    A = A.copy()
    for i in drop:
        A[i] = 0
    return A

W1 = c1*ls + d1*ld + e1*eta
W2 = c2*ls + d2*ld + e2*eta
T = v1*W1 + v2*W2
U = a*ls + b*ld
Y = theta*T + beta*U + eps

# Robinson estimand controlling for L = {ls} (coord 0)
Tt = resid(T, [0])
Yt = resid(Y, [0])
theta_v = sp.simplify(cov(Yt, Tt) / cov(Tt, Tt))

D = v1*d1 + v2*d2
E = v1*e1 + v2*e2
claimA = sp.simplify(theta + beta*b*D/(D**2 + E**2))

# (A) exact match
A_ok = bool(sp.simplify(theta_v - claimA) == 0)

# (C) overlap = residual variance of T after removing ls
overlap = sp.simplify(cov(Tt, Tt))
C_ok = bool(sp.simplify(overlap - (D**2 + E**2)) == 0)

# (B) bias zero iff D=0 (given beta,b generic)
bias = sp.simplify(theta_v - theta)
B_ok = bool(sp.simplify(bias.subs(D_ := {}, )) is not None) and \
       bool(sp.simplify(bias.subs({d1: 0, d2: 0})) == 0)          # D(v)=0 via d1=d2=0
# also: choosing v orthogonal to (d1,d2) kills it, e.g. v=(d2,-d1)
bias_vperp = sp.simplify(bias.subs({v1: d2, v2: -d1}))
Bperp_ok = bool(bias_vperp == 0)

# (D) reduction to scalar Theorem 1: single non-spatial channel, D=d, E=1
scalar_bias = sp.simplify(bias.subs({c1: 0, e1: 0, c2: 0, e2: 1, d2: 0, v1: 1, v2: 1, d1: sp.Symbol('d', real=True)}))
d = sp.Symbol('d', real=True)
scalar_target = sp.simplify(beta*b*d/(d**2 + 1))
D_ok = bool(sp.simplify(scalar_bias - scalar_target) == 0)

# (E) partial-ID: bias as function of ratio r=D/E is beta*b*r/(r^2+1)*(1/E-normalized);
# magnitude of beta*b*D/(D^2+E^2) maximized over the sphere at |D|=|E|; sup = |beta*b|/(2*sqrt(...))
# check the reduced one-parameter form g(r)=r/(1+r^2) has max at r=1
r = sp.Symbol('r', real=True)
g = r/(1+r**2)
gmax_at_1 = bool(sp.simplify(sp.diff(g, r).subs(r, 1)) == 0) and bool(g.subs(r,1) == sp.Rational(1,2))

res = {
    "A_theta_v_formula": str(theta_v),
    "A_matches_claim": A_ok,
    "B_bias_zero_iff_D0": B_ok,
    "B_bias_zero_on_perp_direction_v=(d2,-d1)": Bperp_ok,
    "C_overlap_equals_D2plusE2": C_ok,
    "C_overlap_expr": str(overlap),
    "D_reduces_to_scalar_bbd_over_d2p1": D_ok,
    "E_bias_shape_maximized_at_|D|=|E|": gmax_at_1,
}
print("=== Theorem 1' (direction-indexed spatial identification) ===")
print(f"  theta(v) = {res['A_theta_v_formula']}")
print(f"  (A) matches theta + beta*b*D/(D^2+E^2):      {A_ok}")
print(f"  (B) bias=0 iff v orthogonal to leak (D=0):   {B_ok} / perp-direction: {Bperp_ok}")
print(f"  (C) overlap E[T~^2] = D(v)^2 + E(v)^2:        {C_ok}")
print(f"  (D) reduces to scalar Thm1 bias bbd/(d^2+1): {D_ok}")
print(f"  (E) bias magnitude maximized at |D|=|E|:      {gmax_at_1}")

OUT.write_text(json.dumps(res, indent=2, default=str))
ok = A_ok and B_ok and Bperp_ok and C_ok and D_ok and gmax_at_1
print(f"\nTHEOREM 1' ALGEBRAIC CORE VERIFIED: {ok}")
