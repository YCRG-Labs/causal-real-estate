"""Numerical confirmation of Theorem 1' with real (flexible-ML) DML on finite samples.

Instantiates the direction-indexed spatial-identification model and checks that a
cross-fit gradient-boosting DML estimator recovers the closed-form
    theta(v) = theta + beta*b * D(v)/(D(v)^2 + E(v)^2)
across a sweep of treatment directions v, and in particular that identification
(theta_hat(v) = theta) holds exactly on the non-spatial subspace D(v)=0.

Latent factors (iid N(0,1)): ls (smooth loc, captured by control L), ld (discrete
loc leak, NOT captured), eta (non-spatial signal), eps.
  W1 = c1 ls + d1 ld + e1 eta ;  W2 = c2 ls + d2 ld + e2 eta
  U  = a ls + b ld ;  T = v'W ;  Y = theta T + beta U + eps
Control L = ls only (smooth captured, discrete leak ld missed).
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from sklearn.model_selection import KFold
sys.path.insert(0, "data/scripts")
from booster import make_regressor

rng = np.random.default_rng(20260712)
n = 8000
theta, beta, a, b = 0.30, 0.80, 0.5, 0.9
c1, d1, e1 = 0.7, 0.6, 0.3
c2, d2, e2 = 0.2, 0.1, 0.9

ls = rng.standard_normal(n); ld = rng.standard_normal(n)
eta = rng.standard_normal(n); eps = rng.standard_normal(n)
W1 = c1*ls + d1*ld + e1*eta
W2 = c2*ls + d2*ld + e2*eta
U = a*ls + b*ld
L = ls.reshape(-1, 1)  # control: smooth channel only

def dml(Tvec):
    kf = KFold(5, shuffle=True, random_state=1)
    Yr = np.zeros(n); Tr = np.zeros(n)
    Y = theta*Tvec + beta*U + 0.3*eps
    for tr, te in kf.split(np.arange(n)):
        my = make_regressor(n_estimators=150, max_depth=3, learning_rate=0.1, random_state=42, n_jobs=1)
        my.fit(L[tr], Y[tr]); Yr[te] = Y[te] - my.predict(L[te])
        mt = make_regressor(n_estimators=150, max_depth=3, learning_rate=0.1, random_state=42, n_jobs=1)
        mt.fit(L[tr], Tvec[tr]); Tr[te] = Tvec[te] - mt.predict(L[te])
    return float(np.mean(Tr*Yr)/np.mean(Tr**2))

def closed_form(v1, v2):
    D = v1*d1 + v2*d2; E = v1*e1 + v2*e2
    return theta + beta*b*D/(D**2 + E**2), D, E

print(f"true theta={theta}; leak (d1,d2)=({d1},{d2}); non-spatial (e1,e2)=({e1},{e2})")
print(f"{'angle':>6} {'v':>16} {'D(v)':>7} {'E(v)':>7} {'theta_hat':>10} {'closed':>8} {'|diff|':>7}")
# identified direction: v perp to (d1,d2) => v=(d2,-d1) normalized
vid = np.array([d2, -d1]); vid /= np.linalg.norm(vid)
rows = []
for deg in [0, 20, 40, 60, 80, 100, 140, 180]:
    aa = np.radians(deg); v1, v2 = np.cos(aa), np.sin(aa)
    T = v1*W1 + v2*W2
    th = dml(T); cf, D, E = closed_form(v1, v2)
    rows.append((deg, th, cf))
    print(f"{deg:>6} ({v1:+.2f},{v2:+.2f}) {D:>7.3f} {E:>7.3f} {th:>+10.4f} {cf:>+8.4f} {abs(th-cf):>7.4f}")
# the identified direction D=0
T = vid[0]*W1 + vid[1]*W2
th_id = dml(T); cf_id, D_id, E_id = closed_form(*vid)
print(f"\nIDENTIFIED direction v=(d2,-d1)/||.|| = ({vid[0]:+.3f},{vid[1]:+.3f}): D={D_id:+.4f}")
print(f"  theta_hat = {th_id:+.4f}  (true theta={theta})  closed_form={cf_id:+.4f}  -> identification holds: {abs(th_id-theta)<0.02}")
maxdiff = max(abs(th-cf) for _, th, cf in rows)
print(f"\nmax |theta_hat - closed_form| across sweep = {maxdiff:.4f}")
print(f"THEOREM 1' NUMERICAL CONFIRMATION: {maxdiff < 0.03 and abs(th_id-theta) < 0.02}")
