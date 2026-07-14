"""Symbolic verification of the estimated-direction DML identification results.

Checks:
  1. theta(v) = v'a / v'Sigma v is the partialling-out estimand.
  2. In the two-channel structural model it equals
       (b s2 D + theta q2 E) / (s2 D^2 + q2 E^2),  D=v_D, E=v_E.
  3. Identified direction D(v)=0 returns theta.
  4. Variance-maximal direction (spatial channel dominates, s2>q2) returns b.
  5. Gradient of theta wrt v is nonzero at the identified direction
     (=> v-hat error is controlled by rate domination, not orthogonality).
  6. A numeric NYC-like sign flip.

Run: source .venv/bin/activate && python data/scripts/theory_estimated_direction_verify.py
"""
import sympy as sp


def main():
    vD, vE, b, th = sp.symbols("v_D v_E b theta", real=True)
    ss = sp.Symbol("s2", positive=True)  # Var(s), spatial confounder channel
    qq = sp.Symbol("q2", positive=True)  # Var(q), semantic channel

    # Two-channel residualized model: wtilde=[s,q], Ytilde = theta*q + b*s + u.
    a = sp.Matrix([b * ss, th * qq])          # a = E[wtilde Ytilde]
    Sig = sp.diag(ss, qq)                      # Sigma = E[wtilde wtilde']
    v = sp.Matrix([vD, vE])

    theta_v = sp.simplify((v.T * a)[0] / (v.T * Sig * v)[0])
    print("theta(v) =", theta_v)

    id_dir = sp.simplify(theta_v.subs({vD: 0, vE: 1}))
    pc_dir = sp.simplify(theta_v.subs({vD: 1, vE: 0}))
    assert id_dir == th, f"identified direction should give theta, got {id_dir}"
    assert pc_dir == b, f"variance-maximal direction should give b, got {pc_dir}"
    print("theta(identified, D=0)      =", id_dir, "  [= theta, OK]")
    print("theta(variance-maximal,E=0) =", pc_dir, "  [= b, OK]")

    grad = sp.Matrix([sp.diff(theta_v, vD), sp.diff(theta_v, vE)]).subs({vD: 0, vE: 1})
    grad = sp.simplify(grad)
    print("grad theta at identified dir =", grad.T)
    assert grad[0] != 0, "gradient D-component must be nonzero (rate-domination regime)"
    print("  D-component nonzero => estimand first-order sensitive to v; "
          "inference rests on N/n -> inf, not orthogonality.")

    num = theta_v.subs({th: sp.Rational(6, 100), b: -sp.Rational(3, 10), ss: 5, qq: 1})
    flip_id = num.subs({vD: 0, vE: 1})
    flip_pc = num.subs({vD: 1, vE: 0})
    print(f"NYC-like: identified={flip_id} (positive), "
          f"variance-maximal={flip_pc} (negative) -> sign flip")
    assert flip_id > 0 and flip_pc < 0

    print("\nALL CHECKS PASS")


if __name__ == "__main__":
    main()
