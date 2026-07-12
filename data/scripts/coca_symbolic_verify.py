"""Exact symbolic verification of Proposition (CoCA) with SymPy.

Everything is done in a residualized (partialled-on-X) covariance system, where the
proposition's quantities have closed forms and no sampling error. Let Yt, Tt, Ct be
Y, T, C after partialling out the controls X. Structural long model:

    Yt = theta*Tt + gamma*Ct + eps,   eps _|_ (Tt, Ct),
    Var(Tt)=vt, Var(Ct)=vc, Cov(Tt,Ct)=ctc, Var(eps)=ve.

We verify three things exactly.

  (A) Partial-correlation identity and the protective term. With raw correlations
      rho_TC, rho_TX, rho_CX among (T, C, X-projection), the partial correlation
      rho_{TC.X} = (rho_TC - rho_TX*rho_CX)/sqrt((1-rho_TX^2)(1-rho_CX^2)). We show
      the "protective" subtraction rho_TX*rho_CX -> 0 as rho_CX -> 0, so conditioning
      on X removes none of C's association with T when C is orthogonal to X, and we
      report the exact rho_CX -> 0 limit (which is rho_TC/sqrt(1-rho_TX^2), NOT the
      bare rho_TC -- this sharpens the proposition's wording).

  (B) The Cinelli-Hazlett / CCNSS bias bound is EXACT for a single omitted
      confounder: |theta_short - theta_long| equals se_pop(theta_short)*sqrt(df) *
      sqrt( r_D * r_Y / (1 - r_D) ), with r_D = R^2(T~C|X), r_Y = R^2(Y~C|T,X). We
      prove equality of the squared quantities by simplify(lhs - rhs) == 0.

  (C) Benchmark non-identification: with rho_CX = 0 the confounder's partial strength
      r_D = rho_{TC.X}^2 = rho_TC^2/(1-rho_TX^2) is a free function of rho_TC that the
      observed-covariate strength rho_TX does not bound above; symbolically,
      d r_D / d rho_TC != 0 at fixed rho_TX, so no observed statistic pins r_D.

    python data/scripts/coca_symbolic_verify.py
"""
from __future__ import annotations

import json
from pathlib import Path

import sympy as sp

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "coca_symbolic_verify.json"


def part_A():
    rho_TC, rho_TX, rho_CX = sp.symbols("rho_TC rho_TX rho_CX", real=True)
    num = rho_TC - rho_TX * rho_CX
    den = sp.sqrt((1 - rho_TX**2) * (1 - rho_CX**2))
    partial = num / den
    protective_term = rho_TX * rho_CX
    limit0 = sp.simplify(sp.limit(partial, rho_CX, 0))
    prot0 = sp.simplify(protective_term.subs(rho_CX, 0))
    return {
        "partial_corr_expr": str(partial),
        "protective_subtraction": str(protective_term),
        "protective_subtraction_at_rhoCX_0": str(prot0),
        "partial_corr_limit_rhoCX_to_0": str(limit0),
        "note": "limit is rho_TC/sqrt(1-rho_TX^2), not bare rho_TC",
    }


def part_B():
    theta, gamma, vt, vc, ctc, ve = sp.symbols(
        "theta gamma v_t v_c c_tc v_e", positive=True, real=True)
    # allow ctc and gamma any sign
    gamma = sp.symbols("gamma", real=True)
    ctc = sp.symbols("c_tc", real=True)

    # Population moments of the residualized system.
    cov_Y_T = theta * vt + gamma * ctc          # Cov(Yt, Tt)
    cov_Y_C = theta * ctc + gamma * vc          # Cov(Yt, Ct)
    var_Y = theta**2 * vt + gamma**2 * vc + 2 * theta * gamma * ctc + ve

    # Short (omit C) and long (include C) coefficients on T.
    theta_short = cov_Y_T / vt
    # theta_long recovers theta by construction; verify it.
    #   regress Yt on [Tt, Ct]:  solve normal equations.
    b_t, b_c = sp.symbols("b_t b_c", real=True)
    normal = sp.solve([
        sp.Eq(b_t * vt + b_c * ctc, cov_Y_T),
        sp.Eq(b_t * ctc + b_c * vc, cov_Y_C),
    ], [b_t, b_c], dict=True)[0]
    theta_long = sp.simplify(normal[b_t])
    bias = sp.simplify(theta_short - theta_long)

    # Partial-R^2 quantities.
    r_D = ctc**2 / (vt * vc)                                   # R^2(T~C | X)
    var_Y_given_T = var_Y - cov_Y_T**2 / vt                    # Var(Yt | Tt)
    var_C_given_T = vc - ctc**2 / vt                           # Var(Ct | Tt)
    cov_Y_C_given_T = cov_Y_C - cov_Y_T * ctc / vt             # Cov(Yt, Ct | Tt)
    r_Y = cov_Y_C_given_T**2 / (var_Y_given_T * var_C_given_T)  # R^2(Y~C | T, X)

    # se_pop(theta_short) * sqrt(df)  ->  sqrt( Var(Yt|Tt) / vt )  (population form)
    se_sqrt_df = sp.sqrt(var_Y_given_T / vt)
    ch_bound = se_sqrt_df * sp.sqrt(r_D * r_Y / (1 - r_D))

    diff = sp.simplify(bias**2 - ch_bound**2)
    theta_long_check = sp.simplify(theta_long - theta)
    return {
        "theta_short": str(sp.simplify(theta_short)),
        "theta_long_minus_theta": str(theta_long_check),
        "bias": str(bias),
        "CH_bound": str(sp.simplify(ch_bound)),
        "bias_sq_minus_CH_sq_simplified": str(diff),
        "exact_equality_holds": bool(diff == 0),
        "theta_long_recovers_theta": bool(theta_long_check == 0),
    }


def part_C():
    rho_TC, rho_TX = sp.symbols("rho_TC rho_TX", real=True)
    # Orthogonal case rho_CX = 0.
    r_D = rho_TC**2 / (1 - rho_TX**2)
    d_rD_d_rhoTC = sp.simplify(sp.diff(r_D, rho_TC))
    d_rD_d_rhoTX = sp.simplify(sp.diff(r_D, rho_TX))
    return {
        "r_D_when_orthogonal": str(r_D),
        "d_rD_d_rho_TC": str(d_rD_d_rhoTC),
        "d_rD_d_rho_TX": str(d_rD_d_rhoTX),
        "r_D_grows_freely_in_rho_TC": bool(d_rD_d_rhoTC != 0),
        "note": ("r_D depends on rho_TC (the confounder's own strength), which no "
                 "observed statistic bounds once rho_CX=0; the observed rho_TX only "
                 "rescales it. The benchmark 'C no stronger than X_j' has no anchor."),
    }


def main():
    A, B, C = part_A(), part_B(), part_C()

    print("=== (A) partial-correlation identity and the protective term ===")
    print(f"  partial corr = {A['partial_corr_expr']}")
    print(f"  protective subtraction rho_TX*rho_CX at rho_CX=0 : {A['protective_subtraction_at_rhoCX_0']}")
    print(f"  partial corr limit as rho_CX -> 0 : {A['partial_corr_limit_rhoCX_to_0']}")
    print(f"  ({A['note']})")

    print("\n=== (B) Cinelli-Hazlett bias formula is EXACT (single confounder) ===")
    print(f"  theta_long - theta (should be 0)       : {B['theta_long_minus_theta']}")
    print(f"  bias = theta_short - theta_long        : {B['bias']}")
    print(f"  CH_bound                               : {B['CH_bound']}")
    print(f"  simplify(bias^2 - CH_bound^2)          : {B['bias_sq_minus_CH_sq_simplified']}")
    print(f"  EXACT EQUALITY HOLDS                    : {B['exact_equality_holds']}")

    print("\n=== (C) benchmark non-identification when rho_CX = 0 ===")
    print(f"  r_D (orthogonal case)                  : {C['r_D_when_orthogonal']}")
    print(f"  d r_D / d rho_TC                       : {C['d_rD_d_rho_TC']}")
    print(f"  r_D is a free function of rho_TC       : {C['r_D_grows_freely_in_rho_TC']}")

    OUT.write_text(json.dumps({"A": A, "B": B, "C": C}, indent=2))
    print(f"\nwrote {OUT}")
    ok = B["exact_equality_holds"] and B["theta_long_recovers_theta"]
    print(f"\nCORE PROPOSITION (bias bound is exact for a single confounder): "
          f"{'VERIFIED' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
