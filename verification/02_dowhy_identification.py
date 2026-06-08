"""
Independent corroboration of Proposition 1 / Theorem 1 / Theorem 2 via an
explicit backdoor-criterion check on the SCM graph (networkx), plus a numerical
sanity regression. (Ported off DoWhy/pgmpy, which are not installable in the
PEP-668 environment; networkx implements the same graph semantics.)

1. Under the *true* SCM (no T -> Y edge): there is no directed path T -> Y, so
   the causal effect is structurally zero. Corroborates Proposition 1 / Corollary 1.

2. Under a hypothetical SCM with T -> Y added: the set {L, X, C} satisfies Pearl's
   backdoor criterion relative to (T, Y) -- it contains no descendant of T and it
   d-separates T from Y in the backdoor graph (T's outgoing edges deleted) -- and
   it equals the parent set of T, the canonical sufficient adjustment. Matches
   Theorem 1 / Theorem 2.

3. Numerical sanity: OLS of Y on (T, L, X, C) on data simulated from the true SCM
   returns a coefficient on T near zero.
"""

import json
import sys
from pathlib import Path

import numpy as np
import networkx as nx

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

SCM_EDGES = [
    ("L", "X"), ("L", "C"), ("L", "T"), ("X", "T"), ("C", "T"),
    ("L", "Y"), ("X", "Y"), ("C", "Y"),
]
Z = {"L", "X", "C"}


def true_scm():
    g = nx.DiGraph(); g.add_nodes_from(["L", "X", "C", "T", "Y"]); g.add_edges_from(SCM_EDGES)
    return g


def hypothetical_scm():
    g = true_scm(); g.add_edge("T", "Y")
    return g


def valid_backdoor(g, treatment, outcome, z):
    """Pearl's backdoor criterion: z has no descendant of treatment, and z
    d-separates treatment from outcome in the graph with treatment's outgoing
    edges removed."""
    if z & nx.descendants(g, treatment):
        return False, "adjustment set contains a descendant of the treatment"
    g_bd = g.copy()
    g_bd.remove_edges_from(list(g.out_edges(treatment)))
    if not nx.is_d_separator(g_bd, {treatment}, {outcome}, set(z)):
        return False, "adjustment set does not block all backdoor paths"
    return True, "valid"


def synthetic_data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    L = rng.normal(size=n)
    X = 0.7 * L + rng.normal(scale=0.5, size=n)
    C = 0.5 * L + rng.normal(scale=0.5, size=n)
    T = 0.4 * L + 0.3 * X + 0.2 * C + rng.normal(scale=0.5, size=n)
    Y = 1.0 * L + 0.6 * X + 0.4 * C + rng.normal(scale=0.5, size=n)   # no T term
    return L, X, C, T, Y


def main():
    gt, gh = true_scm(), hypothetical_scm()

    # (1) zero causal effect under the true SCM: no directed path T -> Y
    declares_zero = not nx.has_path(gt, "T", "Y")

    # (2) backdoor validity of {L,X,C} under the hypothetical SCM
    is_valid, reason = valid_backdoor(gh, "T", "Y", Z)
    parents_T = set(gh.predecessors("T"))
    matches_parents = (Z == parents_T)
    pass_thm = is_valid and matches_parents

    # (3) numerical sanity: OLS of Y on (T, L, X, C), true-SCM data
    L, X, C, T, Y = synthetic_data()
    M = np.column_stack([np.ones_like(T), T, L, X, C])
    beta, *_ = np.linalg.lstsq(M, Y, rcond=None)
    t_coef = float(beta[1])
    near_zero = abs(t_coef) < 0.05

    overall = declares_zero and pass_thm and near_zero
    result = {
        "verdict": "PASS" if overall else "FAIL",
        "tool": "networkx " + nx.__version__,
        "proposition_1_under_true_scm": {
            "claim": "No directed path T->Y in the true SCM, so the causal effect is zero.",
            "no_directed_path_T_to_Y": declares_zero,
        },
        "theorem_1_2_under_hypothetical_scm": {
            "claim": "{L,X,C} is a valid backdoor adjustment set and equals pa(T).",
            "backdoor_valid": is_valid, "reason": reason,
            "adjustment_set": sorted(Z), "parents_of_T": sorted(parents_T),
            "matches_parents": matches_parents,
        },
        "sanity_regression_true_scm": {
            "claim": "OLS coefficient on T (controlling for L,X,C) is near zero.",
            "t_coefficient": t_coef, "near_zero": near_zero,
        },
    }
    out = RESULTS / "02_dowhy_identification.json"
    out.write_text(json.dumps(result, indent=2))
    if not overall:
        print(f"[02] FAIL — see {out}")
        return 1
    print(f"[02] PASS — no T->Y path under true SCM (Prop 1); "
          f"{{L,X,C}}=pa(T) is a valid backdoor set (Thm 1/2); "
          f"OLS T-coef {t_coef:+.4f} near zero. Wrote {out}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
