"""
Independent corroboration of Proposition 1 / Theorem 1 / Theorem 2 via DoWhy.

DoWhy implements graph-based identification (Pearl's ID algorithm + the
backdoor/frontdoor criteria). We run it twice on the SCM:

1. Under the *true* SCM (no T -> Y edge): DoWhy must report that the causal
   effect is structurally zero. This corroborates Proposition 1 and
   Corollary 1.

2. Under a hypothetical SCM with T -> Y added: DoWhy must return a backdoor
   estimand with adjustment set {L, X, C}, matching Theorem 1 / Theorem 2.

Two implementations agreeing is stronger than either alone.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from dowhy import CausalModel  # noqa: E402

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)


GML_TRUE_SCM = """graph [
    directed 1
    node [ id "L" label "L" ]
    node [ id "X" label "X" ]
    node [ id "C" label "C" ]
    node [ id "T" label "T" ]
    node [ id "Y" label "Y" ]
    edge [ source "L" target "X" ]
    edge [ source "L" target "C" ]
    edge [ source "L" target "T" ]
    edge [ source "X" target "T" ]
    edge [ source "C" target "T" ]
    edge [ source "L" target "Y" ]
    edge [ source "X" target "Y" ]
    edge [ source "C" target "Y" ]
]"""

GML_HYPOTHETICAL_SCM = GML_TRUE_SCM.rstrip().rstrip("]") + '    edge [ source "T" target "Y" ]\n]'


def synthetic_data(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    L = rng.normal(size=n)
    X = 0.7 * L + rng.normal(scale=0.5, size=n)
    C = 0.5 * L + rng.normal(scale=0.5, size=n)
    T = 0.4 * L + 0.3 * X + 0.2 * C + rng.normal(scale=0.5, size=n)
    Y = 1.0 * L + 0.6 * X + 0.4 * C + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"L": L, "X": X, "C": C, "T": T, "Y": Y})


def main() -> int:
    df = synthetic_data()

    model_true = CausalModel(
        data=df, treatment="T", outcome="Y", graph=GML_TRUE_SCM,
    )
    estimand_true = model_true.identify_effect(proceed_when_unidentifiable=False)
    estimand_true_str = str(estimand_true)
    declares_zero = ("causal effect is zero" in estimand_true_str.lower()) or \
                    ("no directed path" in estimand_true_str.lower())

    model_hyp = CausalModel(
        data=df, treatment="T", outcome="Y", graph=GML_HYPOTHETICAL_SCM,
    )
    estimand_hyp = model_hyp.identify_effect(proceed_when_unidentifiable=False)
    bd_dict = estimand_hyp.backdoor_variables or {}
    backdoor_vars = set(bd_dict.get("backdoor", bd_dict.get("backdoor1", [])))
    expected_adjustment = {"L", "X", "C"}
    adjustment_matches = backdoor_vars == expected_adjustment
    estimand_text = str(estimand_hyp)
    method_is_backdoor = "backdoor" in estimand_text.lower() and "estimand expression" in estimand_text.lower()

    estimate = model_hyp.estimate_effect(
        estimand_hyp,
        method_name="backdoor.linear_regression",
        confidence_intervals=False,
        test_significance=False,
    )
    point_estimate = float(estimate.value)

    pass_prop1 = declares_zero
    pass_thm1_thm2 = method_is_backdoor and adjustment_matches
    sanity_estimate_low = abs(point_estimate) < 0.1
    overall_pass = pass_prop1 and pass_thm1_thm2 and sanity_estimate_low

    result = {
        "verdict": "PASS" if overall_pass else "FAIL",
        "proposition_1_under_true_scm": {
            "claim": "Under SCM with no T->Y edge, DoWhy must declare zero causal effect.",
            "dowhy_declares_zero": declares_zero,
            "estimand_text": estimand_true_str[:400],
            "holds": pass_prop1,
        },
        "theorem_1_2_under_hypothetical_scm": {
            "claim": "If T->Y were added, the backdoor adjustment set would be {L,X,C}.",
            "dowhy_returned_backdoor_estimand": method_is_backdoor,
            "dowhy_backdoor_variables": sorted(backdoor_vars),
            "expected_adjustment_set": sorted(expected_adjustment),
            "adjustment_set_matches_paper": adjustment_matches,
            "estimand_text_excerpt": estimand_text[:600],
            "holds": pass_thm1_thm2,
        },
        "sanity_estimate_under_hypothetical_scm": {
            "note": "DGP has zero causal effect of T on Y; estimate via backdoor "
                    "adjustment on hypothetical T->Y SCM should be near zero.",
            "point_estimate": point_estimate,
            "near_zero": sanity_estimate_low,
        },
    }

    out = RESULTS / "02_dowhy_identification.json"
    out.write_text(json.dumps(result, indent=2))

    if not overall_pass:
        print(f"[02] FAIL — see {out}")
        return 1
    print(f"[02] PASS — DoWhy declares zero under true SCM (Prop 1) AND returns "
          f"backdoor adjustment {sorted(backdoor_vars)} under hypothetical SCM "
          f"(Thm 1/Thm 2). Wrote {out}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
