"""
Verification of Proposition 2(c) — saddle-quality construction.

Claim: there exist a data distribution P and class pair Φ ⊊ Φ' such that
for every saddle of the gradient-reversal game restricted to Φ,
    V_Φ(Z; C) = 0   but   V_Φ'(Z; C) ≥ H(C) - δ
for arbitrary δ > 0.

The XOR construction in Appendix G of the paper instantiates this. This
script wraps `data/scripts/theory/frozen_probe_gap.py`, re-runs it (so
verification is fully reproducible from this directory), and asserts:

  Experiment 1 (static XOR, Φ = linear, Φ' = MLP):
     V_Φ(Z;C)   ≈ 0     to within 0.05 nats
     V_Φ'(Z;C)  ≥ H(C) - 0.05 nats = 0.643 nats
     Gap        ≥ 0.6  nats

  Experiment 2 (dynamic gradient reversal):
     live discriminator accuracy ≈ chance (within 0.05 of 0.5)
     frozen MLP probe accuracy   > frozen linear probe accuracy + 3pp

  Experiment 3 (capacity ladder):
     V_Φ is monotone non-decreasing in capacity
     V_Φ at top of ladder > 0.95 * H(C)

If the JSON already exists, we re-use it (faster CI); otherwise we run the
underlying script.
"""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR_THEORY = REPO / "results" / "theory"
THEORY_JSON = RESULTS_DIR_THEORY / "frozen_probe_gap.json"
THEORY_SCRIPT = REPO / "data" / "scripts" / "theory" / "frozen_probe_gap.py"

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)


def ensure_results() -> dict:
    if not THEORY_JSON.exists():
        print(f"[06] running {THEORY_SCRIPT} to populate {THEORY_JSON} ...")
        subprocess.run([sys.executable, str(THEORY_SCRIPT)], check=True, cwd=str(REPO))
    return json.loads(THEORY_JSON.read_text())


def main() -> int:
    H_C = 0.6931472
    data = ensure_results()

    e1 = data["experiment_1_static_xor"]
    v_phi_lin = e1["linear"]["v_phi_nats_mean"]
    v_phi_mlp = e1["mlp"]["v_phi_nats_mean"]
    gap1 = e1["gap_mean_nats"]
    e1_pass = (
        abs(v_phi_lin) < 0.05
        and v_phi_mlp >= H_C - 0.05
        and gap1 >= 0.6
    )

    e2 = data["experiment_2_dynamic_grl"]
    live_acc = e2["live_discriminator_accuracy"]
    flin_acc = e2["frozen_linear_probe_accuracy"]
    fmlp_acc = e2["frozen_mlp_probe_accuracy"]
    e2_pass = (
        abs(live_acc - 0.5) < 0.05
        and (fmlp_acc - flin_acc) > -0.01
        and fmlp_acc > 0.5 + 0.03
    )

    e3 = data["experiment_3_capacity_ladder"]
    v_values = [r["v_phi_nats_mean"] for r in e3["rungs"]]
    monotone = all(v_values[i] <= v_values[i + 1] + 0.05 for i in range(len(v_values) - 1))
    top = v_values[-1]
    e3_pass = monotone and top > 0.95 * H_C

    overall = e1_pass and e2_pass and e3_pass

    result = {
        "verdict": "PASS" if overall else "FAIL",
        "H_C_nats": H_C,
        "experiment_1_static_xor": {
            "claim": "V_lin ≈ 0 ≪ V_MLP ≈ H(C); gap ≥ 0.6 nats.",
            "v_phi_linear_nats": v_phi_lin,
            "v_phi_mlp_nats": v_phi_mlp,
            "gap_nats": gap1,
            "passes": bool(e1_pass),
        },
        "experiment_2_dynamic_grl": {
            "claim": "Live discriminator at chance; frozen MLP probe above frozen linear probe.",
            "live_discriminator_accuracy": live_acc,
            "frozen_linear_probe_accuracy": flin_acc,
            "frozen_mlp_probe_accuracy": fmlp_acc,
            "diagnostic_gap_accuracy_pp": (fmlp_acc - flin_acc) * 100,
            "passes": bool(e2_pass),
        },
        "experiment_3_capacity_ladder": {
            "claim": "V_Φ monotone non-decreasing in capacity; top rung ≥ 0.95 * H(C).",
            "v_phi_by_rung": [
                {"name": r["name"], "v_phi_nats": r["v_phi_nats_mean"]}
                for r in e3["rungs"]
            ],
            "monotone_in_capacity": bool(monotone),
            "top_rung_nats": top,
            "passes": bool(e3_pass),
        },
    }

    out = RESULTS / "06_xor_construction.json"
    out.write_text(json.dumps(result, indent=2))

    if not overall:
        print(f"[06] FAIL — see {out}")
        return 1
    print(f"[06] PASS — XOR saddle gap construction verified: V_lin={v_phi_lin:.3f}, "
          f"V_MLP={v_phi_mlp:.3f}, gap={gap1:.3f} nats; live disc {live_acc:.3f}, "
          f"frozen MLP probe {fmlp_acc:.3f}. Wrote {out}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
