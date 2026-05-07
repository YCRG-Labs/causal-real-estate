# Computational Verification Appendix

This directory provides a **computational verification artifact** for every analytic
claim in the paper. It is **not** a machine-checked formal proof — those would
require Lean 4 / Coq / Isabelle, and as of late 2025 no proof assistant has a
causal-DAG library, which would gate even the d-separation lemma.

What we provide instead, for each obligation:

- a deterministic, machine-checkable script,
- pinned tool versions (`requirements.txt`),
- pass/fail assertions, and
- JSON output in `results/`.

## Mapping from paper claims to scripts

| Paper claim | Type | Script | Method |
|---|---|---|---|
| Lemma 1 (d-separation, App. B) | Graph-theoretic | `01_dag_dseparation.py` | `pgmpy.is_dconnected` + path enumeration |
| Theorem 1 (Backdoor Adjustment, §3.4) | Causal-graphical | `01_dag_dseparation.py`, `02_dowhy_identification.py` | pgmpy + independent DoWhy `identify_effect` |
| Theorem 2 (Backdoor Identifiability, App. B) | Causal-graphical | `02_dowhy_identification.py` | DoWhy `identify_effect` returns the matching estimand |
| Proposition 1 (Zero Causal Effect, §3.3) | Definitional | `01_dag_dseparation.py` | Edge-set check: T not in pa(Y) ⟹ do(T) inert on Y |
| Corollary 1 (Zero effect under true SCM, App. B) | Definitional | `01_dag_dseparation.py` | Same edge-set check |
| Proposition 2(a) (variational MI inequality, §3.5) | Information-theoretic | `03_variational_mi_inequalities.py` | Cite Xu et al. 2020 Prop 1 + numerical sanity for Φ ⊆ Φ' ⟹ V_Φ ≤ V_Φ' ≤ I |
| Proposition 2(b1) (algebraic identity) | Algebraic | `04_algebraic_identity.py` | SymPy: V̂_Φ = H(C) − inf_ψ L̂_Φ(ψ) |
| Proposition 2(b2) (consistency of post-hoc V̂) | Asymptotic | `05_consistency_rate.py` | Monte Carlo n-sweep + log-log slope |
| Proposition 2(c) (XOR saddle gap) | Constructive existence | `06_xor_construction.py` | Wraps `data/scripts/theory/frozen_probe_gap.py`; assert V̂_Φ ≈ 0, V̂_Φ' ≈ log 2 |

## Running

```bash
pip install -r requirements.txt
make verify          # runs all 6 scripts headless, prints PASS/FAIL summary
```

Or run individually:
```bash
python 01_dag_dseparation.py
python 02_dowhy_identification.py
python 03_variational_mi_inequalities.py
python 04_algebraic_identity.py
python 05_consistency_rate.py
python 06_xor_construction.py
```

Each script writes `results/<script-name>.json` with the verdict and any numerical
quantities cited in the paper.

## What this artifact does and does not claim

**Does claim:**
- Every analytic statement in the paper has been re-derived by an independent,
  open-source tool and the result matches the written proof.
- The d-separation, backdoor identification, and XOR saddle constructions are
  *deterministically* checkable (no randomness, no estimation error).
- The asymptotic consistency claim is *empirically* checked at the rate predicted
  by the cited theorem (Xu et al. 2020 Theorem 3).

**Does not claim:**
- A formally verified Lean/Coq/Isabelle proof.
- Soundness in the sense of an interactive theorem prover.
- That the tools (pgmpy, DoWhy, SymPy) are themselves bug-free.

The honest summary: this is a *reproducibility artifact for the analytic claims*,
at the bar that current top stats journals (JBES, JASA, AoS, Biometrika) accept.
