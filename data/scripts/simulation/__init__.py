"""JBES-style simulation validation for the four causal estimators.

Validates DR (binarized doubly-robust ATE), DML (continuous-treatment partially
linear), adversarial deconfounding, and randomization tests against:

  - SCM_0: no direct text -> price effect (size / coverage check)
  - SCM_1: known direct effect calibrated to 1%, 5%, 10% of Var(Y) (power check)

The data-generating process samples synthetic embeddings from a per-zip-bin
Gaussian-mixture conditional generator fitted on real (E, z) pairs from the
SF MPNet release. See `research/simulation/research_notes.md` for the full
dossier (DGP options, table format, citations).

Refs:
  Chernozhukov et al. 2018 EJ — DML simulation template (bias/SD/RMSE/coverage)
  Athey, Imbens, Metzger & Munro 2024 JoE — generator-based MC populations
  Knaus, Lechner & Strittmatter 2021 EJ — empirical Monte Carlo
  Belloni, Chernozhukov & Hansen 2014 REStud — table conventions
  Wager & Athey 2018 JASA — power-curve reporting

Usage:
  python -m simulation.run_simulation               # smoke (R=20, N=500)
  python -m simulation.run_simulation --n_reps 200  # default smoke grid
  python -m simulation.run_simulation --n_reps 1000 --full_grid  # full
  python -m simulation.plot_results                 # JBES-style figures
"""
from __future__ import annotations
