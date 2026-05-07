"""Counterfactual LLM rewrite pipeline for causal-NLP real-estate analysis.

Estimates Natural Direct Effect (NDE) of submarket-evocative listing style on
predicted log-price, holding factual property content fixed. Implements two
counterfactual designs:

  - style-swap: rewrite as if listing were in a target submarket
  - style-stripped: rewrite removing all neighborhood-evocative language

All rewrites pass through three validation checks (slot preservation,
perplexity sanity, attribute-classifier flip) before being scored against the
production DML pipeline.

Frame: Pearl (2001) NDE; Veitch et al. (2021) counterfactual invariance;
CEBaB (Abraham et al. 2022) as closest precedent.
"""
from __future__ import annotations
