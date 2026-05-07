"""Replications of published real-estate-NLP papers on the SF dataset.

Two replications are included; each reproduces the published method on our
SF data and then re-runs the same outcome through the project's DML
continuous-treatment pipeline so that the reported predictive / OLS gain can
be contrasted against the corresponding causal estimate.

  - shen_2021: Shen & Ross (JUE 2021) "Information value of property
    description". TF-IDF based "uniqueness" score → hedonic OLS coefficient,
    then DML on uniqueness as continuous treatment.
  - baur_2023: Baur, Rosenfelder & Lutz (ESWA 2023) "Automated real estate
    valuation with machine learning models using property descriptions".
    GBM on structured-only vs structured+BERT, then DML on PC1 of BERT.

The intent is a rebuttal exercise: each paper reports a real predictive or
hedonic gain attributable to listing text, and the same gain is re-estimated
under the project DML adjustment to show that the causal interpretation is
not supported by the same data.
"""
from __future__ import annotations
