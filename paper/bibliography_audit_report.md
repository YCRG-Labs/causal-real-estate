# Bibliography Hallucination Audit Report

Date: 2026-06-02. Audit conducted via research agent with WebFetch on each citation. 38 references checked, 24 verified clean (63%), 14 flagged for correction.

## Critical errors (MUST fix before submission)

### Fabricated authorships
- **Bach/Schacht 2024 (arXiv 2409.04874)** — actual authors are **Ballinari & Bearth**. Title: "Improving the Finite Sample Estimation of Average Treatment Effects using Double/Debiased Machine Learning with Propensity Score Calibration."
- **Ravfogel, Vargas, Goldberg, Cotterell (2023) "Log-linear Guardedness"** — "Vargas" is a phantom author. Real authors: Ravfogel, Goldberg, Cotterell only.

### Fully hallucinated paper
- **"Hahn & Imbens bootstrap for cross-fit DML"** — does not exist in any literature I could verify. Substitute **Tang & Westling 2024 (arXiv 2404.03064)** for the cross-fit bootstrap consistency citation.

### Two-paper conflation
- **Imai & Nakamura GenAI** — arXiv 2410.00903 ("Causal Representation Learning with Generative AI") and arXiv 2507.03897 ("GenAI-Powered Inference") are DIFFERENT papers. Pick one per claim; do not conflate.

## Title and venue corrections

- **Bach et al. DoubleML** — arXiv 2103.09603 is the **R** paper (JSS 108(3), 2024). The **Python** paper is arXiv 2104.03220 (JMLR 23(53), 2022). Pick the right one for the citation context.
- **Pryzant, Card, Jurafsky, Veitch, Sridhar (2021)** — venue is **NAACL 2021**, not just arXiv.
- **Egami, Hinck, Stewart, Wei** — year is **2023 (NeurIPS 2023)**, not 2024. Title: "Using Imperfect Surrogates for Downstream Inference: Design-based Supervised Learning for Social Science Applications of Large Language Models."
- **Kumar, Tan, Sharma (2022)** — title: "Probing Classifiers are Unreliable for Concept Removal **and Detection**." (Missing "and Detection".)
- **Feder et al. (2022) TACL** — title: "Causal Inference in NLP: **Estimation, Prediction, Interpretation and Beyond**." (Add subtitle.)
- **Rosen (1974) JPE** — title: "Hedonic Prices and Implicit Markets: **Product Differentiation in Pure Competition**." (Add subtitle.)
- **Van Lissa, Van Erp, Clapper (2023)** — full title: "**Selecting relevant moderators with** Bayesian regularized meta-regression." (Add "Selecting relevant moderators with".)

## Page-range corrections

- **Higgins & Thompson (2002)** — pages **1539–1558**, not 1559–1573.
- **Van Lissa, Van Erp, Clapper (2023)** — pages **301–322**, not just 301.

## Lam (2022) "Cheap Bootstrap" — unverified venue

Paper exists at arXiv:2202.00090 but JASA publication could not be confirmed via WebFetch (Taylor & Francis 403). Cite as arXiv preprint until JASA listing verified directly.

## Verified clean (24 references — no changes needed)

1. Chernozhukov et al. (2018) EJ 21(1):C1-C68
2. Chiang, Kato, Ma, Sasaki (2022) JBES 40(3):1046-1056
3. Bia, Huber, Lafférs (2024) JBES 42(3):958-969
4. Knaus (2022) EJ 25(3):602-627
5. Cinelli & Hazlett (2020) JRSSB 82(1):39-67
6. Chernozhukov et al. (2024) "Long Story Short" arXiv:2112.13398
7. Gilbert et al. (2024) "Spatial confounding" arXiv:2112.14946
8. Gibbons & Overman (2012) JRS 52(2):172-191
9. Tang & Westling (2024) arXiv:2404.03064
10. Knapp & Hartung (2003) Stat Med 22:2693-2710
11. Viechtbauer (2010) JSS 36(3):1-48
12. Higgins & Thompson (2004) Stat Med 23(11):1663-1682
13. Veroniki et al. (2016) RSM 7(1):55-79
14. Shen & Ross (2021) JUE 121:103299
15. Baur, Rosenfelder, Lutz (2023) ESWA 213:119147
16. Belrose et al. (2023) NeurIPS LEACE arXiv:2306.03819
17. Iskander, Radinsky, Belinkov (2023) ACL Findings arXiv:2305.10204
18. Holstege, Ravfogel, Wouters (2025) NeurIPS SPLINCE arXiv:2506.10703
19. Voita & Titov (2020) EMNLP arXiv:2003.12298
20. Veitch, Sridhar, Blei (2020) UAI PMLR 124:919-928
21. Lopez-Paz & Oquab (2017) ICLR arXiv:1610.06545
22. Borenstein et al. (2009) Wiley meta-analysis textbook
23. Le & Mikolov (2014) ICML PMLR 32(2):1188-1196
24. Cohen (1988) Lawrence Erlbaum power analysis textbook
25. Saco (2025) arXiv:2512.07083

Net audit: 60% fabrication-rate prior did NOT bear out — 63% verified clean. Errors concentrated in newer preprints (where venue/year ambiguity is high) and page-range slips. Two fabricated authorships and one fully hallucinated paper are the real critical fixes.
