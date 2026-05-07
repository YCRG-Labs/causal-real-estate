# Counterfactual LLM Experiments — Research Dossier (Item 1)

A research note compiled for: testing whether NLP-derived semantic features causally affect real-estate prices, or are spatial-confounding artifacts, via LLM-generated counterfactual listing rewrites.

---

## 1. Foundational Literature on Counterfactual Data Augmentation for NLP Causal Inference

### Established anchors
- **Feder, Keith, Jo, Manning, et al. (2022).** "Causal Inference in Natural Language Processing: Estimation, Prediction, Interpretation, and Beyond." *TACL.* Section 4 ("Causal Estimation with Text") and Section 6 ("Interpreting Predictions") are load-bearing. Establishes "text as treatment" framing — exactly our setup. arXiv:2109.00725.
- **Kaushik, Hovy, Lipton (2020).** "Learning the Difference that Makes a Difference with Counterfactually-Augmented Data." *ICLR.* Demonstrates humans can edit text to flip a single label-relevant attribute while holding others fixed. IMDB and SNLI CAD datasets are the reference benchmarks. arXiv:1909.12434.
- **Veitch, D'Amour, Yadlowsky, Eisenstein (2021).** "Counterfactual Invariance to Spurious Correlations in Text Classification." *NeurIPS.* The formal target we want: a model whose predictions are invariant to "non-causal" text variation. Defines counterfactual invariance and its testable implications. arXiv:2106.00545.

### Recent LLM-driven counterfactual work (2023–2025)
- **Gat, Calderon, Feder, Chapanin, Sharma, Reichart (2024).** "Faithful Explanations of Black-box NLP Models Using LLM-generated Counterfactuals." *ICLR 2024.* Closest published precedent — directly proposes LLM prompting for matched counterfactuals in causal explanation. arXiv:2310.00603.
- **Wang, Culotta (2021, then 2023 extensions).** "Identifying Spurious Correlations and Correcting Them with an Explanation-based Test." Uses GPT-style LLMs for matched-on-confounder counterfactuals. EMNLP 2021; ACL 2023 follow-up. arXiv:2106.02112.
- **Bhattacharjee, Liu (2024).** "Towards LLM-guided Causal Explainability." Position paper on LLMs as counterfactual generators. arXiv:2312.13127.
- **Fryer, Axelrod, Packer et al. (Google, 2022).** "Flexible text generation for counterfactual fairness probing." T5/PaLM-style demographic attribute swapping. arXiv:2206.13757. Method generalizes to submarket-style attributes.
- **Madaan, Setlur, Parekh, Poczos, Neubig, Yang, Hovy, Salakhutdinov (2021/2023).** "Generate Your Counterfactuals: Towards Controlled Counterfactual Generation for Text." AAAI 2021. arXiv:2012.04698.
- **Sen, Logeswaran, Lee, Lee (2023).** "People Make Better Edits: Measuring the Efficacy of LLM-Generated Counterfactual Edits." EMNLP Findings. Compares human vs LLM CAD quality — key for validation budget justification.

**Citation backbone for methods section:** Veitch et al. (2021) for formal target, Feder et al. (2022) for framing, Gat et al. (2024) and Sen et al. (2023) for LLM-generation legitimacy, Kaushik et al. (2020) as origin paradigm.

---

## 2. Prompt-Engineering Best Practices for Fact-Preserving Counterfactuals

### What works
- **Explicit constraint lists outperform free-form prompts.** Madaan et al. (2021) and Dixit et al. (2022, "CORE: A Retrieve-then-Edit Framework") both find enumerating preserved invariants ("preserve: square footage, bedroom count, year built; change: neighborhood vibe lexicon") cuts hallucination materially.
- **Retrieve-then-edit beats generate-from-scratch.** Dixit et al. CORE (NAACL 2022, arXiv:2210.04873) — retrieve a real listing close in factual content, then *edit* rather than *write*. Well-suited to SF-submarket-swap design.
- **Structured output (JSON with separate `preserved_facts` / `changed_style` fields)** forces explicit reasoning. See Polyjuice (Wu et al. 2021).
- **Chain-of-thought with verification step** ("first list every numeric/factual claim; then write rewrite; then verify list is preserved") reduces fact drift. Bhattacharjee & Liu (2024).
- **Multi-turn editing** outperforms single-shot for inputs >300 tokens. Sen et al. (2023): two-turn pipelines (draft → critique → revise) substantially improve preservation rates.

### Recommended invariant verification (essentially NLI / fact-extraction)
1. **Slot-extraction approach:** Define fixed schema (beds, baths, sqft, year, lot size, parking) and run deterministic extractor (regex or low-temp LLM) on both original and counterfactual. Reject any rewrite where any slot changes.
2. **NLI bidirectional entailment** for soft facts: each factual sentence in original must be entailed by rewrite, and vice versa for non-style claims. Use DeBERTa-v3-MNLI; threshold ~0.7 entailment.
3. **Attribute classifier** for the *intended* shift: train (or zero-shot prompt) a submarket classifier; rewrite should flip its prediction.

Cite: Wu et al. (Polyjuice, 2021) for control-code prompting; Ross et al. (2021) "Tailor: Generating and Perturbing Text with Semantic Controls" (ACL 2022, arXiv:2107.07150) for slot-aware editing.

---

## 3. Existing Code / Frameworks to Borrow

- **Polyjuice** (Wu, Ribeiro, Heer, Weld, ACL 2021). https://github.com/tongshuangwu/polyjuice. Pip-installable. Control codes too coarse for "submarket swap" but API and eval harness reusable.
- **Tailor** (Ross, Wu, Peng, Gardner, Marasović). https://github.com/allenai/tailor. AllenNLP-style. SRL-based perturbations. Useful as non-LLM baseline.
- **CausalNLP / DoWhy text-treatment module** (Microsoft Research, py-why). https://github.com/py-why/dowhy. Active.
- **CEBaB** (Abraham, D'Oosterlinck, Feder, Gat, Geiger, Potts, Reichart, Wu, NeurIPS 2022). https://cebabing.github.io/CEBaB/ and https://github.com/CEBaBing/CEBaB. **Most relevant existing benchmark.** Restaurant reviews with human-written counterfactuals across multiple aspects (food, service, ambiance, noise) for measuring causal concept effects on a sentiment regressor — structurally identical to our design. arXiv:2205.14140. **Recommendation: clone this and adapt the causal-effect-estimation harness almost wholesale.**
- **CausaLM** (Feder, Oved, Shalit, Reichart, CL 2021). https://github.com/amirfeder/CausalM. Counterfactual representation learning via concept-based adversarial training.
- **CheckList** (Ribeiro et al., ACL 2020). https://github.com/marcotcr/checklist. INV (invariance) and DIR (directional) tests map onto our two counterfactual types.
- **HuggingFace `evaluate`** has perplexity and BERTScore loaders for validation pipeline.

---

## 4. Validation Protocols

### Standard multi-pronged validation (do all three)
1. **Automated invariant checks**
   - Slot-level fact preservation, 100% of rewrites
   - Perplexity sanity check under frozen LM (GPT-2 or Llama-3-8B); flag >3× original's PPL distribution mean
   - NLI entailment for non-style content (DeBERTa-v3-large-MNLI)
   - Attribute classifier for the *intended* style shift

2. **Human validation on a stratified sample**
   - Standard N: **100–300 examples per condition** for tight Wilson CIs
   - For our 5 conditions × 500 = 2500 rewrites, validate **~50 per condition (250 total) min, 100/condition (500 total) for strong claim**
   - Two binary tasks: "Are all factual specs preserved?" and "Does the listing read like submarket X?"
   - Cohen's κ or Krippendorff's α; aim for κ > 0.6

3. **Round-trip / pair-discrimination**
   - Held-out classifier should distinguish original-vs-rewrite at near-100%
   - Held-out classifier should *not* distinguish style-stripped-vs-style-stripped across original submarkets

### Reporting
CAD-quality table mirroring Sen et al. (2023) Table 2: % facts preserved, % style-shifted, perplexity ratio, human IAA.

---

## 5. Technical Pitfalls and Mitigations

- **Hallucinated facts (#1 risk).** Plan for ~20% rejection / regeneration. Mitigation: slot-extractor reject loop.
- **Style leakage.** Mitigation: (a) two-pass generation (strip first, then re-style); (b) explicit "do not preserve any of the following lexical items" prompt; (c) attribute-classifier filter.
- **Distribution shift in style — counterfactuals don't match real listings.** Critical: if LLM's "Mission style" is a caricature, our model's response says nothing about real Mission. Mitigation: (a) few-shot prompting with 5–10 actual listings per submarket; (b) Mauve / FID-style divergence between rewrite distribution and real distribution.
- **Asymmetric difficulty across classes.** Rewriting Mission → Pacific Heights may be easier than reverse. Report per-direction quality metrics.
- **Annotator confound: humans recognize "this was AI-rewritten."** Blind annotation; mix originals into validation pool.
- **Spurious style-fact correlations / overlap violation.** "Luxury" listings have larger sqft. Rewriting style without changing sqft is OOD. Cite Veitch et al. (2021) §4 in limitations.
- **Prompt sensitivity / generator variance.** Use ≥2 generators (Claude + GPT-4), report inter-generator agreement, treat generator as random effect.

---

## 6. Connection to Causal Interpretability

Frame as estimating the **Natural Direct Effect of submarket style on predicted price, holding factual content fixed** (Pearl 2001). Style-stripped condition estimates NDE; original→swap estimates Total Effect. Difference identifies mediated effect of style.

- **Vig et al. (NeurIPS 2020).** "Investigating Gender Bias in Language Models Using Causal Mediation Analysis." Foundational. arXiv:2004.12265.
- **Pearl (2001).** "Direct and Indirect Effects." Formal source.
- **Geiger, Lu, Icard, Potts (2021/2023).** "Causal Abstractions of Neural Networks." ICML 2022. arXiv:2106.02997. Interchange interventions framework.
- **Abraham et al. CEBaB (NeurIPS 2022).** Closest methodological precedent.
- **Stolfo, Belinkov, Sachan (2023).** "A Mechanistic Interpretation of Arithmetic Reasoning in LLMs via Causal Mediation Analysis." EMNLP. arXiv:2305.15054.
- **Wu, Geiger, Icard, Potts, Goodman (2023).** "Interpretability at Scale: Identifying Causal Mechanisms in Alpaca." NeurIPS.

---

## Citation Backbone (Methods Section)

1. Feder et al. (TACL 2022) — framing
2. Veitch et al. (NeurIPS 2021) — formal target
3. Kaushik et al. (ICLR 2020) — paradigm origin
4. Abraham et al. CEBaB (NeurIPS 2022) — closest precedent
5. Gat et al. (ICLR 2024) — LLM-as-generator legitimacy
6. Sen et al. (EMNLP Findings 2023) — quality-vs-human comparison
7. Vig et al. (NeurIPS 2020) — mediation framing
8. Wu et al. Polyjuice (ACL 2021) — control-code prompting
9. Pearl (2001) — NDE/NIE definitions
10. Ribeiro et al. CheckList (ACL 2020) — INV/DIR test taxonomy

---

## Concrete Next Steps

1. **Clone CEBaB**, adapt causal-effect-estimation harness to real-estate listings.
2. **Build slot extractor first** (regex + small LLM call). Reject filter.
3. **Pre-register analysis** — counterfactual designs most credible when ATE estimands and rejection rules fixed before generation.
4. **Pilot of 25 listings × 4 variants** with both Claude and GPT-4; validate manually; *then* scale to 500.
5. **Budget human annotation** for ~250–500 rewrites. Prolific ~$1.50–3 per binary annotation; 500 × 2 questions × 3 annotators ≈ $4.5–9k.
6. **Pre-commit to reporting null results.** A null NDE *is* the finding if SCM₀ holds.
