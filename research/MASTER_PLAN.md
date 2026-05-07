# Master Plan — 13-Week Research & Implementation Roadmap

For taking the JBES paper from ~25% acceptance odds → 60–70%.

Per-item dossiers (deeper detail, full citations) live under:
- `research/counterfactual/research_notes.md` — Item 1
- `research/negative_controls/research_notes.md` — Item 2
- `research/replications/research_notes.md` — Item 3
- `research/mechanism/research_notes.md` — Item 4
- `research/simulation/research_notes.md` — Item 5
- `research/cate/research_notes.md` — Item 6
- `research/sensitivity/research_notes.md` — Item 7
- `research/theory/research_notes.md` — Item 8
- `research/packaging/research_notes.md` — Item 9

This file is the synthesis: dependency graph, citation backbone, infrastructure decisions, week-by-week deliverables.

---

## 1. Dependency Graph

```
Item 3 (replications) ────────┐
                              │
Item 1 (counterfactual) ──────┤
                              │
Item 4 (mechanism) ───────────┼──► Item 5 (simulations) ──► Item 9 (packaging)
                              │
Item 2 (neg controls) ────────┤        │
                              │        │
Item 6 (CATE)         ────────┤        ▼
                              │
Item 7 (sensitivity)  ────────┘   Item 8 (theory) — independent track
```

- Items 1–4, 6 are largely independent and can run in parallel where compute and human attention allow
- Item 5 (simulations) wants methods to be settled, so it lands after the empirical work
- Item 7 (sensitivity) reuses outputs from Items 2 and 6
- Item 8 (theory) is independent — pursue with co-author
- Item 9 (packaging) consumes outputs from all empirical items; must come last

---

## 2. Single Citation Backbone (references.bib additions)

Curated minimum-but-complete additions across all nine items. ~30 entries.

### Counterfactual NLP (Item 1)
- Feder et al. (TACL 2022) — framing
- Veitch et al. (NeurIPS 2021) — formal target (counterfactual invariance)
- Kaushik et al. (ICLR 2020) — paradigm origin
- Abraham et al. CEBaB (NeurIPS 2022) — closest precedent
- Gat et al. (ICLR 2024) — LLM-as-generator legitimacy
- Sen et al. (EMNLP Findings 2023) — quality-vs-human comparison
- Vig et al. (NeurIPS 2020) — mediation framing
- Wu et al. Polyjuice (ACL 2021) — control-code prompting
- Pearl (2001) — NDE/NIE definitions
- Ribeiro et al. CheckList (ACL 2020) — INV/DIR test taxonomy

### Negative Controls (Item 2)
- Lipsitch et al. (Epidemiology 2010) — seminal NCO/NCE
- Shi, Miao, Tchetgen Tchetgen (CER 2020) — modern review, proximal connection
- Schuemie et al. (SiM 2014, 2016; PNAS 2018) — empirical calibration
- Chernozhukov, Cinelli, Newey, Sharma, Syrgkanis (2022, arXiv:2112.13398) — OVB-DML
- Cui, Pu, Shi, Miao, Tchetgen Tchetgen (AJE 2024) — proximal regression

### Replication targets (Item 3)
- Shen & Ross (JUE 2021)
- Baur, Rosenfelder & Lutz (ESWA 2023)
- Zhang et al. MugRep (KDD 2021)

### Mechanism (Item 4)
- Hovy & Yang (NAACL 2021) — central anchor for stylometric vs lexical
- Eisenstein et al. (EMNLP 2010) — geographic lexical variation
- Burrows (LLC 2002) — Delta stylometry
- Eder, Rybicki, Kestemont (R Journal 2016) — `stylo`
- Sundararajan, Taly, Yan (ICML 2017) — IG attribution
- Yang & Pedersen (ICML 1997) — IG/MI feature selection
- Heylighen & Dewaele (Foundations of Science 2002) — F-score formality
- Hutto & Gilbert (ICWSM 2014) — VADER

### Simulation (Item 5)
- Chernozhukov et al. (EJ 2018) — DML
- Athey, Imbens, Metzger, Munro (JoE 2024) — WGAN simulation design
- Knaus, Lechner, Strittmatter (EJ 2021) — empirical Monte Carlo
- Wager & Athey (JASA 2018) — table format reference
- Belloni, Chernozhukov, Hansen (REStud 2014) — table format reference
- Farrell, Liang, Misra (Econometrica 2021) — neural-nuisance simulation
- Ravfogel et al. INLP (ACL 2020) — frozen-probe construction
- Elazar & Goldberg (EMNLP 2018) — empirical analog of frozen probe
- Belghazi et al. MINE (ICML 2018) — Donsker-Varadhan MI

### CATE (Item 6)
- Kennedy (EJS 2023) — DR-Learner; arXiv:2004.14497
- Kennedy, Ma, McHugh, Small (JRSS-B 2017) — continuous-T DR
- Semenova & Chernozhukov (EJ 2021) — debiased ML for CATE/structural functions
- Chernozhukov, Demirer, Duflo, Fernandez-Val (NBER WP 24678, 2018) — generic ML inference, BLP/GATES
- Athey & Imbens (PNAS 2016) — recursive partitioning

### Sensitivity (Item 7)
- VanderWeele & Ding (Annals Int. Med. 2017) — E-value
- Linden, Mathur, VanderWeele (Stata J. 2020) — `EValue` software, OLS extension
- Cinelli & Hazlett (JRSS-B 2020) — sensitivity / RV (current method)
- McCandless & Gustafson (SiM 2017) — BSA vs MCSA
- Manski (AER P&P 1990) — partial identification

### Theory (Item 8)
- Ganin et al. (JMLR 2016) — DANN
- Zhao et al. (ICML 2019) — invariance impossibility
- Moyer et al. (NeurIPS 2018) — direct gradient-reversal critique
- Hewitt & Liang (EMNLP 2019) — probe-vs-discriminator selectivity
- Pimentel et al. (ACL 2020) — info-theoretic probing (cleanest framework)
- Voita & Titov (EMNLP 2020) — MDL probing
- Belrose et al. LEACE (NeurIPS 2023) — perfect linear erasure baseline
- Achille & Soatto (JMLR 2018) — emergence of invariance

### Packaging (Item 9)
- DoubleML (Bach et al., JOSS 2022) — JOSS reference
- DoWhy (Sharma & Kiciman, JOSS 2022) — JOSS reference
- EconML (Battocchi et al., 2019) — package reference

**Total ~50 citations to add to `references.bib`.** Most already in our bib (84 entries) — need to audit overlap; expect ~30-40 net additions.

---

## 3. Infrastructure & Tooling Decisions

These are settled — adopt across all items unless a specific item argues otherwise.

| Decision | Choice | Why |
|---|---|---|
| Causal estimation core | `econml.dml.LinearDML` for continuous-T DML | JBES audience cites it; closed-form HC SEs |
| CATE estimator | `econml.dml.LinearDML` + Semenova-Chernozhukov BLP projection | Continuous T + low-dim heterogeneity |
| Heterogeneity test | Chernozhukov et al. 2018 BLP/GATES | Most defensible at JBES |
| Sensitivity (DML) | `dml.sensemakr` (R, via `rpy2`) | Only canonical implementation; RV + RVa |
| E-values | `EValue` (R, via `rpy2`) | No first-class Python equiv |
| BSA | Custom 40-line MC (no Python port of `episensr`) | Inherits CCNSS 2022 bias formula |
| Empirical calibration | `EmpiricalCalibration` (R) | OHDSI-maintained, citation-ready |
| Partial-ID | Skip `autobounds` for now; report plausibility-bounded Manski | Full bounds too wide to be useful |
| Counterfactual generation | Claude + GPT-4 dual-generator | Reduces single-model artifact risk |
| CF eval framework | Adapt CEBaB (NeurIPS 2022) | Closest published precedent |
| Stylometric features | textstat + spaCy + Burrows-Delta function-word MFW | Replicable, citable |
| Attribution | `captum.attr.LayerIntegratedGradients` | Captum is the standard for transformer attribution |
| Simulation generator | Conditional RealNVP via `nflows` | Athey-Imbens-Metzger-Munro 2024 standard |
| MC orchestration | `joblib.Parallel`, 1000-2000 reps/cell | Standard |
| Refutation | DoWhy `refute_estimate` for placebo / random-cause | Clean API |
| Package layout | src layout, hatchling, uv.lock | Modern Python research standard |
| Docs | Sphinx + Furo + sphinx-gallery + myst | Causal-inference field default |
| CI | GitHub Actions (test, docs, publish, pre-commit, reproduce) | scientific-python/cookie pattern |
| JOSS | Submit in parallel with JBES | Citable software DOI strengthens reproducibility claim |

---

## 4. Week-by-Week Deliverables

### Weeks 1–2 — Counterfactual LLM experiments
**Output:** new §5.6 "Counterfactual Intervention Test", Table 8, Figure 5
**Concrete tasks:**
- W1.1: Build slot extractor (regex + LLM call) for fact preservation
- W1.2: Set up CEBaB-style eval harness adapted to real estate
- W1.3: Pilot 25 listings × 4 variants × 2 generators (Claude + GPT-4); manual validation
- W2.1: Scale to 500 listings × 4 variants
- W2.2: Run automated invariant checks (slot, NLI, attribute classifier)
- W2.3: Human validation on stratified 250-rewrite sample (Prolific or similar)
- W2.4: Run all 5 variants through valuation model; estimate NDE; write up
**Cost:** ~$200 API + ~$5k human annotation

### Week 3 — Negative controls
**Output:** new §5.7 "Negative Control Validation", Table 9
**Concrete tasks:**
- W3.1: NCO panel — rerun DML with `y` ← {parcel_area, year_built, n_stories}
- W3.2: NCE panel — rerun DML with `T` ← {row-permuted, synthetic-text-encoded, wrong-corpus}
- W3.3: Schuemie empirical calibration via `EmpiricalCalibration` R package (rpy2)
- W3.4: Chernozhukov et al. 2022 OVB sensitivity via `DoubleML.sensitivity_analysis()`
**Cost:** ~2 days compute + 3 days writing

### Weeks 4–6 — Replications
**Output:** new §5.8 "Re-evaluation of Published Models", Table 10
**Order: Shen → Baur → MugRep**
- W4: Shen & Ross 2021 — uniqueness measure on our SF data; verify +15% headline; apply our DML/frozen-probe
- W5: Baur et al. 2023 — GBM + BERT pipeline; verify ΔMAPE; apply our DML/frozen-probe
- W6: MugRep — partial replication (graph beats GBRT; not chasing exact MAPE numbers); apply our DML/frozen-probe
**Note:** MugRep is 6–10 person-weeks for full replication; we deliver a *partial* replication that captures the qualitative claim and apply our causal pipeline. Document the partial-replication scope honestly.

### Week 7 — Mechanism analysis
**Output:** new §5.9 "Mechanism: How Language Encodes Location", Table 11, Figure 6
**Concrete tasks:**
- W7.1: Extract ~230-dim stylometric feature vector (textstat + spaCy + Burrows function-word MFW + VADER)
- W7.2: Train zip-classifier on stylometric features alone; compare to full-embedding classifier
- W7.3: Compute IG/MI per word against zip-bin; rank top-100; manual coding
- W7.4: Captum LayerIntegratedGradients on location probe; report top-20 attributed tokens
- W7.5: Cross-tabulate IG-attribution vs MI ranking

### Weeks 8–9 — Simulation study
**Output:** new Appendix F "Simulation Validation", Table A1, Figure A1
**Concrete tasks:**
- W8.1: Build conditional RealNVP generator on (E, z) pairs (`nflows`)
- W8.2: Implement SCM₀ DGP (no direct effect); SCM₁ DGP (β grid 1%/5%/10%)
- W8.3: Wire `econml.dml.LinearDML`, `econml.dr.LinearDRLearner`, custom adversarial+frozen-probe, custom randomization
- W9.1: Run 1000-replication MC across estimators × N × DGP × β
- W9.2: Build coverage/power tables (Wager-Athey 2018 format)
- W9.3: Frozen-probe diagnostic test cell (linear vs MLP discriminator)
- W9.4: Power curves figure

### Week 10 — CATE
**Output:** new §5.10 "Heterogeneous Treatment Effects" (replacing deleted CATE methods section), Table 12, Figure 7
**Concrete tasks:**
- W10.1: `econml.dml.LinearDML` with `discrete_treatment=False` on SF
- W10.2: CATE-by-quartile (price + description-length); IF-based stratum CIs
- W10.3: BLP heterogeneity test (Chernozhukov et al. 2018)
- W10.4: Across-50-seed stability check
- W10.5: Overlap diagnostic plots

### Week 11 — Sensitivity bounds
**Output:** new §5.11 "Robustness to Unobserved Confounding", Table 13, Figure 8
**Concrete tasks:**
- W11.1: `dml.sensemakr` (rpy2) for honest DML RV / RVa
- W11.2: E-values via custom 5-line + `EValue` (rpy2) cross-check
- W11.3: 40-line BSA Monte Carlo over (η_Y, η_D) ~ Beta(2,8)
- W11.4: Manski plausibility-bounded interval
- W11.5: Synthesis plot: E-value, RV, BSA posterior on same axis

### Week 12 — Theory sketch
**Output:** new §3.5 "Theoretical Properties of the Frozen-Probe Diagnostic", Appendix G
**Concrete tasks:**
- W12.1: Co-author session — write Proposition 2 statement + proof sketch
- W12.2: Worked Gaussian example (Proposition 1) with concrete gap magnitudes
- W12.3: Connect to empirical 19–116× numbers
- W12.4: Internal review (Brandon or advisor)
**Critical:** This week needs Brandon or an advisor with formal causal-inference / representation-learning theory background. **Not a solo task.**

### Week 13 — Toolkit packaging
**Output:** `pip install spatial-confounding-audit`, JOSS submission, Zenodo DOI
**Concrete tasks:**
- W13.1: Restructure existing `data/scripts/` into `src/spatialaudit/` per skeleton
- W13.2: Build `Audit` facade class
- W13.3: Sphinx + Furo + sphinx-gallery docs (4 tutorials)
- W13.4: pytest suite (>70% coverage)
- W13.5: GitHub Actions: test, docs, publish, pre-commit
- W13.6: Dockerfile + `make reproduce`
- W13.7: JOSS paper.md
- W13.8: Zenodo deposit on tag

---

## 5. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Counterfactual rewrites fail validation at scale | Medium | Pilot 25 first; iterate prompts; budget for 20% rejection |
| MugRep replication blows budget | High | Pre-commit to *partial* replication targeting qualitative claim |
| Theory step (Item 8) needs deeper math chops | High | Co-author dependency; have backup plan to ship as Proposition 1 only |
| Counterfactual annotation budget ($5–9k) | Medium | Cut to N=250 from N=500 if budget pressure; report Wilson CIs |
| EconML/sklearn version drift breaks pipeline | Low | Pin versions in `pyproject.toml`; uv.lock |
| `dml.sensemakr` rpy2 integration fails on local TeX | Low | Port the bias formula manually (~50 LOC, not hard) |
| JOSS reviewer asks for more tutorials/tests | Low | Standard; respond in revision |
| Brandon disengaged from theory section | Medium | Schedule explicit co-author meetings W11-W12 |

---

## 6. What This Buys at JBES

If all 9 items execute on schedule and content is sound:

- **Empirical foundation** moves from "single-city DML on N=995" to "DML + counterfactual experiments + replication of three competing models + simulation validation"
- **Methodology** moves from "we observed the frozen probe works" to "we proved why it works" (Proposition 2)
- **Robustness** moves from "Cinelli-Hazlett RV=0.654" to "OVB-DML RV + E-values + BSA + Manski bounds + Schuemie calibration + negative controls"
- **Reproducibility** moves from "scripts in a repo" to "pip-installable package + Zenodo DOI + JOSS-citable artifact + Dockerfile"

These are exactly the dimensions JBES reviewers grade. **My honest read: 60–70% acceptance odds after this work, vs the 20–30% the current paper has.**

---

## 7. Pre-Registration

Before generating counterfactuals or running placebos, **pre-register**:
- Counterfactual: rejection rules, intended ATE estimands (NDE on style-stripped, TE on submarket-swap), human-validation κ threshold
- Negative controls: NCO panel choice, NCE panel choice, calibration p-value threshold
- CATE: stratifier choice, BLP basis dimension, seeds

OSF or AsPredicted; ~2 hours of work; **massively strengthens the paper's credibility**.
