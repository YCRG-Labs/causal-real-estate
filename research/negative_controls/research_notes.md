# Negative Controls (NCO/NCE) — Research Dossier (Item 2)

For Item 2: placebo validation of the DML pipeline using negative-control outcomes (NCO) and exposures (NCE).

---

## A1. Foundational Literature

### Lipsitch, Tchetgen Tchetgen & Cohen (2010)
"Negative Controls: A Tool for Detecting Confounding and Bias in Observational Studies", *Epidemiology* 21(3):383-388. https://pmc.ncbi.nlm.nih.gov/articles/PMC3053408/

Seminal paper. Distinguishes:
- **NCO**: outcome that exposure cannot plausibly cause but which shares suspected confounders with the real outcome
- **NCE**: variable that cannot plausibly cause real outcome but is subject to similar selection / confounding pressures as real exposure

If pipeline returns non-null effect on either, identifying assumptions are likely violated.

### Shi, Miao & Tchetgen Tchetgen (2020)
"A Selective Review of Negative Control Methods", *Current Epidemiology Reports* 7(4):190-202. arXiv:2009.05641, doi:10.1007/s40471-020-00243-4. https://pmc.ncbi.nlm.nih.gov/articles/PMC8118596/

Modern reference. Formalizes assumptions (U-comparability, conditional independence given confounders). Connects negative controls to **proximal causal inference** — a *pair* of NCE+NCO can be used to *identify* the causal effect under unmeasured confounding via a "confounding bridge" function.

### Schuemie et al. (2014, 2016)
"Interpreting observational studies: why empirical calibration is needed to correct p-values" (*SiM* 2014, doi:10.1002/sim.5925) and "Robust empirical calibration of p-values using observational data" (*SiM* 2016, doi:10.1002/sim.6977).

Estimates **empirical null distribution** of test statistic across panel of negative-control hypotheses; calibrated p-values absorb both random and systematic error. Across pharmacoepidemiology benchmarks: ≥54% of nominal p<0.05 findings not significant after calibration.

### Recent advances (2022-2025)
- **Chernozhukov, Cinelli, Newey, Sharma, Syrgkanis (2022)** — "Long Story Short: Omitted Variable Bias in Causal Machine Learning", NBER WP 30302 / arXiv:2112.13398. https://arxiv.org/pdf/2112.13398. **OVB framework underlying `DoubleML` sensitivity API.** Negative controls plug in as benchmarks: explanatory power of placebo bounds strength of confounding the pipeline could plausibly miss.
- **Kummerfeld, Lim & Shi (2024)** — "Data-driven Automated Negative Control Estimation (DANCE)", *JMLR* 25:22-1062. https://jmlr.org/papers/v25/22-1062.html. First algorithm to *find and validate* candidate NCs from data.
- **Penning de Vries & Groenwold (2023)** — "Negative controls: Concepts and caveats", *Stat Methods Med Res*. https://journals.sagepub.com/doi/10.1177/09622802231181230. When NCs *fail* to detect bias.
- **Liu, Levis et al. (2023/2024)** — *Semiparametric Proximal Causal Inference*, JASA. https://www.tandfonline.com/doi/full/10.1080/01621459.2023.2191817. Cui, Pu, Shi, Miao, Tchetgen Tchetgen (2024), *AJE* "Regression-based Proximal Causal Inference". https://academic.oup.com/aje/article/194/7/2030/7775568. **DML-compatible nuisance estimation under proximal framework.**
- **Yu et al. (2024)** — "Advances in methodologies of negative controls: a scoping review", *J Clin Epidemiol* 166:111-119.
- **Norgaard, Rambachan et al. (2023/2024)** — "Negative Control Falsification Tests for Instrumental Variable Designs", arXiv:2312.15624. Conditions translate cleanly to DML.

---

## A2. Choosing NCOs for Real Estate

### Desiderata (Lipsitch 2010, Shi/Miao/Tchetgen Tchetgen 2020)
1. **A priori implausible causation** from exposure to NCO ("U-comparability")
2. **Shared confounding structure** — same latent U bias E[Y|T] and E[NCO|T]
3. **Observable, well-measured, non-degenerate** (variability comparable to Y)
4. **Pre-treatment** (or temporally fixed) — eliminates reverse causation

### Applied to candidates (text embedding T as exposure, log price Y as outcome)
- **Parcel area (lot size)**: cannot be caused by listing text — physically determined at subdivision time. Strongly pre-treatment. Confounding overlap with price is partial: parcel area co-varies with neighborhood (zoning, suburban-vs-urban) — exactly the unobserved-neighborhood-quality confounder we worry about. **Best NCO #1.**
- **Year built**: also pre-treatment, uncausable by listing copy. Co-varies strongly with neighborhood vintage and unobserved housing stock quality. **Strong NCO #2**, especially if text encoders pick up "vintage" / "historic" language.
- **Number of stories**: pre-treatment, uncausable. Discrete with low variability; confounding overlap weaker. **Weakest of three** — tertiary check only.

### Recommendation
**Primary NCO panel: {parcel area, year built}**, optionally stories as third. Reporting all three reinforces "pipeline doesn't always find nulls" because three placebos have *different* confounding strengths.

---

## A3. NCE Choice for High-Dimensional Embedding Treatments

Random row-permutation of embeddings is standard "label shuffle" placebo but has known weaknesses for embeddings: destroys both causal signal AND any *legitimate* correlation with covariates → test too easy to pass.

### Better-graded alternatives (increasing realism)
1. **Row-permuted embeddings (baseline)** — cheap, simple. Use it. Should produce null tightly centered on zero.
2. **Within-stratum permutation** — permute only within (year × ZIP × bed/bath) cells. Preserves marginal distribution of embedding conditional on coarse covariates; non-null more clearly attributable to spurious flexibility in DML nuisance learners.
3. **Synthetic embeddings from random text** — encode random Wikipedia or news paragraphs through same encoder. Vectors with same support, norm, within-encoder geometry as real listing embeddings but no semantic content tied to property. **Closer to true NCE under Lipsitch definition** — shares encoder-induced geometry but no causal channel.
4. **Pre-encoder Gaussian noise injection** — replace listing description with random tokens drawn from empirical token distribution, then encode.
5. **"Wrong corpus" embedding** — encode each row's listing with a *different* row's listing text, sampled from different city or year.

### Recommendation
**Report (1) plus either (3) or (5).** (3) is closer to true NCE.

### Stronger reading (proximal causal inference)
Treat parcel area as NCO and synthetic-text embeddings as NCE in a confounding-bridge equation to *identify* residual causal effect of text on price under unmeasured confounding (Shi/Miao/Tchetgen Tchetgen 2020, Cui et al. 2024). Overkill for first-pass rebuttal but worth citing as "stronger reading also goes through" footnote.

---

## A4. Calibration (Going Beyond Validation)

Standard validation use: estimate effect on each placebo, check whether CI contains zero. **Calibration** turns placebo panel into a correction.

### Schuemie's recipe
1. Run full DML pipeline on panel of K negative-control hypotheses (K=10–50 placebo outcomes, or K independent permuted-embedding draws)
2. Collect K point estimates and standard errors
3. Fit **Gaussian empirical null** $N(\mu, \sigma^2 + \tau^2)$ where $\tau^2$ is systematic-error variance beyond nominal SE
4. **Calibrated p-value** for real estimate: tail probability under empirical null

`EmpiricalCalibration` R package (CRAN: https://cran.r-project.org/package=EmpiricalCalibration; source: https://github.com/OHDSI/EmpiricalCalibration). Implements `fitNull()`, `calibrateP()`, `calibrateConfidenceInterval()`.

### DML-specific calibration
Cross-walk Schuemie's empirical null with **Chernozhukov et al. 2022 OVB sensitivity bounds**: report **robustness value RV** — strength of unmeasured confounding (parameterized by partial $R^2$ of nuisance functions on residualized outcome and treatment) needed to overturn null. Benchmark RV against partial-$R^2$ achieved by placebo treatments. DoubleML implements this: https://docs.doubleml.org/stable/guide/sensitivity.html

---

## A5. Code and Packages

| Package | Lang | What it gives | URL |
|---|---|---|---|
| `EmpiricalCalibration` | R | Empirical null fitting, calibrated p-values/CIs | https://cran.r-project.org/package=EmpiricalCalibration |
| `DoubleML` | Py + R | DML estimation; built-in OVB sensitivity (Chernozhukov 2022); easy synthetic placebo column | https://docs.doubleml.org/stable/ |
| `EconML` | Py | DML, DR-Learner, Causal Forest; placebo via `BootstrapInference` and refutation hooks | https://github.com/py-why/EconML |
| `DoWhy` | Py | `refute_estimate` API: built-in "placebo treatment" and "random common cause" refuters → direct NCE/NCO checks | https://github.com/py-why/dowhy |
| `proximalDML` | R | Confounding-bridge / proximal estimators (Cui, Pu, Shi, Miao, Tchetgen Tchetgen) | https://github.com/yifan-cui/Proximal-Inference |
| Sentinel DANCE | R/Py (in dev) | Data-driven NCO discovery (Kummerfeld 2024) | https://www.sentinelinitiative.org/methods-data-tools/methods/evaluate-use-dance-algorithm-find-disconnected-negative-controls-and |

### Cleanest path for our pipeline
`DoubleML` (already used) → run same `DoubleMLPLR` with (a) `y` swapped for parcel-area, (b) `d` swapped for shuffled-embedding columns → call `.sensitivity_analysis()` for Chernozhukov OVB benchmark → pass K placebo estimates into `EmpiricalCalibration::fitNull()` for calibrated p-value.

---

## Concrete Next Steps
1. Build NCO panel: parcel area (primary) + year built (secondary) + stories (tertiary). Re-run DML with `y` swapped to each; expect tight nulls.
2. Build NCE panel: row-permuted embeddings + synthetic-text embeddings (Wikipedia paragraphs through same encoder). Run K=20+ replicates, feed K coefficient estimates into `EmpiricalCalibration::fitNull()` for calibrated p-value on real estimate.
3. Add Chernozhukov 2022 OVB sensitivity via `DoubleML.sensitivity_analysis()`; report robustness value with placebo partial-$R^2$ as benchmark.
4. Cite proximal causal inference machinery (Shi/Miao/Tchetgen Tchetgen 2020; Cui et al. 2024) as "stronger identification also goes through" footnote.
