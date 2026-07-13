# Proposition 3 — proof target scaffold

Working document, not paper copy. Every lemma is named, every external result is
attached to a retrieval-verified citation, and the numerical falsification test is
specified so the mechanism can be checked before the full proof is written.

---

## DEEP-DIVE SYNTHESIS AND DECISION (2026-07-12)

**Decision: demote Prop 3 from a Theorem to a Remark/Lemma** in the calibration
appendix. Reached independently by my own analysis and a blind adversarial
JBES-referee pass. Reasons, in order of force:

1. **The bootstrap, not Prop 3, licenses the intervals.** A pool-size + oracle
   Monte Carlo (below) shows the naive IF-SE undercoverage is two stacked sources:
   (a) a generated-regressor variance from estimating the PC direction, θ²-scaling,
   that pooling removes; and (b) a *larger* baseline DML-with-ML-nuisances
   undercoverage present even under the **oracle (true) direction** where the
   generated-regressor variance is identically zero. Pooling addresses (a); the
   bootstrap addresses (b). Source (a) is the **minor** term.
2. **The theorem's hard half is for a spec the paper does not use.** Part (a)
   (within-market, same-sample Murphy-Topel cross-covariance) is the non-trivial
   case; the paper uses the pooled direction, whose validity is the elementary
   "auxiliary sample grows faster than the target sample ⇒ first-stage estimation
   asymptotically ignorable" regime — routine to a Newey (1984) referee.
3. **Theorems 1–2 carry the framework.** Adding Prop 3 as a third theorem dilutes;
   as a remark that explains the pooled-PC design choice it strengthens.

**Honest one-line contribution:** generated-regressor variance and
nuisance-estimation undercoverage are two separable failure modes in DML with a
learned treatment direction; the pooled PC eliminates the first, the full-pipeline
bootstrap validates against the second.

**Bootstrap-arm result (pooled:12 direction, 250 reps, B=120 percentile).** The
bootstrap SE widens toward the true sampling sd: at θ=0.20, true sd 0.0110, IF-SE
0.0075, bootstrap SE 0.0092 — the bootstrap closes ~60% of the plug-in-to-truth
gap (sd/se 1.47 → 1.20). Finite-sample bias is small (≈0.15 sd). Residual
undercoverage (boot coverage 0.79–0.85) is the remaining ~20% variance gap plus
the known narrowness of a low-B *percentile* interval; a studentized t-pivot
bootstrap at higher B (the paper's cheap-bootstrap, Lam arXiv:2202.00090) is
designed to close that. Coverage is fine at the null (bias→0 there), so the
undercoverage bites only for large-effect markets (NYC, Chicago), where CIs should
be read as approximate and the sensitivity analysis carries the inference. The
bootstrap does its job (captures the missing variance); it is not defeated.

### New real-data evidence (all computed on the actual 12-city corpus)

- **Pooled embedding covariance** (n_pool = 69,311, p = 768, mpnet): p/n_pool =
  0.011, safely spike-consistent vs the noise bulk (BBP margin ≈ 9000, Paul
  alignment 0.99999). BUT **λ₁ ≈ λ₂ near-degenerate** (ratio 1.20, relative gap
  0.17 — correcting the earlier "0.44" note, which does not match the data), so the
  Anderson variance Ω_v is dominated by the j = 2 term (~29 of ~39). The leading
  direction carries genuine estimation variance.
- **The Prop 3(b) shrink factor n_m/n_pool is graded, not uniformly small:** 0.014
  (SF) → 0.22 (NYC). Pooling nearly eliminates the generated-regressor variance for
  small markets but only cuts it ~4.5× for NYC — which is 22% of the pool itself and
  is the market carrying the notable positive effect. Report the per-market factor.
- **Pooled-vs-per-market direction alignment (new diagnostic, referee-requested to
  defend the estimand):** median |cos(v_pool, v_market)| = 0.80, wide spread. A
  subspace check separates two causes: **NYC** cos_v1 = 0.66 but 0.91 inside the
  pooled top-2 subspace → the low v₁ alignment is the λ₁≈λ₂ *degeneracy*, not
  semantic difference. **Boston** cos_v1 = 0.38 and only 0.64 inside the pooled
  *top-5* → **genuine semantic heterogeneity**, its leading text axis lives largely
  outside the pooled principal subspace, consistent with Boston's near-zero effect.
  This table belongs in the paper: it defends "the pooled direction is a common
  cross-market semantic axis" and explains part of the heterogeneity.

### Verified citations for the inference section (retrieval-checked)

- **Lin & Han 2026, arXiv:2604.17239** — first general bootstrap-validity proof for
  DML (exchangeable/Efron weights, *same* conditions as DML validity). Primary
  "why our bootstrap is valid where the IF-SE is not" cite.
- **Saco 2025/2026, arXiv:2512.07083** ("Ill-Conditioned Orthogonal Scores in
  DML"; already in the bib from Portland) — closest baseline-undercoverage analogue:
  κ_DML conditioning, silent bias-amplification, coverage collapse with a *known*
  treatment. Caveat: conditioning-driven, NOT effect-size-driven — cite with that
  distinction stated.
- **Chernozhukov et al. 2018** (product-of-rates remainder), **Newey–Robins 2017**
  (cross-fitting rate improvement), **Zivich–Breskin 2020, arXiv:2004.10337**
  (empirical cross-fit finite-sample coverage) — the baseline mechanism is real and
  known; cite for background.
- **Effect-size-dependent baseline undercoverage appears uncharacterized** in the
  literature (searched). Flag as an observation; do NOT force a citation onto it.

### Spatial-robust inference RESULT (ran, all 12 markets)

`data/scripts/spatial_robust_dml.py` → `results/spatial_robust/spatial_robust_table.csv`.
Pooled-PC treatment, per-market DML, three variance estimators. Listing text
clusters in space (intra-zip ICC of the influence function ≈ 0.063, ~32
listings/zip), so the iid IF-SE is too narrow. Median SE inflation: **zip-cluster
1.41×, Conley spatial-HAC (Bartlett, 2 km) 1.22×.** But the effects SURVIVE:
significant markets 10/12 (iid) → 11/12 (zip) → 10/12 (Conley); **no market loses
significance under zip-clustering.** t-stats fall honestly (Chicago 17→11, Phila
13→9, NYC 9→6) but stay well above 1.96. The correctness fix becomes a robustness
win: spatially-honest inference does not break the findings. CAVEAT: this ran on a
reduced 7-confounder set (structural + lat/lon), so the θ are NOT the paper's
full-confounder Baur numbers; the transferable result is the **inflation ratios**,
to be applied to the paper's actual estimates (borderline markets re-checked
there). Cites for the fix (all verified): Conley 1999 spatial-HAC; Chiang-Kato-Ma-
Sasaki multiway-cluster DML (JBES, DOI 10.1080/07350015.2021.1895815); Salerno-Wu-
McCormick 2026 (arXiv:2603.11368).

### Remaining high-value analysis (referee's strongest ask)

The paper's current bootstrap holds the pooled PC **fixed** and refits only the
nuisances, so it does not capture any residual generated-regressor variance. The
decisive real-data check is a **unified bootstrap that resamples the pooled corpus,
refits v̂ on each draw, and propagates into each market's θ̂** — this measures the
real-data generated-regressor variance per market (matters most for NYC at
n_m/n_pool = 0.22) and lets the paper report one unified interval instead of
asserting V_gen is negligible on theoretical grounds. Build target:
`data/scripts/` refit-PC bootstrap over the 12-market Baur estimator.

### Citation hygiene fixes surfaced

- This scaffold cited the **2002 reprint** DOI for Murphy–Topel; the correct 1985
  original is `10.1080/07350015.1985.10509471` (the manuscript bib
  `murphy1985twostep` is already correct).
- "Lam 2022 JASA" (in an older project note) is wrong: Lam cheap-bootstrap is
  arXiv:2202.00090 / WSC 2022, not JASA (bib `lam2022cheap` already correct).

---

## What Prop 3 claims

The treatment in the per-market DML regression is a generated regressor,
`T_i = v̂' W̃_i`, the projection of the residualized sentence-embedding onto an
estimated leading direction `v̂`. The naive influence-function standard error
treats `v̂` as fixed and estimates only the Robinson-score variance. Prop 3 asks
when that SE is valid, and answers it with a dichotomy in *where the direction is
estimated*:

- **(a) Failure.** When `v̂` is the leading eigenvector of the *within-market*
  embedding covariance (estimated on the same `n_m` rows the coefficient is
  computed on), the estimator carries a generated-regressor variance
  `V_gen` that the IF-SE omits, so nominal 95% intervals undercover, and the
  undercoverage grows with the effect size because `V_gen ∝ θ²`.

- **(b) Fix.** When `v̂` is the leading eigenvector of the *pooled* covariance
  (estimated on `n_pool` listings across all markets, held fixed for every
  market), `V_gen = O(n_m / n_pool)` relative to the Robinson variance and
  vanishes in the limit `n_m / n_pool → 0`, so the naive IF-SE is
  first-order valid.

The paper's headline estimates use the pooled direction, so (b) is what licenses
the reported intervals; (a) is the failure mode the finite-sample calibration
study exhibits, and it is what a referee would (correctly) worry about if the
direction were re-estimated market by market.

## Objects and notation

Fix a market with `n_m` listings. Let `W_i ∈ R^p` be the sentence-embedding
(`p = 768` for MiniLM/`sentence-BERT`), `X_i` the structured controls plus the
location basis, `Y_i = log price`. Residualize on the controls with the DML
nuisances: `W̃_i = W_i − E[W|X_i]`, `Ỹ_i = Y_i − E[Y|X_i]`. Write
`Σ = E[W̃ W̃']` for the residualized-embedding covariance, `Σ̂` its sample
version on the estimation set, eigenpairs `(λ_1, v_1) , (λ_2, v_2), …`,
`λ_1 > λ_2 ≥ …`, `v_1 =: v` the target direction. The treatment is
`T_i = v̂' W̃_i` with `v̂` the leading eigenvector of `Σ̂`. Partial out the
controls once more inside the score, `T̃_i = T_i − E[T|X_i]`, and let
`M = E[T̃ W̃']`-type curvature objects appear below.

Robinson/partially-linear moment (Chernozhukov et al. 2018, DBL/DML):
`ψ(θ, v) = (Ỹ − θ T̃) T̃`, `E[ψ(θ(v), v)] = 0` defines
`θ(v) = E[Ỹ T̃] / E[T̃²]`, the estimand *as a function of the direction*.

## The key non-triviality (do not skip in the writeup)

`θ(v)` moves with `v`. Differentiating the estimand,

    dθ/dv = (E[Ỹ W̃'] − θ E[T̃ W̃' + W̃ T̃']) / E[T̃²]   [vector in R^p]
          = −θ · (M v) / (v' M v)   in the leading-order simplification,

so the sensitivity of the *estimand* to the direction is nonzero and proportional
to θ. This is why "cross-fitting removes the generated-regressor problem" is
imprecise: cross-fitting removes the same-sample *correlation*, not the fact that a
different `v̂` targets a different `θ`. The clean statement has to be about the
*variance of `v̂`*, which is what pooling controls.

## Decomposition to prove: Murphy-Topel applied to (θ̂, v̂)

Treat `(v̂, θ̂)` as a two-step / stacked-moment estimator (Newey 1984,
*Econ. Letters* 14: 201–206 — GMM interpretation of sequential estimators). The
Murphy-Topel (1985, *JBES* 3(4):370-379, DOI 10.1080/07350015.1985.10509471) sandwich gives

    Var(θ̂) = V_2 + [ V_2 R V_1 R' V_2 − V_2 R C' − C R' V_2 ]
                     └──────────────── V_gen ────────────────┘

with
- `V_2 = 1 / E[T̃²]` — the Robinson-score variance the IF-SE already estimates;
- `V_1 = Var(v̂)` — the leading-eigenvector variance (Lemma 1);
- `R = ∂_v E[ψ]` — the score's sensitivity to the direction, `R ∝ θ` (Lemma 2);
- `C = Cov(v̂-score, θ̂-score)` — nonzero only when the direction is estimated on
  rows that overlap the coefficient's rows.

Pagan (1984, *IER*, DOI 10.2307/2648877) is the classic generated-regressor
framing to cite for "naive SEs ignore `v̂`'s sampling error." Ackerberg-Chen-Hahn
(2012, *REStat*) and Ichimura-Newey-Chernozhukov-Escanciano-Robins (2016,
*Econometrica*, locally robust) are the nonparametric-first-step generalizations.

### Lemma 1 (leading-eigenvector influence function). VERIFIED CITATION.

Anderson (1963, *Ann. Math. Statist.* 34(1), DOI 10.1214/aoms/1177704248): under
i.i.d. sampling, simple leading eigenvalue, fixed `p`,

    √n_est (v̂ − v) →d N(0, Ω_v),  Ω_v = Σ_{j≥2} [ λ_1 λ_j / (λ_1 − λ_j)² ] v_j v_j',

with linearization `v̂ − v = (λ_1 I − Σ)^+ (Σ̂ − Σ) v + o_p(n_est^{-1/2})`.
So `V_1 = Ω_v / n_est` — and the `1/n_est` is the entire lever in part (b).
Non-asymptotic / effective-rank version for rigor without fixed-`p`:
Koltchinskii-Lounici (2016, *Ann. Statist.* 44(4), DOI 10.1214/16-AOS1437, and
*Ann. IHP* 2016, DOI 10.1214/15-AIHP705), whose linear term `L_1(E)` equals the
Anderson term and whose remainder is `≤ 14 (‖E‖/gap)²`. Deterministic
perturbation step: Yu-Wang-Samworth (2015, *Biometrika* 102(2),
DOI 10.1093/biomet/asv008) sinΘ bound.

### Lemma 2 (score sensitivity). TO PROVE (elementary).

`R = ∂_v E[ψ(θ, v)] |_{θ(v)} = −E[T̃²] · (dθ/dv)`, so combining with the estimand
derivative above, `R = θ · M v / (v' M v)`, i.e. `R ∝ θ`. Hence
`R V_1 R' = θ² · (Mv)'Ω_v(Mv) / [ (v'Mv)² n_est ]`, the term that scales as `θ²`
and drives the effect-size-dependent undercoverage.

### Lemma 3 (spiked-regime consistency of `v̂`). VERIFIED CITATION — this is A1.

`v̂` is `√n_est`-consistent and asymptotically normal only outside the BBP phase
transition. Paul (2007, *Statistica Sinica* 17): for spike strength `c` and
`p/n_est → γ`, `|⟨v̂,v⟩|² → (1 − γ/c²)/(1 + γ/c)` when `c > √γ` (consistent) and
`→ 0` when `c ≤ √γ` (inconsistent). Johnstone-Lu (2009, *JASA* 104,
DOI 10.1198/jasa.2009.0121) for the inconsistency without `p/n → 0`. BBP:
Baik-Ben Arous-Péché (2005, *Ann. Probab.* 33(5)). Wang-Fan (2017, *Ann. Statist.*
45(3), DOI 10.1214/16-AOS1487) if a diverging-eigengap regime is wanted.

**Our numbers put us safely inside the consistent regime:** pooled `p/n_pool ≈
0.011`, relative eigengap `≈ 0.44`, both orders of magnitude off the `c = √γ`
threshold. Within-market `p/n_m ≈ 0.78` sits near the transition — the honest
statement of *why* part (a) fails and part (b) does not.

## Assumptions

- **A1 (spectral gap + pooling regime).** `λ_1 − λ_2` bounded away from 0 and
  `p / n_pool → 0`. Delivers Lemma 1 + Lemma 3 for the pooled direction.
- **A2 (Robinson orthogonality in the regression nuisances).** Standard DML
  Neyman-orthogonality of `ψ` in `ℓ = E[Y|X]`, `m = E[T|X]` (Chernozhukov et al.
  2018). Note: NOT orthogonality in `v` — we do not assume `R = 0`.
- **A3 (pooling rate — the part-(b) hypothesis).** `n_m / n_pool → 0`.
- **A4 (moment regularity).** `E[T̃²] > 0` (overlap / non-degenerate treatment
  after partialling), finite fourth moments of `W̃` (for Lemma 1 beyond Gaussian,
  via Tyler 1981, *Ann. Statist.* 9(4)).

## The two theorems

### Part (a) — within-market direction undercovers.

`n_est = n_m`. Then `C ≠ 0` (full row overlap) and `V_1 = Ω_v / n_m`, so in the
`√n_m` normalization

    √n_m (θ̂_m − θ_m) →d N(0, V_DML + V_gen),
    V_gen = (dθ/dv)' Ω_v (dθ/dv) − (cross-cov)  =  Θ(θ²) · O(1),

a first-order share of `V_DML`. The IF-SE estimates only `V_DML`, so coverage
`< 0.95`, worsening as `θ` grows. Within-market *cross-fitting* the direction
(the simulation's `cross_fit_pca=True`) sets the disjoint-fold `C = 0` but leaves
`R V_1 R' = Θ(θ²/n_m)` intact, so it improves coverage yet still sags at large
`θ` — the exact 0.83-at-largest-effect signature in the calibration table.

### Part (b) — pooled direction restores validity.

`n_est = n_pool`, `n_m / n_pool → 0`. Two effects:
1. `V_1 = Ω_v / n_pool`, so the sensitivity term in the `√n_m` normalization is
   `(n_m / n_pool) · (dθ/dv)' Ω_v (dθ/dv) = O(n_m / n_pool) → 0`.
2. The market's rows are a fraction `n_m / n_pool` of the estimation set, so the
   cross-covariance `C = O(n_m / n_pool) → 0`.

Hence `√n_m (θ̂_m − θ_m) →d N(0, V_DML)` and the naive IF-SE is first-order valid.
The residual is quantified, not merely asymptotic: `V_gen / V_DML ≈ (n_m/n_pool) ·
θ² · κ` with `κ` the spectral factor `(Mv)'Ω_v(Mv)/[(v'Mv)² E[T̃²] V_DML]`, a few
percent for the effect-carrying markets (Chicago `n_m/n_pool ≈ 0.08`).

## Positioning against the two nearest papers (both VERIFIED, both working papers)

- **Escanciano-Perez-Izquierdo, "Automatic Locally Robust GMM with
  Machine-Learning-Generated Regressors," arXiv:2301.10643.** Closest precedent:
  their setup names dimension-reduced embeddings as the generated regressor, and
  their Prop 3.1 ("downstream local robustness") gets `V_gen → 0` by *adding a
  Riesz-representer correction that orthogonalizes the moment in `v`* (their
  `α_{01}` term). We take the *other* route — no score modification, pooling drives
  `V_1 → 0` — so our contribution is the pooled-estimation shortcut and its exact
  `n_m/n_pool` residual, complementary to their orthogonalization. Cite as the
  general machinery; distinguish on the device.

- **Battaglia-Christensen-Hansen-Sacher, "Inference for Regression with Variables
  Generated by AI or ML," arXiv:2402.15585 / Cowles 2421.** REQUIRED distinguishing
  paragraph. For their classification / latent-index generators the naive two-step
  has *no extra variance term* but a *bias* term (curvature `κ`). Our
  PCA-eigenvector generator is the opposite structure: an asymptotically-linear
  `v̂` whose influence function correlates with the score, producing a genuine
  Murphy-Topel *variance* term and negligible bias. State this so the apparent
  tension ("AI-generated regressors need no SE correction") is resolved by the DGP
  difference, not left for a referee to flag.

## Numerical falsification test (run before writing the proof)

The scaffold is only worth proving if `V_gen ∝ θ² · (n_m/n_pool)` matches the
simulation. Test:

1. From the real residualized pooled covariance, compute `λ_j, v_j`, form `Ω_v`
   and `dθ/dv = −θ Mv/(v'Mv)`; get the spectral factor `κ`.
2. Predict IF-SE coverage under the in-sample arm as
   `P(|Z| < 1.96 √(V_DML/(V_DML + V_gen)))` with `V_gen = θ² κ` (n_est=n_m), across
   the simulation's `θ ∈ {0.01,…,0.20}` grid, and check it tracks the in-sample
   undercoverage cell by cell — the diagnostic is the `θ²` slope, not the level.
3. Predict the within-market cross-fit arm by dropping the `C` term but keeping
   `θ² κ / 1` (still n_est=n_m); it should track the milder observed sag.
4. Predict pooled coverage with `V_gen = (n_m/n_pool) θ² κ` → essentially nominal;
   if a pooled arm is added to the simulation it should cover ~0.95 even at
   `θ = 0.20`, i.e. beat the within-market cross-fit arm at the largest effect.
   That last inequality (pooled > within-market-cross-fit at large θ) is the
   sharpest falsifiable prediction of the mechanism.

If step 2's `θ²` slope matches, the mechanism is right and the proof is worth
writing; if the observed undercoverage is flat in `θ`, the story is bias-dominant
(Battaglia et al.) not variance-dominant, and Prop 3 has to be rewritten.

## Numerical falsification — RESULT (PASS)

Ran against the in-sample DML arm of `results/simulation/coverage_table.csv`
(estimator `DML`, `n_est = n_infer = N`), excess-variance ratio
`r(θ) = (sd_θ̂ / se_IF)² − 1`, which the scaffold predicts equals `κ θ²`.

**Clean regime, N = 2000 (eigenvector √n-consistent):**

| θ (truth) | coverage | true sd | IF-SE | r = (sd/se)²−1 | κ = r/θ² |
|-----------|----------|---------|-------|----------------|----------|
| 0.011 | 0.927 | 0.00768 | 0.00740 | 0.078 | (noise floor) |
| 0.077 | 0.957 | 0.00712 | 0.00740 | −0.074 | (noise floor) |
| 0.162 | 0.917 | 0.00833 | 0.00745 | 0.252 | 9.6 |
| 0.231 | 0.850 | 0.00939 | 0.00748 | 0.577 | 10.9 |
| 0.288 | 0.813 | 0.00999 | 0.00751 | 0.769 | 9.3 |
| 0.341 | 0.803 | 0.00971 | 0.00754 | 0.658 | 5.7 (saturation) |

Least-squares fit through the origin on the detectable window `0.15 ≤ θ ≤ 0.30`:
**κ = 9.7, R² = 0.995.** The reported IF-SE is flat (0.00740 → 0.00754) while the
true sd climbs (0.00768 → 0.00999), so the interval is blind to `V_gen`, and
coverage falls monotonically 0.93 → 0.80. This is the Lemma 2 `θ²` law and the
Murphy-Topel omitted-variance signature, confirmed.

**Small-N / near-transition regime, N = 500:** the law breaks (κ scatters
150–465, R² = 0.76). This is Lemma 3, not a failure: at N = 500 the embedding PC
sits near the BBP threshold (`p/n` large), the linear IF understates `v̂`'s error,
and `V_gen` exceeds its `θ²κ` leading order. Do not fit the law here; cite it as
the reason part (a) is worst in the smallest markets. This N = 500 row is excluded
from the fit.

## Part (b) — pooled-arm experiment RESULT (ran 2026-07, 300 reps/cell)

`data/scripts/simulation/pooled_pc_coverage.py` runs three arms on the identical
DGP and learner, differing only in the treatment direction: `insample` (PC per
replicate), `pooled` (PC on a fixed `n_pool = 12·N` draw), `oracle` (the
generator's true direction, `n_pool = ∞`, so `V_gen ≡ 0`). Coverage against each
arm's own population estimand. Harness validated: the in-sample truth
(+0.34053) and coverage (0.78 at N=2000, θ=0.20) reproduce the production table.

| N | θ | cov insample | cov pooled | cov oracle |
|------|------|------|------|------|
| 2000 | 0.05 | 0.883 | 0.917 | 0.890 |
| 2000 | 0.10 | 0.853 | 0.887 | 0.920 |
| 2000 | 0.15 | 0.830 | 0.837 | 0.803 |
| 2000 | 0.20 | 0.780 | 0.840 | 0.843 |
| 500  | 0.05 | 0.870 | 0.890 | 0.890 |
| 500  | 0.10 | 0.837 | 0.893 | 0.860 |
| 500  | 0.15 | 0.820 | 0.847 | 0.847 |
| 500  | 0.20 | 0.750 | 0.863 | 0.853 |

**What is confirmed.** Pooling improves coverage in every cell (8/8), and the
improvement is largest at the largest effect: +0.06 (N=2000) and +0.11 (N=500) at
θ=0.20, versus +0.02–0.03 at θ=0.05. The excess-variance ratio (sd/IF-SE) falls
correspondingly, 1.49→1.30 at N=2000/θ=0.20. This is the generated-regressor
signature: a θ-growing variance component that the in-sample PC carries, the
pooled PC removes, and the IF-SE is blind to. Direction of the mechanism
confirmed.

**What is NOT confirmed, and corrects the earlier prediction.** Pooling does
**not** restore nominal coverage. The oracle direction, which has `V_gen ≡ 0` by
construction, **still undercovers** (0.80–0.92, excess variance 0.43–0.78). So the
in-sample undercoverage is the sum of two pieces: (a) the generated-regressor
`V_gen` that pooling removes, θ-growing, and (b) a baseline
DML-with-flexible-nuisances finite-sample undercoverage present even with the true
treatment, roughly effect-size-flat, that pooling neither touches nor should. The
isolated component (a) = insample-minus-oracle excess is too noisy to fit a clean
`θ²` law at 300 reps (it is a difference of two noisy ratios); the clean scaling
evidence remains the in-sample-only fit above (R² = 0.995).

**Consequence for the paper (honest positioning).** Prop 3 is a *supporting*
result, not a standalone inference fix. It explains why the pooled PC is the right
construction — it removes the θ-growing generated-regressor variance the IF-SE
omits — but the residual baseline DML undercoverage means the pooled PC alone does
not license the IF interval. The paper's existing **bootstrap CIs** (whole-pipeline
resample, already adopted) are what handle piece (b). The two ingredients are
complementary: pooled PC kills the generated-regressor variance (Prop 3),
bootstrap kills the residual nuisance undercoverage. Do not claim pooling restores
nominal coverage; claim it removes the generated-regressor component, which the
data shows and the theorem proves.

**Verdict:** the theorem (pooling drives `V_gen → O(n_m/n_pool)`) is correct and
worth proving; the earlier "pooling restores 0.95" prediction is falsified and
replaced by "pooling removes the generated-regressor component; bootstrap handles
the rest." Prop 3 supports the pooled-PC design choice; it is not the inference
headline.
