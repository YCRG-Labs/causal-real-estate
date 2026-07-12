# Referee 2 Report — Round 1

**Manuscript:** Does Listing Language Add Value Beyond Location? A Design-Based Audit of Text Embeddings in Automated Valuation
**Target:** Journal of Business & Economic Statistics
**Date:** 2026-07-12
**Protocol:** Five-audit systematic replication (Code, Cross-Language Replication, Directory, Output Automation, Econometrics). Author code was read and run but never modified; all replication scripts are new files under `data/scripts/replication/`.

## Summary

The paper asks whether the price effect of listing text in an automated valuation setting reflects semantic content or spatial confounding laundered through language, and answers it with a partially-linear DML design across twelve metros, backed by a concept-erasure probe, a counterfactual-rewriting intervention, a finite-sample coverage simulation, and a new composition-and-conditioning diagnostic (CoCA). The methodology is unusually thorough for the question and the honest limitations section preempts most of the obvious referee objections. The estimation arithmetic is sound where I could test it independently: the two-way cluster-robust variance reproduces bit-for-bit, the CoCA proposition checks out symbolically with no tolerance games, and the per-market Baur DML replicates directionally in every city I re-implemented from scratch. However, two findings reach the paper's actual claims and require a revision before I can recommend acceptance: (i) the treatment vector feeding the flagship pooled-PCA table is joined to the outcome by bare row position across parquet files that were rewritten after the treatment CSV was frozen, so the alignment is provably broken in four of twelve cities and unverifiable in the other eight; and (ii) the within-category demeaning module that carries the composition-confound narrative runs its headline theta comparisons on the exact zero-filled feature block its own docstring flags as a property-type leak. Both are fixable, but until they are fixed the composition finding and the precise headline decimals are not established. **Verdict: Major Revisions.**

## Audit 1: Code Audit

Nine scripts requested; seven run live, two (`build_active_confounders.py`, `attach_crime_to_parcels.py`) read-only because they mutate production `*_parcels_micro_geo.gpkg` in place. All test outputs restored to committed state (git diff clean).

### HIGH

**1.1 — Silent listing_id positional misalignment in the headline pooled-PCA table.**
`pooled_pca_treatment.py:41-44` and `baur_pooled_pca.py:38-59` both build `listing_id` as `np.arange(len(df))` over a fresh read of `{city}_embeddings.parquet` — pure row position, no natural key. `results/replications/pooled_pca_treatment.csv` was written Jun 13 17:13; the embeddings parquets were fully rewritten (`to_parquet(index=False)`) Jun 13 21:07–21:59, i.e. after the CSV was frozen. Current row counts vs. the cached CSV:

```
nyc:      pool=15,240  vs  emb=15,270   (Δ30)
chicago:  pool=5,606   vs  emb=5,607    (Δ1)
atlanta:  pool=6,139   vs  emb=6,140    (Δ1)
phoenix:  pool=7,089   vs  emb=7,090    (Δ1)
```

Four of twelve cities are provably drifted; the other eight cannot be disproven because no content key exists to check row-for-row identity. `baur_pooled_pca.py:51` merges `how="left"` and silently fills unmatched rows with `treatment_z=0.0` — a live NYC run reports "30 listings without pooled-PCA score; filling with city mean (0 after z-score)." Beyond the dropped tail, a wholesale parquet rewrite could have reordered or inserted rows mid-file, which would swap treatment scores between unrelated listings rather than merely drop a tail. This feeds `results/replications/baur_pooled_pca/*.json`, the flagship 12-city table.
*Corroboration:* Audit 2 discovered the Chicago Δ1 independently via a clean-room join; two orthogonal implementations landing on the same defect raises confidence that it is real, not a reading artifact.
*Magnitude:* Audit 2's independent recomputation shows the effect on θ is negligible where checked (1–30 rows out of thousands), so this is very unlikely to overturn any point estimate. The problem is reproducibility integrity, not a wrong headline number: the merge has no validation guard, and the alignment of the matched majority is unverifiable.
*Fix:* regenerate `pooled_pca_treatment.csv` from the current parquets immediately before each run, or key the join on a stable content column already in the schema (`url` or `source_html_sha256`) so drift becomes an assertion failure instead of a silent partial merge. `baur_pooled_pca.py:111-113`'s assert only checks array lengths post-filter and cannot catch a length-preserving misalignment — replace it with a key-equality check.

**1.2 — The demeaning DML uses the feature block its own docstring flags as a property-type leak.**
`within_category_demean.py:32-36` warns that a zero-fill "would turn into a perfect land classifier," and computes the AUC diagnostic (line 159) on a separate median-imputed block to avoid exactly that. But the actual headline theta comparisons — `raw`, `dummy_in_ridge`, `demean` at lines 161-164 — all consume `_zerofill_block` (line 142), the leaky block. On DC, non-residential listings are missing beds/baths/sqft 96–98% of the time vs. 2–4% for residential, so the zero-fill encodes property type directly inside the X block the composition story claims is blind to it. This means the composition module's central comparison is run on contaminated controls.

**1.3 — Demeaning does not attenuate θ in most cities, contradicting the module's premise.**
Running `--all_12`: 9/12 cities show demeaning *increasing* |θ| vs. raw on the fixed scale (DC: raw 0.044 → dummy 0.092 → demean 0.169). Only Chicago and Phoenix show the intended attenuation. Cross-city Spearman of %non-residential vs. attenuation is −0.06 (k=11), i.e. no relationship. Most plausibly a downstream symptom of 1.2; fix 1.2 and re-run before citing this module as evidence for the composition narrative.

**1.4 — CoCA Step-2 detection is bimodal and the single summary stat masks it.**
At weak composition strength (a=1.0) Step-2 detection is 0/0/0/2.5% across the four category shares; at a=2.0 it is 85–100%. The reported `composition_step2_detection = 51.2%` averages these and reads as "moderate power" when the truth is "no power when confounding is weak, full power when strong." If the paper quotes 51.2% without the strength breakdown it is materially misleading.

**1.5 — Shen Doc2Vec is not reproducible under its own documented seed.**
`shen_2021.py:131-154` sets `Doc2Vec(..., workers=min(cpu_count(),8), seed=seed)`; gensim does not guarantee seed-reproducibility with `workers>1`. Two identical `--city sf --doc2vec --fast --n_boot 0` runs at `--seed 42` returned θ=+0.1268 and θ=+0.1168, a ~9% swing in the headline coefficient. Force `workers=1` if the quoted decimals are a claim, or caveat the Doc2Vec Shen numbers as reproducible to ~1 significant figure. (Note: this does not revive the earlier TF-IDF "collapse" scare — the committed per-city JSONs confirm production ran `--doc2vec`; it is a run-to-run jitter issue, not a specification issue.)

### MEDIUM

**1.6** — `soldprice_twoway_cluster.py:140-144`: 60,516/192,729 pooled rows (31.4%) have ≥1 NaN feature silently median-imputed with no diagnostic. `median_gross_rent` alone is NaN for 42,908 rows (22.3%), far exceeding the ~290 true census-join failures the `demo_missing` flag tracks — an ACS-suppression pattern is being imputed with no "this was imputed" signal reaching the ridge nuisance.

**1.7** — `build_active_confounders.py:56-57`: `cKDTree` nearest-assessor match has no distance cutoff; a listing outside assessor coverage silently takes the closest parcel however far, with no distance diagnostic.

**1.8** — `baur_pooled_pca.py:111-113`: the alignment assert checks only array lengths, giving false confidence against 1.1.

### LOW / style
- `within_category_demean.py:99-105` — `np.atleast_2d(M.T).T` reshape is harmless but confusing.
- `within_category_demean.py:121` — `n_pca=1` is dead code in the `use_ridge=True` branch.
- `soldprice_twoway_cluster.py:158` — `K=X.shape[1]+1` small-sample correction is a loose OLS analogy for a ridge/DML estimator; inconsequential at N=192,729 but undertheorized.
- `coca_montecarlo.py:92` — `cond_X` can be `inf`; `json.dumps` emits non-strict `Infinity` tokens.
- `baur_pooled_pca.py:214-224` — the RE `oriented_flip` can leave per-city JSON and the pooled CSV showing opposite signs for one city with no cross-reference; documented but reads as a bug.
- `shen_2021.py:173` — dense n×n cosine matrix (~1.9GB for NYC) built though only K neighbors used; fine per-city, dangerous at scale.

### Verified correct
CoCA symbolic identities (all three, no tolerance games); two-way cluster CGM formula, per-term scaling, negative-variance fallback, and `t_{min(G)-1}` dof; CoCA Monte-Carlo DGP mechanics (C truly orthogonal to X; degeneracy corrupts a column only after T/Y drawn); clean/degeneracy false-positive 0% and degeneracy detection 100% as claimed; Shen production runs confirmed `--doc2vec`; `attach_crime_to_parcels.py` haversine and NaN-coordinate handling.

## Audit 2: Cross-Language / Clean-Room Replication

Two independent from-scratch implementations in `data/scripts/replication/`; no author estimator imported for the core estimate.

**Item 1 — Pooled-PCA Baur DML (boston/sf/philadelphia/chicago).** `referee2_replicate_baur_pooled_pca.py` — independent cKDTree confounder join + 5-fold RidgeCV cross-fit + IF SE.

| city | θ author | θ referee2 | se author | se referee2 | sign/order |
|---|---|---|---|---|---|
| boston | −0.0733 | −0.0909 | 0.0220 | 0.0225 | match |
| sf | −0.1056 | −0.0829 | 0.0253 | 0.0252 | match (largest gap) |
| philadelphia | −0.3740 | −0.3681 | 0.0092 | 0.0092 | tight |
| chicago | −0.3579 | −0.3500 | 0.0185 | 0.0186 | tight |

All four match in sign, order of magnitude, and zero-exclusion. The 10–30% point gaps are attributable to confounder-set curation, not a coding error: the naive kitchen-sink join pulled ~40 junk assessor ID/tax-roll numeric columns into SF's control set, and SF/Boston's gaps trace to the author's temporal-mismatch crime-drop guard — which the replication thereby *validates* as doing real work. Independently rediscovered the Chicago Δ1 row drift (see 1.1).

**Item 2 — Metro×quarter two-way cluster SE.** `referee2_replicate_twoway_cluster.py` — reused only the author's OOF-residual assembly (permitted), wrote the CGM variance from scratch.

| quantity | author | referee2 |
|---|---|---|
| θ | −0.061212 | −0.061212 |
| se_if | 0.000775 | 0.000775 |
| se_metro | 0.016243 | 0.016243 |
| se_twoway | 0.015731 | 0.015731 |
| dof | 9 | 9 |

Bit-for-bit. Confirmed: dof = min(10,13)−1 = 9; negative-variance fallback not a near-miss (pre-fallback V=2.47e-4); the `1/N²` normalization is the correct `Var(mean)=Var(sum)/N²` factor, unchanged by the clustering step. The G_metro=10 caveat (below the ~20–50 CGM recommend) holds up under recomputation and is already disclosed in the script docstring; it should be read as a robustness column, not primary inference.

## Audit 3: Directory & Replication Package

**Readiness: 6/10.** More mature than most packages (folder separation, pinned conda env, Makefile orchestrator, HF-style data card), docked for:
1. Absolute paths hardcoding `/Users/jacobcrainic/causal-real-estate` in four files, including `make_fig_coords.py` — a paper-feeding script that breaks on any fresh clone (automatic-failure item per protocol).
2. Two conflicting dependency manifests, both incomplete. Root `requirements.txt` (unpinned) omits `gensim`, `statsmodels`, `sympy`, `concept-erasure`, `aiohttp` — all genuinely imported and all pip-installed mid-analysis this session. `environment_pin.yml` pins some but omits `concept-erasure` and `aiohttp`. Neither is a complete single source of truth; a reviewer following the README hits `ModuleNotFoundError` on the first LEACE or Shen script.
3. No dedicated replication README distinct from the paper abstract page; `data/scripts/` (~150 scripts) has no README.
4. `Makefile.jbes` is a real 275-line orchestrator but stops at "finalize" — it does not run meta-pooling, cross-method concordance, sensitivity, or counterfactual scripts, and regenerates no LaTeX. "One command reproduces the paper" is not true; it reproduces the 12-city DML estimates only.
5. Confidential-source pattern (Redfin ToS) is documented honestly, with released derived parquets for four sale-price markets, but there is no small public toy dataset to smoke-test the pipeline end-to-end.
*Strength:* seeds are set at ~495 call sites across 72 files; no unseeded stochastic procedure found (though see 1.5 — a set seed does not guarantee Doc2Vec reproducibility under `workers>1`).

## Audit 4: Output Automation

**Tables — mostly MANUAL (major).** Exactly one table is code-generated end to end (`assemble_soldprice_table.py` → `tab_soldprice_v1.tex`, `\input`-ed). Every other table — coverage, verification-claims map, pooled Baur, cross-method, meta — is a hand-typed `\begin{tabular}` with literal numeric cells. The sim-coverage cells trace correctly to `results/simulation/coverage_table.csv`, but nothing connects the CSV to the LaTeX, so a rerun that shifts 0.7367 leaves the printed 0.74 silently stale.

**Figures — MIXED.** Two figures fully automated (`make_zip_plot.py`, `gen_soldprice_fig.py`, both `\input`-ed). The rest go through `make_fig_coords.py`, which *prints* pgfplots coordinate blocks to stdout for a human to paste — no diff-check that the pasted coordinates match the script's current output.

**In-text statistics — MANUAL and duplicated (major).** The pooled +0.125 (traces to `meta_regression_pooled_rebuild.json`) is typed as a literal **nine times across four files**; Pearson +0.91 three times; the 0.74/0.86 coverage figures are duplicated between the table and the adjacent prose. A single rerun of the meta-analysis, coverage sim, or concordance would require manually editing 9, 6, and 3 locations respectively with no cross-check. The soldprice table/figure prove the authors know the `\input`-a-generated-fragment pattern; it simply was not applied to the load-bearing numbers.

## Audit 5: Econometrics

**Identification.** The design rests on Assumption 1 (unconfoundedness given controls), correctly flagged as associational. The deeper tension is internal: the robustness-value table benchmarks an unobserved confounder against the strongest *observed* covariate, and Proposition 1 (CoCA) proves that benchmark is uninformative when a confounder is orthogonal to the controls — which the paper then shows property type is. The paper therefore leans on a sensitivity check it has itself proven blind to the composition-type confounder. Either the RV table is demoted, or CoCA is presented as its replacement rather than a companion.

**Inference on the headline markets.** The panel reports influence-function intervals from the in-sample pooled PC — the row of `tab:sim_coverage` that covers at 0.74–0.79 at the effect sizes Chicago and Philadelphia occupy. The cross-fit-PC correction exists and restores coverage, but is not used for the headline; the bootstrap reassurance is demonstrated only on Boston/NYC/SF (the near-zero markets), while the two large-effect markets, where undercoverage is worst, get no such check because their inputs were reconstructed. This is the most serious econometric gap: the inference for the two most important estimates rests on intervals the paper's own simulation flags as anticonservative, unchecked.

**Simultaneity.** The asking-price headline is exposed to reverse causality — the agent writes the description and sets the asking price jointly, so a positive θ is consistent with flowery prose attached to homes already priced high. The sale-price panel mitigates this but is not the headline. This deserves to be the framing caveat.

**Pooling under I²≈97%.** The +0.125 is a near-unweighted mean over −0.009 to +0.374; with that heterogeneity there is no common effect to pool, yet the abstract leads with the single number. Present it as a location summary of a dispersed distribution, or drop it from the headline.

**Concordance leverage.** "Two encodings of one signal, Pearson 0.91" coexists with a Spearman of only 0.64, and Pearson is leverage-driven. Chicago and Philadelphia sit far out on both axes; a leave-two-out concordance should be reported, because if it collapses without them the "robust across encodings" claim is a two-outlier artifact — now more pressing since the Shen channel pools to +0.060, a hair above zero.

**NYC exception.** NYC covering zero is recast as confirmation ("its text is the most geographic"); with one exception that is unfalsifiable, and Seattle also covers zero without an obvious "most geographic" rationale. Either the mechanism predicts Seattle too, or the exception framing is post hoc.

**Magnitude unit.** "More than three times a bedroom" compares a real discrete attribute to one SD of PC1 of a sentence embedding, a construct nobody can manipulate. The comparison flatters an effect whose unit has no economic interpretation.

## Major Concerns (must address before acceptance)

1. **Regenerate the pooled-PCA treatment on a stable key** (1.1). Rebuild `pooled_pca_treatment.csv` from current parquets or key the merge on `url`/`source_html_sha256`, replace the length-only assert with a key-equality check, and re-run the flagship Baur table. Report that the point estimates are unchanged (Audit 2 indicates they will be) so the record shows the fix is cosmetic to the numbers but closes the integrity hole.
2. **Fix the leaky demean block** (1.2–1.3). Run the `raw`/`dummy`/`demean` comparisons on the same non-leaking imputed block used for the AUC diagnostic, and re-establish whether the composition-confound narrative survives. As written, 9/12 cities show demeaning *increasing* |θ|, so the current module does not support the composition story.
3. **Headline inference** (Audit 5). Either promote the cross-fit-PC intervals to the headline, or extend the bootstrap check to the large-effect markets (Chicago, Philadelphia), or state prominently that the two headline markets' intervals are the ones the coverage simulation flags as undercovering.
4. **CoCA Step-2 reporting** (1.4). Break out detection by confounding strength; do not quote the 51.2% pooled figure.

## Minor Concerns

- Shen Doc2Vec reproducibility: `workers=1` or a ~1-sig-fig caveat (1.5).
- soldprice silent imputation of a 22.3% ACS-suppression pattern with no diagnostic (1.6).
- Automation debt: at minimum wire the pooled effect, Pearson, and coverage numbers to `\input`-ed generated fragments so the nine/three/duplicated hardcodes cannot drift (Audit 4).
- Package: relative paths, one complete dependency manifest, a real replication README, a toy dataset (Audit 3).
- Concordance leave-two-out; pooled-under-heterogeneity framing; NYC/Seattle exception logic; bedroom-magnitude unit (Audit 5).

## Questions for Authors

1. Can you confirm, on a content-keyed rebuild, that no matched listing in any city received a treatment score from a different listing (not merely that the tail was dropped)?
2. After de-leaking the demean block, does the property-type composition confound survive in any market beyond Chicago/Philadelphia?
3. Why report in-sample IF intervals as the headline when the cross-fit-PC correction is built and shown to be needed?
4. Does the "most geographic text → covers zero" mechanism predict Seattle, which also covers zero?

## Verdict

- [ ] Accept
- [ ] Minor Revisions
- [x] **Major Revisions**
- [ ] Reject

**Justification:** The core estimation is correct where independently testable (two-way SE bit-for-bit; Baur DML replicates directionally; CoCA proven symbolically), and the limitations section is unusually candid. But two defects reach the claims: the flagship treatment vector is positionally joined across rewritten parquets with no validation guard (provably drifted in 4/12 cities, unverifiable in the rest), and the composition module's headline comparison runs on a control block its own docstring identifies as a property-type leak — with the empirical consequence that demeaning fails to attenuate in 9/12 cities. Neither is fatal and both are fixable within a revision, but the composition narrative and the exact headline decimals are not established until they are fixed. The automation debt (headline numbers hardcoded nine times) and the headline-inference choice (in-sample intervals the paper's own sim condemns) compound the case for a substantive revision rather than a minor one.

## Recommendations (prioritized)

1. Content-key the pooled-PCA merge and re-run the Baur table; add a key-equality assertion.
2. De-leak the demean block and re-adjudicate the composition finding; if it survives only in Chicago/Philadelphia, say so explicitly and soften the general claim.
3. Resolve the headline-inference question (cross-fit intervals or bootstrap the large-effect markets).
4. Break CoCA Step-2 detection out by strength.
5. Wire the pooled effect, Pearson, and coverage numbers to generated `\input` fragments; extend `Makefile.jbes` to regenerate tables/figures.
6. Ship one complete dependency manifest, relative paths, a replication README, and a toy dataset.
7. Add the concordance leave-two-out and reframe the pooled-under-heterogeneity and NYC/Seattle passages.
