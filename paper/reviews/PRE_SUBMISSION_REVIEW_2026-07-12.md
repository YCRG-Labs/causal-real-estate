# Pre-Submission Referee Report

**Paper**: Does Listing Language Add Value Beyond Location? A Design-Based Audit of Text Embeddings in Automated Valuation
**Authors**: (deanonymized author block per repo)
**Date**: 2026-07-12
**Review Standard**: Leading Field Journal (JBES not in the persona list; reviewed at top-field standard; natural targets Real Estate Economics, Journal of Urban Economics, JBES, JAE)

---

## Overall Assessment

The paper puts a widely-cited but causally untested regularity — text embeddings improve automated home valuation — to a design-based test, holding a scalar text direction fixed while moving a flexible location basis on and off the controls in a partially-linear DML estimator across twelve metros, and concludes the text effect is not, in most markets, geography laundered through language. Execution is careful and unusually candid, but the contribution is contested (Significant per the advocate, Incremental per the skeptic), and the single most critical problem is a cluster of internal numeric inconsistencies in which the paper's two most load-bearing exhibits — the decomposition figure and its own backing table — disagree across most markets, alongside a ~50% overstatement of the sale-price panel size and per-market sample counts that exceed the corpus total.

**Contribution Rating**: Significant (advocate) / Incremental (skeptic). Crux of disagreement: whether the geography-deconfounding delta clears a field-journal bar once one concedes, as the paper's own text does, that most of the attenuation versus the Shen-Ross benchmark comes from generic confounder-set enrichment rather than the geography-specific test, and that the diagnostic apparatus (LEACE, Gelbach, Cinelli-Hazlett, CoCA) is assembled rather than invented.

**Preliminary Recommendation**: Revise before sending to referees. The paper is too well-executed to desk-reject, but has a first-order, fixable identification gap (an unaddressed condition/quality confound, pointed to by the paper's own vocabulary evidence), an undelivered robustness check it explicitly promises (MiniLM), a headline that uses the very estimator its own simulation shows is miscalibrated in the two largest-effect markets, and a set of internal numeric contradictions that a referee would catch immediately.

*Novelty relative to literature not cited in the paper has not been verified.*

---

## 1. Central Contribution

### Advocate's Case (Significant)
Converts an established predictive fact (text lifts AVM accuracy; Baur et al. 2023) into a design-based measurement with a clean, largely negative answer to the spatial-confounding worry, replicated across twelve markets and reconfirmed on realized sale prices in ten — a scope no single paper in its own bibliography attempts. Types: primarily new question, plus a new-method combination (Gelbach decomposition + LEACE/IGBP erasure + Cinelli-Hazlett sensitivity on an embedding-valued treatment under continuous spatial confounding) and a new answer. Load-bearing result: the decomposition (effect near-invariant to folding in location even though the treatment demonstrably carries location, R² up to 0.42 in NYC), which is non-vacuous precisely because the treatment's own spatial R² is nontrivial almost everywhere. NYC — the one market the adjustment bites — is also the market whose text is dominated by borough/neighborhood names, giving the result falsifiable mechanism texture. Capped below Transformative by the self-admitted associational identification and the patched coverage weakness.

### Skeptic's Case (Incremental)
The paper's own text concedes the delta is application, not method ("What this paper adds is... neither the erasure operator nor the recognition that a text feature can conflate treatment with confounder... but the application of that apparatus to an economic estimand"), and explicitly disclaims CoCA novelty ("neither a new estimator nor a new limit theorem, only the connection of two existing literatures"). The frozen-probe proposition is folklore (part a) plus an algebraic identity and a toy-XOR existence proof in the style of already-cited work. Most of the Atlanta attenuation vs. Shen's +0.140 (to +0.033) comes from generic confounder enrichment, achieved before the geography test is run; the decomposition then finds adding a flexible spatial basis on top of an already-rich, already-geography-including control set changes little — close to what DML orthogonality predicts by construction when W is rich. The "one signal, two encodings" claim is leverage-driven (Pearson 0.91 vs Spearman 0.64, Lin 0.79; driven by Chicago/Philadelphia), and the "audit for regulators" framing outruns a sensitivity analysis in which only 2/12 markets clear RV=0.10.

### Synthesis
Both reviewers agree the execution and twelve-market-plus-sale-price scope are genuine assets and that the central empirical claim (spatial adjustment barely moves the effect in 11/12 markets) is real. They diverge on whether that claim is surprising or contribution-grade: the skeptic reads it as narrower than advertised once generic-confounder enrichment is separated from the geography-specific test, and notes the paper's own repeated admissions that the method is assembled. The crux the authors must win in the introduction is the delta over Shen-Ross/Baur net of generic confounding control. The single change that would most strengthen the contribution: report the estimate at each successive control-enrichment stage (structured → +context → +raw location → +flexible spatial basis) so the geography-specific increment is visible rather than conflated. Novelty relative to literature not cited in the paper has not been verified.

---

## 2. Referee Assessment (Identification, Analyses, Positioning & Fit)

**Identification.** Honest associational DML; Assumption 1 stated but not defended on priors. The decisive gap: the entire deconfounding architecture targets one confound — spatial confounding through language — and treats it as coextensive with "the" confounding problem. There is no measure of interior condition, renovation, staging, or finish quality in the control set, yet the paper's own loadings table shows the price-loading end of the text axis is dominated by condition/finish language ("spacious, modern, designed, chef, owner suite") outside NYC. "Unobserved quality laundered through words" is a third explanation the paper never tests and which produces exactly the reported pattern (survives location adjustment, heterogeneous across markets, inversely related to how generic a market's stock is). Calling the survivor "semantic content vs. geography" is a false dichotomy. Second, asking-price simultaneity is named and handed to the sale-price panel, but that panel's treatment is still the same agent's listing text and sale price anchors on list price (Genesove-Mayer, uncited), so it may inherit rather than break the simultaneity. Third, the generated-regressor undercoverage is documented (0.74–0.81 at Chicago/Philadelphia effect sizes) but the headline uses the in-sample PC, and the bootstrap check covers only Boston/NYC/SF.

**Required analyses (blockers):**
1. [CRITICAL] A property-condition/quality control or proxy (renovation records, image-derived condition score, price-per-sqft deviation from comparable-sale medians, or an as-is/fixer flag), shown alongside the location decomposition.
2. [CRITICAL] Apply the out-of-fold/cross-fit PC (already built and validated) to the headline table, or extend the full-pipeline bootstrap to Chicago and Philadelphia.
3. [CRITICAL] Deliver the MiniLM cross-encoder recomputation promised in Section 4, or withdraw the claim.
4. [CRITICAL] A direct test that the sale-price panel breaks asking-price simultaneity (control for list price / list-to-sale ratio, or restrict to price-reduced listings).
5. [CRITICAL] Run CoCA as a stated diagnostic across all twelve markets and report the full table, rather than only the vacant-land cases in Boston/Philadelphia/Chicago.

**Suggested analyses:**
1. [MAJOR] Placebo: does one market's text direction "predict" price in an unrelated market once controls are applied?
2. [MAJOR] Extend the Cinelli-Hazlett benchmarking to a quality/condition confounder axis, not only geography.
3. [MAJOR] Cite and test list-price anchoring in the sale-price section (correlate the text direction with list-to-sale ratio).
4. [MAJOR] Report per-market MiniLM results as a full table alongside mpnet.
5. [MAJOR] Check whether the surviving text effect correlates with block-group racial composition net of controls (ACS race vars already in the set), given the fairness framing.

**Positioning.** Core text-valuation cites (Shen 2021, Baur 2023, Nowak, Lorenz) and the spatial-confounding / DML-sensitivity apparatus are the right ones and engaged directly. Gaps: list-price anchoring literature (Genesove-Mayer) absent despite being central to the paper's own simultaneity concern; the housing-quality/unobserved-condition hedonic literature absent, precisely the Part-1 gap; sociology-of-listing-language (racial/neighborhood coding) missing despite the fairness motivation.

**Fit & recommendation.** Best fits: Real Estate Economics, Journal of Urban Economics (housing substance); JBES/JAE if reframed as methods-forward. Recommendation: Revise before sending to referees.

**Questions to authors:** (1) How do you rule out unobserved condition/quality rather than "content vs. geography"? (2) Why is the sale-price panel not just inheriting list-price anchoring? (3) Why report in-sample-PC intervals as headline when your own sim shows they undercover worst at Chicago/Philadelphia and your bootstrap skips both? (4) Where are the MiniLM results Section 4 promises? (5) Was CoCA run across all twelve markets or only where a problem was suspected? (6) How do you distinguish "NYC is where spatial confounding bites" from "NYC is where omitted-quality confounding is worst"? (7) In what sense is a sign-ambiguous, data-adaptive, market-varying PC a single causal object commensurable enough to pool?

---

## 3. Unsupported Claims & Identification Integrity

- [CRITICAL] conclusion ¶1: "a direct stylistic channel and a location-inference channel of comparable size and opposite sign" misstates the paper's own numbers (style +0.074 [0.034,0.113]; submarket-swap +0.002 [-0.007,0.010]) — not opposite-sign, not comparable. Use the appendix's own phrasing ("averages to approximately nothing because the markets divide in sign").
- [CRITICAL] Generated-regressor caveat: the text concludes the bootstrap "standing behind" the IF intervals covers all twelve markets when it covers only Boston/NYC/SF and excludes the highest-risk pair (Chicago/Philadelphia). State this explicitly and report bootstrap intervals for those two.
- [MAJOR] Headline "priced for its content, not its geography" stated as flat fact one sentence after "associational." Reword to "the association ... is not reducible to geography," consistently.
- [MAJOR] Magnitude comparisons ("several times a bedroom"; "several times the racial sale-price gap") borrow causal weight for an object the paper says is not causally identified; mark as scale-illustrative only.
- [MAJOR] NYC mechanism "caught in the act" is an n=1 inference; reframe to "consistent with." Same for the two-market (NYC/Dallas) vocabulary rule.
- [MAJOR] "Effect" is the default noun for θ̂ in nearly every table caption and in prose despite the associational framing; standardize captions to "coefficient/association."
- [MAJOR] "One signal, two encodings" overstates a Pearson 0.91 that a Spearman 0.64 / Lin 0.79 undercut; soften to "consistent with a single signal" and report the weaker rank statistics.
- [MAJOR] "Reassured for most of the country" generalizes from twelve non-representative metros the Limitations paragraph itself flags; scope it.
- [MAJOR] Measurement error in controls (ACS margins, cross-city crime-schema heterogeneity) never discussed though treatment measurement error is treated at length.
- [MAJOR] Priority claim "the question has not been asked. This paper asks it." needs "to our knowledge" and restriction to the specific combination.
- [MINOR] "worth/value/premium/survives/holds" causal-flavored verbs on an associational estimate; retrodesign "true effect" → "benchmark effect."

---

## 4. Internal Consistency & Cross-Reference Verification

- [CRITICAL] Sale-price panel size: abstract "nearly three hundred thousand" / body "roughly two hundred and ninety thousand" vs. tab:soldprice rows summing to **192,729** (~50% overstatement; also "two orders of magnitude larger than the description samples" is wrong — it is 3–4×).
- [CRITICAL] tab:baur-12city NY n=15,255 exceeds corpus NY 15,227; table sums to 69,204 not 69,173; Chicago/Atlanta/Phoenix each +1 over corpus. (Paper-side symptom of the positional treatment-merge drift found in the code audit.)
- [CRITICAL] tab:shen-12city SF n=981 but prose says 986.
- [CRITICAL] Decomposition NYC: prose/appendix −0.119→−0.152 (Δ0.033) vs figure data −0.122→−0.154 and caption Δ0.032.
- [CRITICAL] Decomposition TPRS shift: main text "at most 0.059" (NYC) vs appendix "at most 0.030"; factor ~2.
- [CRITICAL] CoCA Boston θ 0.40/0.45/0.34 is 5–20× every other Boston estimate (Shen +0.020, Baur +0.073) with no stated quantity/specification.
- [MAJOR] Robustness-value scope: sensitivity is computed only for the Shen channel, but abstract/conclusion describe the bound as covering "each market's result" generally; the headline Baur channel is never run through it.
- [MINOR] "Sun Belt" mislabels Seattle/Denver/Portland; I²=98.6% vs table "98%"; truncation "ratio undefined" vs "all ten markets".

---

## 5. Mathematics, Equations & Notation

- [CRITICAL] fig:decomposition vs tab:decomposition-12 diverge for most markets (Atlanta 0.124/0.129 vs 0.253/0.250; Washington 0.129/0.120 vs 0.015/0.013; Denver 0.067/0.082 vs 0.154/0.158; Philadelphia 0.367/0.363 vs 0.408/0.405) though the appendix says it gives "the numbers behind" the figure. Regenerate both from one source.
- [CRITICAL] tab:baur-12city n's exceed corpus totals (impossible for a subsample). Reconcile.
- [MAJOR] Prop. CoCA proof uses Yule's single-scalar partial-correlation recursion, but X is the high-dimensional control vector; the one-step identity does not hold as written. Restate via residuals from multivariate regression on W, or restrict to the single-control case and mark the generalization heuristic.
- [MAJOR] Glyph overloading: C means observed context c, the frozen-probe target, and the CoCA omitted confounder (three referents); W is defined as (ℓ,x,c) in Assumption 1 but redefined as location-free in the decomposition appendix; uppercase/lowercase L,X,C,T,Y mixed between Section 3 and Appendix A. Rename to disambiguate.
- [MAJOR] Decomposition θ_base removes location and is not the headline W-conditioned spec (Chicago headline +0.358 vs base +0.461), never stated.
- [MAJOR] CoCA Boston 0.40/0.45/0.34 channel/spec unstated (echoes §4).
- [MINOR] Truncation Denver λ=0.7857 rounds to 0.79 not 0.78; hat-value 0.71 not in tab:meta-moderators (that column is Cook's distance); q (HKSJ multiplier) undefined; frozen-probe log base (bits vs nats) unstated; `$$`/`\big`/`\text{}`-as-operator LaTeX nits.

---

## 6. Tables, Figures & Documentation

- [CRITICAL] fig:decomposition panel (a) plots point markers with no CIs though its caption invokes "the estimate's own standard error." Add 95% whiskers.
- [CRITICAL] tab:soldprice n sums to 192,729 vs "≈290,000" prose (echoes §4).
- [MAJOR] fig:sim-coverage annotations ("cov. 78%", "89%") disagree with tab:sim_coverage (0.83, 0.88) for the same cells.
- [MAJOR] shen/baur tables: caption/notes never name the control set the DML is net of.
- [MAJOR] tab:cross-method-12 has no n column and switches to parenthetical SE; swap column carries no uncertainty. tab:soldprice listing/corrected arms, tab:counterfactual-12 swap, and tab:decomposition tables lack CIs.
- [MAJOR] Uncited exhibits: tab:corpus-12, tab:nyc-vocab, fig:decomposition-tprs, tab:verification_map never \ref'd.
- [MINOR] soldprice "--" entries unexplained in notes; truncation all-negative sign convention diverges from the oriented-positive convention; decimal-precision drift; convention sentence sometimes in prose, sometimes in caption.

---

## 7. Spelling, Grammar & Style

- [CRITICAL] British/American spelling mix: neighbourhood, relabelling/relabelled, formalised/formalises, colour, labelled — all minority usages against American majority; standardize.
- [CRITICAL] IGBP (used 3×) and GRL never defined at first use. Spell out.
- [MAJOR] Very long multiply-subordinated sentences in Sections 1–2 (~90 words before first period); split over ~60 words.
- [MAJOR] Dangling modifiers: "Applied across twelve metropolitan areas ... the answer runs against ..."; "After restricting to residential single-unit dwellings ... the panel holds ...". Recast with explicit "we".
- [MINOR] Passive→active voice drift in the data-collection appendix; "It is worth separating ..." metadiscourse; numeral-vs-word inconsistency (11 vs eleven d.f.; 10–15% vs ten-to-fifteen percent); "i.e." missing comma; Shen sole-author "her" vs "Shen-Ross".

---

## Priority Action Items

**CRITICAL** (could cause desk rejection or major referee objections):
1. [Contribution] Separate the geography-specific increment from generic confounder enrichment (report the estimate at each control-enrichment stage), and reframe "semantic content vs. geography" — the paper's own loadings show the survivor is condition/quality language, and no condition control exists. Add a quality proxy or narrow the claim to "text adds information beyond beds/baths/location, of unknown provenance."
2. Reconcile the numbers that don't agree: fig:decomposition vs tab:decomposition-12 across most markets; the sale-price panel size (192,729, not ≈290,000); tab:baur-12city n's exceeding the corpus total; SF n 986 vs 981; NYC decomposition values across prose/figure/caption/appendix; the TPRS 0.059-vs-0.030 shift; the CoCA Boston 0.40/0.45/0.34 scale.
3. Fix the headline inference: apply the cross-fit PC or extend the bootstrap to Chicago and Philadelphia, and stop describing the three-market bootstrap as if it covers all twelve.
4. Deliver the MiniLM cross-encoder recomputation Section 4 promises, or delete the sentence.
5. Correct the conclusion's counterfactual claim ("comparable size and opposite sign") to match the appendix (+0.074 vs +0.002, cancellation across markets).
6. Add 95% CIs to fig:decomposition panel (a) and to the tables missing them (cross-method swap, soldprice listing/corrected, counterfactual swap, decomposition).
7. Test whether the sale-price panel actually breaks asking-price simultaneity (control for list price / list-to-sale ratio).
8. Run and report CoCA across all twelve markets.
9. Copy-edit CRITICALs: standardize American spelling; define IGBP and GRL.

**MAJOR** (referees will likely raise):
10. Restate the CoCA proposition for a vector control set (or restrict to the scalar case and flag the generalization heuristic); position CoCA as replacement for, not companion to, the observed-covariate RV benchmark it disarms.
11. Soften "one signal, two encodings" to "consistent with" and report Spearman 0.64 / Lin 0.79 alongside Pearson 0.91; add the concordance leave-two-out (drop Chicago/Philadelphia).
12. Scope the RV/sensitivity claim to the Shen channel it actually covers, or run the Baur channel through it; state that only 2/12 markets clear RV=0.10 before advising regulators.
13. Standardize "effect"→"coefficient/association" in captions and prose; scope the "most of the country" generalization; add "to our knowledge" to the priority claim.
14. Fix glyph overloading (C, W, case convention); state what each decomposition arm adds/removes vs the headline W.
15. Cite and engage list-price anchoring (Genesove-Mayer) and the unobserved-quality hedonic literature; add the racial-composition check.
16. Name the control set in the shen/baur table notes; add n columns and \ref the uncited exhibits.
17. Reconcile the fig:sim-coverage annotations with tab:sim_coverage; split the long sentences; fix the dangling modifiers.

**MINOR** (polish):
18. Truncation λ rounding (0.79); q, log-base, hat-value definitions; Sun Belt mislabel; I² precision; LaTeX bracket/operator nits; numeral-vs-word and CI-bracket-vs-parenthesis conventions; passive-voice drift.
