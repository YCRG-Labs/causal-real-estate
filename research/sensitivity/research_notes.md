# Sensitivity Bounds & E-values — Research Dossier (Item 7)

For Item 7: enrich the current Cinelli-Hazlett RV (= 0.654) with E-values, OVB-DML, Manski bounds, and Bayesian sensitivity.

---

## B.1 E-values: VanderWeele-Ding (2017) and Continuous Extensions

### Original
VanderWeele & Ding, "Sensitivity Analysis in Observational Research: Introducing the E-Value," *Annals of Internal Medicine* 167(4), 268-274 (2017).
- https://www.acpjournals.org/doi/10.7326/M17-1485
- Preprint PDF: https://hrr.w.uib.no/files/2019/01/VanderWeeleDing_2017_e_-value.pdf

### Formula
For RR > 1 (invert for RR < 1):
```
E-value(point) = RR + sqrt(RR * (RR - 1))
E-value(CI)    = same formula applied to CI bound nearest the null
```
Interpretation: unmeasured confounder would need *both* RR_UD and RR_UY at least as large as E-value to fully explain away observed effect.

### Continuous exposures
`evalues.OLS` in R `EValue` dichotomizes OLS coefficient via chosen contrast δ:
```
RR_approx = exp(0.91 * β * sd_Y * δ / sd_Y)
```
Package wraps it.

References:
- Linden, Mathur, VanderWeele (2020), "Conducting sensitivity analysis... using E-values: The evalue package," *Stata Journal* 20(1), 162-175. https://journals.sagepub.com/doi/full/10.1177/1536867X20909696
- R package: https://cran.r-project.org/web/packages/EValue/EValue.pdf

### Python
**No first-class Python package as of 2025.** Either (i) call R via `rpy2`, or (ii) compute formula by hand — 5-line function. No published Python equivalent peer-validated.

### Extension to ML/DML estimands
Mathur & VanderWeele (2020) extended to standardized mean differences and meta-analysis. For DML: convert (θ̂, SE) to approximate RR via standard Chinn / VanderWeele transformation, apply formula. **Approximate** — more principled DML-native approach is Cinelli-Chernozhukov below.

### Recommendation
Report E-values as *minimum-strength-of-confounding* (one for point estimate, one for CI bound), use approximate continuous-exposure version. Note explicitly that for DML the principled approach is OVB-DML. **E-values are widely understood by JBES referees** — easiest sensitivity number to communicate even if not most rigorous.

---

## B.2 Cinelli-Hazlett Extensions to DML — Chernozhukov et al. 2022

### Paper
Chernozhukov, Cinelli, Newey, Sharma, Syrgkanis (2021/2024), "Long Story Short: Omitted Variable Bias in Causal Machine Learning," arXiv:2112.13398. https://arxiv.org/abs/2112.13398. Last revised May 2024.

### What it does
Generalizes Cinelli-Hazlett (2020) OVB-with-partial-R² framework from OLS to wide class of causal-ML estimands defined as continuous linear functionals of a Riesz representer (ATE, ATT, average derivatives, policy effects, partially-linear models, partially-linear IV, etc.).

Bias of long-vs-short estimator tightly bounded by:
```
|Bias|² ≤ S² · η_Y² · η_D² / (1 - η_D²)
```
where η_Y² and η_D² are partial R² of omitted confounder in *outcome regression* and *Riesz representer*; S is known data-dependent scaling constant. Same logic as OLS RV machinery, ML-valid identification.

### Software
- R: **`dml.sensemakr`** (https://github.com/carloscinelli/dml.sensemakr) implements this exactly. Same RV / RVa robustness values as `sensemakr` but for DML-fitted model.
- Python: **No published port yet.** Either (i) port the (relatively short) bias formula, or (ii) call R via `rpy2`. `PySensemakr` (https://github.com/nlapier2/PySensemakr) only covers OLS.

### Recommendation
**This is the correct sensitivity analysis for our DML estimate.** Replace linearized RV=0.654 with honest DML RV via `dml.sensemakr`. Report:
- **RV** (point estimate to zero)
- **RVa-0.05** (95% CI to include zero)

Benchmark against an observed covariate (e.g., strongest single non-text confounder) — confounder "k=1" or "k=2" times as strong as observed is natural plausibility yardstick.

---

## B.3 Manski Bounds and Cornfield-Style Inequalities

### Manski (1990)
"Nonparametric Bounds on Treatment Effects," *AER P&P* 80(2), 319-323. Worst-case bounds on E[Y(1) - Y(0)] given only support of Y, no assumptions about confounding. For binary Y: width 2; bounded continuous Y: width 2·(y_max - y_min). **Typically very wide; not super informative alone but anchors the discussion.**

### Tighter assumptions
- **Monotone Treatment Selection (MTS) / Response (MTR)** — Manski-Pepper 2000: tightens substantially under sign restrictions
- **Cornfield-style inequalities** — minimum RR_UD and RR_UY pair to nullify effect; conceptual root of E-value

### Modern partial-ID Python
- `autobounds` (Duarte, Finkelstein, Knox, Mummolo, Tchetgen Tchetgen) — LP bounds. https://github.com/gjduarte/autobound
- `causaloptim` (R) — wider class of partial-ID problems
- For our application: write *bounded confounder* version — assume unmeasured U has bounded influence (e.g., propensity shift in [-c, c]); solve for implied bound on θ. What Rosenbaum (2002) does for matching; for DML, use Chernozhukov et al. 2022 formula.

### Recommendation
Full Manski worst-case bounds too wide to be informative on text-PC effects. **Report "plausibility-bounded Manski bounds":** assume omitted confounder explains at most r% of residual variance ("no more than 30% as strong as strongest observed covariate"); report implied bound on θ. Operationally identical to `dml.sensemakr` RV exercise but framed as partial-identification interval — pick whichever framing referees prefer.

---

## B.4 Bayesian Sensitivity Priors

### McCandless-Gustafson (2017)
"A comparison of Bayesian and Monte Carlo sensitivity analysis for unmeasured confounding," *Statistics in Medicine* 36(18), 2887-2901. https://onlinelibrary.wiley.com/doi/10.1002/sim.7370

Methodological message: MCSA samples bias parameters from prior and propagates; **BSA actually updates bias-parameter prior on data via Bayes' rule.** Recommend BSA over MCSA — MCSA can give misleadingly wide intervals that ignore data's information about bias parameters.

### Cleanest BSA recipe for our setting
1. Parameterize unmeasured confounder by partial R² with Y and T: (η_Y, η_D) ∈ [0,1]²
2. Place prior — e.g., Beta(2, 8) on each, expressing "probably small," peaked at ~0.2
3. For each draw (η_Y, η_D), compute bias-adjusted θ using Chernozhukov et al. 2022 formula
4. Posterior probability P(|θ_adj| > 0.05 | data) is Monte-Carlo integral over prior weighted by likelihood — in simplest treatment, likelihood factor constant across (η_Y, η_D) prior support ("tipping-point" simplification, Greenland), reducing to prior-weighted area of {|θ_adj| > 0.05} in (η_Y, η_D) space

### Greenland's tools
`episensr` (R) implements probabilistic bias analysis. Greenland's 2017 commentary on McCandless-Gustafson lays out cleanest practical workflow. **No Python port; ~40 lines to reimplement.**

### Recommendation
Add one-paragraph BSA: prior on (η_Y, η_D) ~ Beta(2,8) iid (mean ~0.2); report posterior probability that bias-adjusted DML coefficient exceeds 0.05 in absolute value. Complements RV and E-value with a *probability* statement referees find concrete.

---

## B.5 Negative-Control Calibration (Schuemie)

### References
- Schuemie et al. (2014), "Interpreting observational studies: why empirical calibration is needed to correct p-values," *SiM* 33(2), 209-218
- Schuemie et al. (2016), "Robust empirical calibration of p-values using observational data," *SiM* 35(22), 3883-3888
- Schuemie et al. (2018), "Empirical confidence interval calibration...," *PNAS* 115(11), 2571-2577. https://www.pnas.org/doi/10.1073/pnas.1708282114
- R: `EmpiricalCalibration` (CRAN/OHDSI). https://ohdsi.github.io/EmpiricalCalibration/

### Method
Use panel of "negative controls" — covariates believed *a priori* to have zero causal effect — to estimate bias distribution induced by analysis pipeline; calibrate p-values and CIs against this empirical null. **Addresses systematic error across pipeline, not specifically unmeasured confounding.**

### Should we report both?
**Yes** — answer different questions:
- E-value, RV, OVB-DML, Manski, BSA → "how much unobserved confounding to overturn?" (adversarial sensitivity)
- Schuemie → "do other plausibly-null exposures show systematic non-null associations of magnitude we observe?" If yes, pipeline has bias before unobserved confounding. If no, headline effect stands out.

For text-treatment NLP this is *especially* useful — text features notoriously confounded. Picking ~20 plausibly-null linguistic features (orthogonalized PC-2 or shuffled descriptions) as negative controls and showing our effect's tail position in empirical null is a powerful robustness exhibit.

### Recommendation
Include Schuemie-style negative-control calibration. Frame as **"external validity of the null distribution,"** not as sensitivity analysis; mention it complements OVB-DML / RV / E-value triad which target unobserved confounding specifically.

---

## End-to-End Recipe

```python
# === CATE ===
from econml.dml import LinearDML
from sklearn.ensemble import GradientBoostingRegressor as GBR

est = LinearDML(model_y=GBR(random_state=0), model_t=GBR(random_state=0),
                discrete_treatment=False, cv=5, random_state=0)
est.fit(Y, T, X=X_strat, W=W_conf)

# Cell-level effects, IF-based SEs
for stratifier_name, q in [('price', price_q), ('length', length_q)]:
    for k in range(4):
        m = (q == k)
        ate_k = est.effect(X_strat[m]).mean()
        lo, hi = est.effect_interval(X_strat[m], alpha=0.05)
        report(stratifier_name, k, m.sum(), ate_k, lo.mean(), hi.mean())

# BLP heterogeneity test (Chernozhukov et al. 2018)
inf = est.coef__inference()
print(inf.summary_frame())

# Across-seed stability (50 seeds)
ates = [LinearDML(..., random_state=s).fit(Y,T,X=X_strat,W=W_conf).ate(X_strat)
        for s in range(50)]
report_stability(np.mean(ates), np.std(ates))

# === Sensitivity ===

# OVB-DML RV (call R via rpy2 OR re-implement bias formula from CCNSS 2022)
# R: dml.sensemakr::dml(...) %>% sensemakr() -> RV, RVa
# Bench: 1x and 3x strongest observed covariate

# E-value (continuous via OLS approximation)
def evalue_from_estimate(theta, se, sd_Y):
    rr = np.exp(0.91 * theta / sd_Y)
    rrL = np.exp(0.91 * (theta - 1.96*se) / sd_Y)
    rr_ci = max(1.0, rrL) if theta>0 else 1/min(1.0, rrL)
    ev_pt = rr + np.sqrt(rr * (rr-1)) if rr>1 else None
    ev_ci = rr_ci + np.sqrt(rr_ci * (rr_ci-1)) if rr_ci>1 else 1.0
    return ev_pt, ev_ci

# Bayesian sensitivity (40-line MC over partial-R^2 priors)
import scipy.stats as st
B = 20000
eta_Y = st.beta(2,8).rvs(B); eta_D = st.beta(2,8).rvs(B)
S = compute_S_from_data(Y, T, W)            # CCNSS scaling constant
bias = S * np.sqrt(eta_Y**2 * eta_D**2 / (1 - eta_D**2))
theta_adj = theta_hat - np.sign(theta_hat) * bias
print("P(|theta_adj|>0.05):", np.mean(np.abs(theta_adj) > 0.05))

# Manski-style plausibility bounds
print("Plausibility bound (95% prior mass):",
      np.quantile(theta_adj, [0.025, 0.975]))

# Negative-control calibration (Schuemie)
# EmpiricalCalibration::fitMcmcNull on ~20 negative-control DML estimates
# -> calibrated p-value for the focal estimate
```

---

## Citations to Bundle (references.bib)

- Kennedy 2023 EJS — DR-Learner; arXiv:2004.14497
- Kennedy, Ma, McHugh, Small 2017 JRSS-B — continuous-T doubly-robust; arXiv:1507.00747
- Semenova & Chernozhukov 2021 *Econometrics J.* — debiased ML for CATE/structural functions; arXiv:1702.06240
- Chernozhukov, Demirer, Duflo, Fernandez-Val 2018 NBER WP 24678 — generic ML inference, BLP/GATES; arXiv:1712.04802
- Athey & Imbens 2016 PNAS — recursive partitioning / honest causal trees
- VanderWeele & Ding 2017 *Annals Int. Med.* — E-value
- Linden, Mathur, VanderWeele 2020 *Stata J.* — `EValue` software, OLS extension
- Cinelli & Hazlett 2020 *JRSS-B* — sensitivity / RV (current method)
- Chernozhukov, Cinelli, Newey, Sharma, Syrgkanis 2021/2024 — OVB in causal ML; arXiv:2112.13398; `dml.sensemakr` R package
- McCandless & Gustafson 2017 *Stat. Med.* — BSA vs MCSA
- Greenland 2017 *Stat. Med.* (commentary) — BSA workflow
- Manski 1990 *AER P&P* — partial identification
- Schuemie et al. 2014, 2016, 2018 *Stat. Med.* / *PNAS* — empirical calibration; OHDSI `EmpiricalCalibration` R package
