# CATE via DR-Learner — Research Dossier (Item 6)

For Item 6: deliver the CATE-by-quartile analysis we removed, using modern doubly-robust machinery.

---

## A.1 Kennedy (2022/2023) DR-Learner: Canonical Reference

Edward H. Kennedy, "Towards optimal doubly robust estimation of heterogeneous causal effects," arXiv:2004.14497, *Electronic Journal of Statistics* 17(2), 3008-3049 (2023).

URLs:
- arXiv: https://arxiv.org/abs/2004.14497
- EJS: https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-17/issue-2/Towards-optimal-doubly-robust-estimation-of-heterogeneous-causal-effects/10.1214/23-EJS2157.full

### Estimand
CATE τ(x) = E[Y(1) - Y(0) | X = x] for binary T.

### Procedure (binary case)
1. **Cross-fit nuisances**: K folds; on fold-k-out data, estimate propensity π(x) = P(T=1|X=x) and outcome regressions μ_a(x) = E[Y|X=x, T=a]
2. **Construct DR pseudo-outcome** on each held-out fold:
   ```
   φ(Z) = [(T - π(X)) / (π(X)(1-π(X)))] * (Y - μ_T(X)) + μ_1(X) - μ_0(X)
   ```
3. **Regress φ(Z) on X** using any second-stage learner f̂ (kernel, series, Lasso, RF, NN) to get τ̂(x)

### Key theoretical guarantee
Theorem 2: model-free oracle inequality. L2 error of τ̂ at second stage bounded by *oracle* second-stage error (regressing true φ on X) **plus product-of-nuisance-error term**. Quasi-oracle / weakly-oracle property: if propensity and outcome each converge at n^(-1/4), second stage attains oracle rate as if φ were known.

### Standard errors
- IF-based for *averages* of τ̂ over subgroups: immediate. EIF of ATE on subgroup S is `1{S} * φ(Z) / P(S)` → sample-mean of φ within each quartile gives valid asymptotic SEs (σ̂ / √n_S).
- Pointwise SEs at specific x: kernel/series with IF correction, or post-hoc HC SEs from second-stage regression conditional on cross-fit nuisances.
- **Standard practice in applied work: report subgroup-average effects with IF-based SEs, not pointwise CATEs.**

---

## A.2 Continuous Treatment: Recommended Approach

### Foundational continuous-T paper
Kennedy, Ma, McHugh & Small (2017), "Nonparametric methods for doubly robust estimation of continuous treatment effects," *JRSS B* 79(4), 1229-1245. arXiv:1507.00747. https://arxiv.org/abs/1507.00747

Builds DR pseudo-outcome for continuous T; regresses nonparametrically on T (not X) for dose-response E[Y(t)]. Adapting to CATE: regress pseudo-outcome on (T, X) jointly or stratify by X.

### General framework — cite this for JBES
Semenova & Chernozhukov (2021), "Debiased Machine Learning of Conditional Average Treatment Effects and Other Causal Functions," *Econometrics Journal* 24(2), 264-289. arXiv:1702.06240. https://arxiv.org/abs/1702.06240

Setup:
- Express structural function (CATE, dose-response, average derivative) as conditional expectation of Neyman-orthogonal signal depending on first-stage nuisances
- Estimate nuisances by ML with cross-fitting
- Project orthogonal signal onto **finite-dimensional dictionary of basis functions φ(X)** of heterogeneity covariate (Best Linear Predictor — BLP)
- Get √n-consistent, asymptotically normal coefficients with closed-form HC standard errors

**This is the cleanest formal apparatus for "CATE stratified by quartile":** dictionary = indicator basis functions for quartiles; uniform confidence bands across cells.

For continuous T + heterogeneity in X, Semenova-Chernozhukov "average partial effect" (§3.3): treat T linearly (or low-order polynomial), project partial-effect signal onto quartile basis.

### Recommendation
**Cite Kennedy 2022 as foundational DR-Learner. Report Semenova-Chernozhukov BLP estimator for actual CATE-by-quartile numbers.** More defensible asymptotic framework when treatment is continuous and heterogeneity is low-dimensional (4 cells per stratifier).

---

## A.3 EconML Library: Current API (v0.16.0, July 2025)

### Critical fact
`econml.dr.DRLearner` and variants (`LinearDRLearner`, `SparseLinearDRLearner`, `ForestDRLearner`) **are designed for discrete T** — require `model_propensity` to be a classifier. **Don't force continuous T into DRLearner.**

### Production-quality choices for continuous T + heterogeneity
- **`econml.dml.LinearDML`** — partially-linear DML with sklearn first-stage, linear final stage in X. Closed-form HC SEs. Use when dim(X) << n.
- **`econml.dml.SparseLinearDML`** — same but with debiased Lasso final stage; use when dim(X) ≈ n.
- **`econml.dml.CausalForestDML`** — fully non-parametric final stage with bootstrap-of-little-bags CIs; supports continuous T natively. Estimates Cov[Y,T|X] / Var[T|X]-style average partial effect.
- **`treatment_featurizer=PolynomialFeatures(...)`** — standard way to get nonlinear continuous-T effects without abandoning linear-final-stage CIs.

### API skeleton (continuous T, low-dim X)
```python
from econml.dml import LinearDML
est = LinearDML(model_y=..., model_t=..., discrete_treatment=False, cv=5, random_state=0)
est.fit(Y, T, X=X, W=W)
est.effect(X_test)
est.effect_interval(X_test, alpha=0.05)
inf = est.const_marginal_effect_inference(X_test)
```

### Docs
- https://www.pywhy.org/EconML/
- https://www.pywhy.org/EconML/_autosummary/econml.dml.LinearDML.html
- GitHub: https://github.com/py-why/EconML

### Recent versions
- 0.15: added `treatment_featurizer`
- 0.16: scikit-learn 1.4 compat, better `RScorer` and CATE intent-to-treat helpers, deprecates legacy bootstrap modes

None change public API for our use case.

---

## A.4 Heterogeneity Testing — Most Defensible

### Three options, ranked

1. **Chernozhukov-Demirer-Duflo-Fernandez-Val (2018)** — "Generic Machine Learning Inference on Heterogeneous Treatment Effects," NBER WP 24678 / arXiv:1712.04802. https://arxiv.org/abs/1712.04802. Introduces Best Linear Predictor (BLP) test, GATES (Group Average Treatment Effects, sorted by estimated CATE quintile), CLAN (Classification ANalysis). Test statistic: slope of `Y - μ_0(X)` on demeaned ML proxy of CATE; null = no heterogeneity. Requires only randomized/conditionally-randomized experiment; valid even when ML is poor approximation. **Most defensible for JBES** — formal sample-splitting validity, explicit null.

2. **Semenova-Chernozhukov BLP test** (1702.06240). Same idea generalized — test if projection coefficients on quartile dictionary are jointly zero (Wald test on dictionary coefficients). Strong fit if we adopt their estimator.

3. **Athey-Imbens (2016)** — "Recursive partitioning for heterogeneous causal effects," *PNAS* 113(27). https://www.pnas.org/doi/10.1073/pnas.1510489113. Honest causal trees; nominal coverage with honest sample-splitting. Doesn't directly give single global heterogeneity test. Useful as *visualization* tool, not inferential anchor.

### Recommendation
**Use Chernozhukov et al. 2018 BLP/GATES test**. (a) Adopted as standard in JBES-adjacent econometrics, (b) directly tests null "no heterogeneity," (c) composes cleanly with whatever DR/DML estimator we pick. Athey-Imbens as complementary plot.

---

## A.5 Practical Pitfalls

- **Sample-splitting randomness.** Small n per cell → cross-fit folds change point estimates 5-10% on re-runs. Standard practice: (i) fix `random_state`, (ii) average over ~50 random seeds, report median + across-seed SD as part of SE budget. Hartford et al. and Athey-Wager (GRF) report this stability check.
- **Convergence failures of nuisance models** with sklearn defaults — propensity LogisticRegression diverges with collinear text-PC features. Use `LogisticRegressionCV(Cs=10, max_iter=5000)` and standardize features.
- **High variance in stratified estimates.** With 4 quartiles × 4 quartiles = 16 cells, thin-cell instability. Report cell sample sizes alongside CATEs; use joint Wald tests of "any quartile differs" rather than pairwise; consider regularization on quartile-dictionary coefficients (Semenova-Chernozhukov framework allows ridged BLP).
- **Overlap diagnostics.** For continuous T, plot conditional density of T|X within each quartile; trim cells where overlap is poor.

---

## A.6 Code Skeleton

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.linear_model import LassoCV
from econml.dml import LinearDML

# Inputs:
#   Y: outcome (n,)
#   T: continuous treatment - first PC of embedding, z-scored (n,)
#   X: heterogeneity covariates (n, d_x) - just stratifiers
#   W: confounders (n, d_w)
#   strata: dict {'price_q': (n,) ints 0..3, 'len_q': (n,) ints 0..3}

est = LinearDML(
    model_y=GBR(n_estimators=300, max_depth=3, random_state=0),
    model_t=GBR(n_estimators=300, max_depth=3, random_state=0),
    discrete_treatment=False,
    cv=5,
    random_state=0,
)
est.fit(Y, T, X=X, W=W)

# CATE per cell with HC SEs
def cate_by_quartile(strata_vec):
    out = []
    for q in range(4):
        mask = strata_vec == q
        eff = est.effect(X[mask])
        ci = est.effect_interval(X[mask], alpha=0.05)
        out.append({"q": q, "n": mask.sum(),
                    "ate": eff.mean(),
                    "lo": ci[0].mean(), "hi": ci[1].mean()})
    return out

results_price = cate_by_quartile(strata['price_q'])
results_len = cate_by_quartile(strata['len_q'])

# BLP heterogeneity test
inf = est.coef__inference()
print(inf.summary_frame())

# Stability across seeds
seeds = range(50)
all_ates = []
for s in seeds:
    e = LinearDML(model_y=GBR(random_state=s), model_t=GBR(random_state=s),
                  discrete_treatment=False, cv=5, random_state=s)
    e.fit(Y, T, X=X, W=W)
    all_ates.append(e.ate(X))
print(f"Across-seed SD: {np.std(all_ates):.4f}")
```

### Citations to bundle
- `econml.dml.LinearDML` package
- Kennedy (2017) for continuous-T DR pseudo-outcome
- Semenova-Chernozhukov (2021) for BLP framework
- Chernozhukov et al. (2018) for BLP heterogeneity test
