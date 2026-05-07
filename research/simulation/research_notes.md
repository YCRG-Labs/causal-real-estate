# Simulation Study Design — Research Dossier (Item 5)

For Item 5: simulation validation of DR / DML / adversarial deconfounding / randomization estimators across SCM₀ and SCM₁ × N grid.

---

## B.1 DML Simulation Templates

### Canonical reference
Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey & Robins (2018), "Double/debiased machine learning for treatment and structural parameters," *The Econometrics Journal* 21(1):C1-C68 (https://doi.org/10.1111/ectj.12097).

§6, Tables 1-3. JBES-style template:
- DGP: `Y = θ·D + g(X) + U`, `D = m(X) + V`, X high-dimensional
- Vary: N ∈ {500, 1000, 2000}; dimension p; nuisance complexity (linear, sparse non-linear, deep non-linear)
- Estimator comparison: naïve plug-in, single-fold DML, K-fold cross-fitted (K=2, K=5), oracle
- Report: bias, variance, RMSE, coverage of 95% CI, CI length

**Use this exact reporting template.** Replace X with text embeddings; θ is text→price effect.

### Recent high-dimensional templates worth mirroring

**Athey, Imbens, Metzger & Munro (2024), "Using Wasserstein Generative Adversarial Networks for the Design of Monte Carlo Simulations," *Journal of Econometrics* 240(2):105076** (https://doi.org/10.1016/j.jeconom.2020.09.013). Train WGAN on real data; sample synthetic populations preserving covariate structure but with known intervention effect. **Right template for SCM₁ — train a small generator on real listings, inject known θ.**

**Knaus, Lechner & Strittmatter (2021), "Machine learning estimation of heterogeneous causal effects: Empirical Monte Carlo evidence," *The Econometrics Journal* 24(1):134-161** (https://doi.org/10.1093/ectj/utaa014). Empirical Monte Carlo: real covariate distribution fixed, simulate counterfactual outcomes from fitted model. **Template for SCM₀ — keep real embeddings, simulate prices from location alone.**

---

## B.2 Simulating Text Embeddings from Location

Three options:

### Option 1: Gaussian conditional on location bins
For each zip-bin `z`, sample `E ~ N(μ_z, Σ_z)` from real estimates, optionally low-rank `Σ_z = UU^T + σ²I`.
- Pro: tractable, fast, ground-truth transparent
- Con: loses manifold structure; reviewers will say "too easy"

### Option 2: Generator trained on real (text, location) pairs
Train small conditional VAE or normalizing flow `p(E | z)`.
- Pro: preserves embedding manifold
- Con: generator quality is itself a confounder

### Option 3: Latent factor model
Decompose `E = LF + ε` where L is location-loaded, F is residual style. Re-sample factor scores conditional on z.
- Pro: explicit linear structure; matches stylometric/lexical decomposition theme
- Con: linearity restrictive

### Recommendation
Report all three in appendix; lead with **Option 2** in main text. Matches Athey et al. (2024).

Concretely: train 2-layer RealNVP (Dinh, Sohl-Dickstein & Bengio 2017, ICLR, https://arxiv.org/abs/1605.08503) with location as conditioning vector on held-out half of real data. Sample N synthetic embeddings per bin. Verify by training location classifier on synthetic data; confirm accuracy ≈ accuracy on real data.

### SCM₁ extension
Start from SCM₀, add `Y = α·z + β·f(E_direct) + U`, where `E_direct` is designated coordinate or low-rank projection. Calibrate β so direct effect explains 1%, 5%, or 10% of price variance. Standard JBES "effect-size grid."

---

## B.3 Simulating Adversarial-Deconfounding Dynamics

### Background
Ganin et al. (2016), "Domain-Adversarial Training," *JMLR* 17(59):1-35 (https://jmlr.org/papers/v17/15-239.html).

Xie et al. (2017), "Controllable Invariance through Adversarial Feature Learning," *NeurIPS* (https://proceedings.neurips.cc/paper/2017/hash/8cb22bdd0b7ba1ab13d742e22eed8da2-Abstract.html).

### Key empirical precedent
Elazar & Goldberg (2018), "Adversarial Removal of Demographic Attributes from Text Data," *EMNLP* (https://aclanthology.org/D18-1002/). **The empirical analog of our finding** — discriminator at chance, frozen probe recovers attribute well above chance. Cite prominently.

Ravfogel, Elazar, Gonen, Twiton & Goldberg (2020), "Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection," *ACL* (https://aclanthology.org/2020.acl-main.647/). **Use their construction as template for simulation cell.**

### Right way to test frozen-probe diagnostic in simulation
1. Construct DGP where location signal is provably preserved (e.g., one coord of E is exactly zip-bin index plus noise)
2. Train adversarial-deconfounding pipeline; confirm discriminator loss reaches near-uniform
3. Freeze deconfounded representation; train independent MLP probe on (deconfounded → zip-bin)
4. **Failure mode to construct:** discriminator architecture too weak (linear) while signal non-linearly encoded; discriminator hits "saddle" of high loss but signal preserved. Frozen MLP probe should detect this.
5. **Success mode to verify:** both at near-chance — evidence deconfounding genuinely worked.

### Concrete simulation cell
- DGP: `E_1 = z + N(0,1)` plus random style coords `E_2..E_d`
- Adversary: linear (under-powered) vs 2-layer MLP (matched-power)
- Frozen probe: 2-layer MLP, post-hoc on held-out fold
- Report: discriminator final loss, frozen-probe accuracy, downstream estimator bias on θ
- Show divergence in linear-adversary cell — **this is the publishable finding**

---

## B.4 Coverage and Power Reporting for JBES

### Standard table format (Wager-Athey 2018, Chernozhukov 2018)
Columns: Estimator | N | Bias | SD | RMSE | 95% CI Coverage | 95% CI Length

Rows: cross-product {DR, DML, AdvDeconf, Randomization} × {N=500, 2000, 10000} × {SCM₀, SCM₁(1%), SCM₁(5%), SCM₁(10%)}

1000+ Monte Carlo replications per cell. **Bold cells where coverage falls outside [0.93, 0.97]** (binomial 95% acceptance band for 1000 reps).

### Type I error under SCM₀
For each estimator × N, fraction of replications rejecting `H₀: θ=0` at α=0.05. Should equal 0.05 ± MC error. **"Size table" — SCM₀ analogue of coverage table.**

### Power under SCM₁ × effect size × N
Fraction rejecting at α=0.05 when true θ matches 1%/5%/10% direct effect. **Plot as power curves (one panel per N, x-axis effect size, one line per estimator) — most reviewer-effective figure.**

### References for table format JBES will accept
- Belloni, Chernozhukov & Hansen (2014), "Inference on Treatment Effects after Selection," *REStud* 81(2):608-650 — exemplary
- Athey & Wager (2021), "Policy Learning with Observational Data," *Econometrica* 89(1):133-161
- Farrell, Liang & Misra (2021), "Deep Neural Networks for Estimation and Inference," *Econometrica* 89(1):181-213 — neural-nuisance simulation tables, directly relevant
- Kennedy (2023), "Towards optimal doubly robust estimation of heterogeneous causal effects," *EJS* 17(2):3008-3049

---

## B.5 Existing Simulation Libraries

- **DoWhy** (Sharma & Kiciman 2020, https://github.com/py-why/dowhy, arXiv:2011.04216). Refutation tests (placebo, random common cause, subset). **Use refutation suite as standard robustness checks, not engine.**
- **EconML** (Microsoft, https://github.com/py-why/EconML; Battocchi et al. 2019). `econml.dml.DML`, `econml.dr.DRLearner`, `econml.dml.LinearDML`, `CausalForestDML`. **Use directly for DR/DML estimators** — don't reimplement; reviewers appreciate canonical lib citation. Cross-fitting and bootstrap CI built in.
- **synthpop** (R, Nowok, Raab & Dibben 2016, JSS 74(11), https://doi.org/10.18637/jss.v074.i11). Synthetic population generation; sequential CART. **Useful baseline-data simulator** but Athey et al. (2024) WGAN more current.
- **CausalML** (Uber, https://github.com/uber/causalml; Chen et al. 2020, arXiv:2002.11631). EconML preferred for econometrics-flavored JBES audience.

---

## B.6 Skeleton Simulation Plan

| Component | Choice | Citation |
|-----------|--------|----------|
| SCM₀ DGP | Conditional flow on real (E, z) pairs, no E→Y direct | Athey et al. 2024, JoE |
| SCM₁ DGP | SCM₀ + Y = α·z + β·proj(E) + U; β grid 1%/5%/10% R² | Chernozhukov et al. 2018, EJ |
| N grid | 500, 2000, 10000 | Standard |
| Reps | 1000 per cell | Belloni et al. 2014, REStud |
| DML | `econml.dml.LinearDML`, K=5 cross-fitting | Chernozhukov et al. 2018 |
| DR | `econml.dr.LinearDRLearner` | Kennedy 2023, EJS |
| AdvDeconf | Custom + frozen-probe diagnostic | Ravfogel et al. 2020, ACL |
| Randomization | Permutation test on z | Standard |
| Reporting | Bias/SD/RMSE/coverage/CI tables; power curves | Wager & Athey 2018, JASA |
| Robustness | DoWhy refutation tests on real data | Sharma & Kiciman 2020 |

### Toolchain
- Simulation DGP: custom code, conditional flow generator (`nflows`, https://github.com/bayesiains/nflows)
- Estimators: `econml.dml.LinearDML` (DML), `econml.dr.LinearDRLearner` (DR), custom adversarial deconfounding (PyTorch, follow Ravfogel et al. 2020), custom randomization
- Evaluation: custom Monte Carlo loop, 1000-2000 reps/cell, `joblib.Parallel`
- Reporting: pandas → LaTeX via `df.to_latex()`; matplotlib for power curves

---

## Cross-Topic Synthesis

The stylometric-vs-lexical decomposition in Item 4 motivates Option-3 (latent factor) DGP here. If reviewers buy that geographic signal in text decomposes into lexical and stylometric components, the simulation should respect that decomposition: `E = E_lex(z) + E_sty(z) + ε`. SCM₁ direct effect can load on either component selectively. Power-curve figure: "estimator X recovers θ correctly when direct effect is in lexical component but biases when in stylometric component" — strong publishable finding.
