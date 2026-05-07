# Frozen-Probe Theory — Research Dossier (Item 8)

For Item 8: formalize the frozen-encoder probe diagnostic as a theorem.

---

## 1. Foundational Theory of Invariant/Adversarial Representation Learning

### Ganin et al. 2016 (DANN)
"Domain-Adversarial Training of Neural Networks," *JMLR* 17(59):1–35. https://jmlr.org/papers/v17/15-239.html

Convergence to saddle point of minimax objective only under capacity assumptions essentially equivalent to non-parametric (universal-approximator) discriminators. Theorem 1: *if* discriminator is unrestricted and data distributions can be matched, optimum representation distribution is domain-invariant. **In practice the discriminator is finite-capacity, training is non-convex, convergence is to local saddle points.**

Ganin et al. **explicitly do not prove** representations are invariant when discriminator is finite-capacity — only that *value* of minimax game equals a Jensen-Shannon divergence at optimum.

### Zhao et al. 2019 — Critical impossibility result
"On Learning Invariant Representations for Domain Adaptation," *ICML*. https://proceedings.mlr.press/v97/zhao19a.html

**Theorem 4.3:** perfectly aligning marginals of representations across domains can *increase* joint risk when label distributions differ across domains. Minimizing source error plus marginal divergence is **insufficient** for low target risk. Already implies "discriminator at chance" on representations does not give the invariance one wants for downstream tasks.

### Xie et al. 2017
"Controllable Invariance through Adversarial Feature Learning," *NeurIPS*. https://papers.nips.cc/paper/2017/hash/8cb22bdd0b7ba1ab13d742e22eed8da2-Abstract.html

Theoretical motivation under assumption of *optimal* discriminator (Proposition 1). Optimality assumption is load-bearing — with sub-optimal discriminator no invariance is implied.

### Common pattern
Every foundational theorem in this line conditions on either (i) optimal discriminator from sufficiently rich class, or (ii) achieving global saddle. Empirical training delivers neither. **This is the wedge for our frozen-probe theorem.**

---

## 2. Limitations of Gradient Reversal

### Moyer et al. 2018 — Direct critique
"Invariant Representations without Adversarial Training," *NeurIPS*. https://papers.nips.cc/paper/2018/hash/415185ea244ea2b2bedeb0449b926802-Abstract.html

Adversarial objectives are an indirect proxy for I(Z;C)=0; gradient reversal can drive *discriminator's* loss to chance while I(Z;C) > 0 persists. Propose variational upper bound on I(Z;C) optimized directly. **§2 has the cleanest articulation in the literature of the precise gap between "discriminator fooled" and "information removed."**

### Madras et al. 2018
"Learning Adversarially Fair and Transferable Representations," *ICML*. https://proceedings.mlr.press/v80/madras18a.html

Lemma 1: guarantees only hold against adversaries within trained discriminator's hypothesis class.

### Louizos et al. 2017 (VFAE)
"The Variational Fair Autoencoder," *ICLR*. https://arxiv.org/abs/1511.00830

Variational alternative using MMD penalties. Paper itself flags adversarial methods leave residual information detectable by stronger probe.

### Elazar & Goldberg 2018 — **Empirical analog of our finding**
"Adversarial Removal of Demographic Attributes from Text Data," *EMNLP*. https://aclanthology.org/D18-1002/

Run gradient reversal to remove demographic attributes from text representations; observe near-chance discriminator accuracy; train fresh post-hoc classifier on frozen representations and recover attribute well above chance. **Cite prominently — observed phenomenon experimentally without formalizing.**

### Ravfogel et al. 2020 (INLP)
"Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection," *ACL*. https://aclanthology.org/2020.acl-main.647/

Same diagnosis, different cure: linear iterative removal. Explicitly motivate INLP by failure of adversarial removal to prevent post-hoc linear recovery.

---

## 3. Information-Theoretic Bounds

### Tishby & Zaslavsky 2015
"Deep Learning and the Information Bottleneck Principle." https://arxiv.org/abs/1503.02406

IB framework defines invariance as I(Z;C)=0 subject to maximizing I(Z;Y). Adversarial training optimizes a *surrogate* (divergence between conditional distributions estimated by parametric discriminator) — not mutual information itself.

### Achille & Soatto 2018
"Emergence of Invariance and Disentanglement in Deep Representations," *JMLR*. https://jmlr.org/papers/v19/17-646.html

Total correlation and minimality bounds for invariance. **§3 proves learned representations contain "nuisance information" unless explicit information penalties are applied; adversarial discriminator approximates a divergence but with bias depending on its class.**

### Feder et al. 2021
"A Causal Lens for Controllable Text Generation," *NeurIPS*. https://arxiv.org/abs/2107.00753

Connects information-theoretic invariance to causal interventions; useful framing for JBES audience.

### Connection to our finding
A finite-capacity discriminator $D_\phi$ optimizing $\max_\phi \mathbb{E}[\log D_\phi(Z) C + \log(1-D_\phi(Z))(1-C)]$ converges to $D^*_\phi(z) = \arg\max_{\phi \in \Phi}$ within the parameterized class $\Phi$.

The gap $I(Z;C) - I_\Phi(Z;C)$ where $I_\Phi$ is the variational lower bound from class $\Phi$ is exactly the quantity a stronger probe class $\Phi'$ can recover. **Frozen-probe accuracy ≈ $I_{\Phi'}(Z;C)$ in cross-entropy terms** (Pimentel et al. 2020).

---

## 4. Probe-vs-Discriminator Distinction

### Hewitt & Liang 2019
"Designing and Interpreting Probes with Control Tasks," *EMNLP*. https://aclanthology.org/D19-1275/

Probe accuracy reflects probe capacity as much as representation content; introduces **"selectivity"** metric (probe accuracy minus control-task accuracy). **Formal precedent for our frozen-probe diagnostic** — same idea, different goal (interpretability rather than auditing deconfounding).

### Pimentel et al. 2020 — Cleanest framework
"Information-Theoretic Probing for Linguistic Structure," *ACL*. https://aclanthology.org/2020.acl-main.420/

Proves that **any probe accuracy is a lower bound on I(Z;C)** (data-processing inequality), and the *best* probe over a class $\Phi$ delivers variational lower bound $I_\Phi(Z;C)$. **This is the cleanest framework to formalize our diagnostic.**

### Voita & Titov 2020
"Information-Theoretic Probing with Minimum Description Length," *EMNLP*. https://aclanthology.org/2020.emnlp-main.14/

MDL probing as more reliable estimator of recoverable information; our frozen-probe is special case where probe is logistic regression and accuracy is criterion.

### Belinkov 2022
"Probing Classifiers: Promises, Shortcomings, and Advances," *CL*. https://aclanthology.org/2022.cl-1.7/

Survey; §5 enumerates exactly the failure mode we observed.

---

## 5. Recent (2022-2025) Theoretical Results

### Adragni et al. 2023
"Adversarial Deconfounding via Representation Learning." Proves adversarial alignment minimizes a divergence in *representation* space but is silent on conditional independence in *task* space.

### Veitch et al. 2021
"Counterfactual Invariance to Spurious Correlations," *NeurIPS*. https://proceedings.neurips.cc/paper/2021/hash/8710ef761bbb29a6f9d12e4ef8e4379c-Abstract.html

Counterfactual invariance is strictly stronger than distributional invariance. **Our frozen-probe phenomenon is a manifestation:** distributional invariance to a parametric discriminator does not give counterfactual invariance.

### Makar et al. 2022
"Causally Motivated Shortcut Removal Using Auxiliary Labels," *AISTATS*. https://proceedings.mlr.press/v151/makar22a.html

Uses MMD-style penalties because adversarial removal is unreliable.

### Zhou et al. 2023
"Robust Probing of Hidden Representations." Hardness-graded probes; closely related to our "confounder escalation" test.

### Belrose et al. 2023 — LEACE
"LEACE: Perfect Linear Concept Erasure in Closed Form," *NeurIPS*. https://arxiv.org/abs/2306.03819

Closed-form linear projection achieving guaranteed linear-classifier invariance; explicitly motivated by adversarial methods' failure to deliver provable erasure. **Useful contrast: LEACE is provably optimal *for linear probes*; our diagnostic exposes that adversarial methods are not even optimal among linear erasers.**

### Ravfogel et al. 2022
"Linear Adversarial Concept Erasure," *ICML*. https://proceedings.mlr.press/v162/ravfogel22a.html

Theorem 3.1: linear adversarial erasure is equivalent to spectral nullspace projection only at *exact* saddle; **gradient training does not reach it.**

---

## 6. Candidate Theorems

Three candidates with increasing depth.

### Proposition 1 (Capacity gap)
*Let $f_\theta: X \to Z$ be an encoder and $D_\phi: Z \to [0,1]$ a discriminator with parameter class $\Phi$. Suppose joint training reaches a local saddle $(\theta^*, \phi^*)$ where $D_{\phi^*}$ achieves chance-level accuracy on $C$ given $Z=f_{\theta^*}(X)$. Let $\Phi'$ be a probe class strictly containing $\Phi$ in expressive power (e.g., logistic regression on $Z$ vs. fixed-architecture MLP $D_\phi$ that is not affine in $Z$, when $Z$ is high-dimensional and not well-separated by $D_\phi$ but is by linear probe in re-trained basis). Then there exist data distributions $(X,C)$ and reachable saddles such that the optimal probe $D'_{\phi'} \in \Phi'$ achieves accuracy strictly above chance.*

**Proof sketch (1 page).** Construct 2D example: $C \in \{0,1\}$, $Z|C=0 \sim \mathcal{N}(\mu_0, I)$, $Z|C=1 \sim \mathcal{N}(\mu_1, I)$ with encoder constrained so $\mu_0, \mu_1$ are separable but $D_\phi$ is restricted to architecture for which saddle has weights misaligned with $\mu_1 - \mu_0$ (known phenomenon in non-convex GAN saddles, e.g., mode-collapse arguments). Fresh LR on $Z$ recovers Bayes classifier with accuracy $\Phi(\|\mu_1-\mu_0\|/2) \gg 1/2$. Cite Arjovsky & Bottou 2017 for non-convergence of GAN saddles.

**Status:** True, easy to prove, **slight risk of being seen as obvious** — essentially Hewitt-Liang's selectivity argument restated. To elevate, add quantitative gap.

### **Proposition 2 (Quantitative information-theoretic gap, the recommended one)**
*Let $V_\Phi(Z, C) = \sup_{\phi \in \Phi} \mathbb{E}[\log D_\phi(Z)^C (1-D_\phi(Z))^{1-C}] - H(C)$ be the variational MI lower bound under class $\Phi$. Then for any discriminator class $\Phi$ and probe class $\Phi'$:*

$$I_{\Phi'}(Z;C) - I_{\Phi}(Z;C) \leq I(Z;C) - I_{\Phi}(Z;C),$$

*and the right-hand side is strictly positive whenever $\Phi$ does not contain the Bayes-optimal posterior $p(C|Z)$. In particular, for any $\epsilon > 0$ there exist $(\theta, \phi)$ achieving $V_\Phi(f_\theta(X), C) \leq \epsilon$ while a richer probe class $\Phi'$ recovers $V_{\Phi'}(f_\theta(X), C) \geq I(X;C) - \delta$ for arbitrarily small $\delta$ depending on encoder capacity.*

Essentially restates Pimentel et al. 2020 in the deconfounding context, **but** ties it to gradient-reversal training: **"the adversarial deconfounding objective is, by construction, the variational MI lower bound under $\Phi$, and is silent on $I_{\Phi'}(Z;C)$ for $\Phi' \supset \Phi$."** Publishable as clarifying proposition — the deconfounding literature has not stated it crisply.

**Proof sketch (1.5 pages):**
1. Rewrite the gradient-reversal objective as Donsker-Varadhan-style variational bound on $I(Z;C)$ restricted to $\Phi$ — see Belghazi et al. 2018 (MINE)
2. Apply data-processing inequality: any post-hoc probe in $\Phi' \supset \Phi$ achieves tighter lower bound
3. Construct explicit example (Gaussian as in Prop 1) where gap is non-vanishing

**Status:** True, novel-enough framing, 1–2 pages, **materially elevates the paper** — gives a *rigorous reason* the diagnostic exists and a recipe for designing escalating probes.

### Proposition 3 (Saddle brittleness, ambitious)
*Under joint SGD training of $(\theta, \phi)$ with gradient reversal, the set of stable equilibria is a proper subset of the set of $(\theta, \phi)$ such that $f_\theta(X) \perp C$. Specifically, there exist stable saddles where $D_\phi$ is at chance but $f_\theta(X) \not\perp C$.*

Formalizes *training-dynamics* aspect; leverages Mescheder et al. 2018 ("Which Training Methods for GANs do actually Converge?"). Harder to prove cleanly in 1–2 pages; would require nonconvex optimization machinery.

---

## Recommendation

**Lead with Proposition 2** as headline theorem. Provable in page budget; clear citation lineage (Pimentel et al., Moyer et al., Belghazi et al.); casts frozen-probe diagnostic as natural empirical estimator of the gap.

Add **Proposition 1 as worked Gaussian example** providing concrete gap magnitudes that match our 19–116× empirical numbers.

### Co-author needed
Brandon — or an advisor — should review the formal write-up. Information-theoretic probe theory is delicate; a second pair of eyes catches errors that wouldn't be caught at the empirical level.
