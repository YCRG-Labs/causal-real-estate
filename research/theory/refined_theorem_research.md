# Refined Theorem Research: The Frozen-Probe Diagnostic for Adversarial Deconfounding

**Status:** Research note for Item 8 / Proposition 2 of the dossier.
**Date:** 2026-04-29.
**Bottom line up front.** The two inequalities $V_\Phi(Z;C) \le V_{\Phi'}(Z;C) \le I(Z;C)$ are **essentially folklore** in the variational MI / probing literature (Barber–Agakov 2003; Poole et al. 2019; Pimentel et al. 2020; Xu et al. 2020). The strict-positivity gap clause is a corollary of the Bayes-optimality characterization of the Donsker–Varadhan / NWJ supremum. Therefore Proposition 2 *as currently written* would not survive a careful referee. **What is genuinely original** — and what we should claim — is the **explicit identification of the gradient-reversal training objective with the discriminator-class variational MI lower bound, and the resulting consistency of the frozen-probe diagnostic as the natural plug-in estimator of the gap $V_{\Phi'} - V_\Phi$.** The novelty is in the *framing for adversarial deconfounding* and in turning the gap into an empirical auditing protocol with quantitative guarantees, not in the inequalities themselves.

---

## Section A. Direct precedents found

### A.1 Pimentel, Valvoda, Hall Maudslay, Zmigrod, Williams, Cotterell (2020). "Information-Theoretic Probing for Linguistic Structure." *ACL.*

This is the paper that comes closest to being a direct precedent. Their **central proposition (informally stated as their main claim, not numbered as a theorem in the body)** is:

> Any probe $q_\theta(C\mid Z)$ supplies a variational lower bound on the mutual information $I(Z;C)$, namely $I(Z;C) \ge H(C) - \mathbb{E}_{(Z,C)}[-\log q_\theta(C\mid Z)]$, and the bound is tight iff $q_\theta = p(C\mid Z)$.

The bound is exactly the **Barber–Agakov 2003** lower bound applied to representations. From this they derive their headline argument: **the highest-capacity probe is preferred**, because larger probe class $\Rightarrow$ higher (tighter) lower bound. This is precisely the right-hand inequality of our proposed Proposition 2.

What Pimentel et al. **do not do**:
- They do not relate this to adversarial training or gradient reversal.
- They do not study the gap as an empirical diagnostic of leakage in deconfounded representations.
- They do not study the asymmetric setting of two probe classes — discriminator-class $\Phi$ trained jointly vs. probe class $\Phi'$ trained post hoc — which is what we need.

So our *inequalities* are downstream of Pimentel et al.; our **application** is not.

### A.2 Xu, Zhao, Song, Stewart, Ermon (2020). "A Theory of Usable Information Under Computational Constraints." *ICLR.*

Defines **predictive $\mathcal{V}$-information** $I_\mathcal{V}(X \to Y)$ for a function family $\mathcal{V}$, with the explicit property (Proposition 1, Theorem 2 of their paper) that:

$$\mathcal{V} \subseteq \mathcal{V}' \;\;\Longrightarrow\;\; I_\mathcal{V}(X \to Y) \le I_{\mathcal{V}'}(X \to Y) \le I(X;Y).$$

This is **literally our two-inequality chain** with $X = Z$, $Y = C$, $\mathcal{V} = \Phi$, $\mathcal{V}' = \Phi'$. They also prove PAC-style consistency for the empirical estimator (their Theorem 3). For fair-representation learning they show explicitly (Section 5.2) that minimizing $I_\mathcal{V}(Z \to C)$ over a restricted $\mathcal{V}$ leaves $I_{\mathcal{V}'}$ for richer $\mathcal{V}'$ unconstrained.

**This is the most direct precedent, and it is decisive.** Our two inequalities are a special case of Xu et al.'s Proposition 1; the strict-positivity claim is their Remark following Theorem 2. We must cite this paper prominently and **not claim the inequalities as our contribution**.

### A.3 Moyer, Gao, Brekelmans, Galstyan, Ver Steeg (2018). "Invariant Representations without Adversarial Training." *NeurIPS.*

Section 2 motivates the work by observing that adversarial objectives are surrogates for $I(Z;C) = 0$ and that "fooling the discriminator" does not entail $I(Z;C) = 0$. They then derive a **variational upper bound** on $I(Z;C)$ (their Eq. 6, Eq. 9) that is optimized directly. They also use a **post-hoc adversary** as an evaluation tool (Section 4) and explicitly note that the post-hoc adversary's accuracy is a *lower-bound proxy* for residual MI.

What they **do not** prove:
- A quantitative lower bound on the gap $I(Z;C) - V_\Phi(Z;C)$.
- Any inequality of the form $V_\Phi \le V_{\Phi'} \le I$. They go straight to the upper-bound formulation.
- Consistency of the post-hoc adversary as an estimator of the gap. They use it as a heuristic.

So Moyer et al. articulate the *qualitative* gap (their Section 2 is the cleanest articulation in the deconfounding literature, as our notes say) but do not prove the quantitative two-class hierarchy nor formalize the diagnostic.

### A.4 Belghazi, Baratin, Rajeshwar, Ozair, Bengio, Courville, Hjelm (2018). "MINE: Mutual Information Neural Estimation." *ICML.*

Establishes that the Donsker–Varadhan and NWJ ($f$-divergence) bounds, when supremized over a function class $\mathcal{F}$, yield **lower bounds** on $I(X;Y)$ that **monotonically tighten as $\mathcal{F}$ enlarges and become tight when $\mathcal{F}$ contains the optimal log-density-ratio** (their Theorem 1, Theorem 2). This is the textbook variational MI hierarchy. The DPI for these bounds is implicit in the supremum monotonicity argument and does not require separate proof.

### A.5 Poole, Ozair, van den Oord, Alemi, Tucker (2019). "On Variational Bounds of Mutual Information." *ICML.*

A survey/unification of variational MI bounds (BA, NWJ, MINE, InfoNCE). Proposition 1 states that all listed lower bounds are **tight iff the critic equals the optimal log-density-ratio**, equivalent to containing $p(C\mid Z)$ in the parameterization. The class-monotonicity property is stated implicitly throughout (their discussion of "bias–variance trade-off across critic capacity") but they focus on bias/variance rather than on hierarchies of nested classes.

### A.6 Cheng, Hao, Dai, Liu, Gan, Carin (2020). "CLUB: A Contrastive Log-ratio Upper Bound of Mutual Information." *ICML.*

Provides a *upper* bound on MI that is tight when the variational posterior matches the true conditional. CLUB is the right tool to *upper-bound* $I(Z;C)$ and could in principle bracket residual leakage: if CLUB upper bound $\to 0$, then $I(Z;C)\to 0$ in truth. We should cite CLUB as the natural complement to our diagnostic — the diagnostic gives a lower bound, CLUB gives an upper bound; together they bracket the gap.

### A.7 Choi, Kim, Watanabe (2024). "Understanding Probe Behaviors through Variational Bounds of Mutual Information." *ICASSP.* arXiv:2312.10019.

The most recent (2024) directly relevant work. Connects linear probing of self-supervised speech representations with the variational bounds in Poole et al. 2019, observing that intermediate-layer probe accuracies form non-monotone curves and explaining this via a separability-vs-MI tradeoff. **They do not address adversarial deconfounding or the discriminator-vs-probe gap**, but the framing of "the probe is a variational MI lower bound and probe class matters" is now an active 2024 research line. We should cite to show the framing is current and recognized, not stale.

### A.8 Belrose, Schneider-Joseph, Ravfogel, Cotterell, Raff, Biderman (2023). "LEACE: Perfect Linear Concept Erasure in Closed Form." *NeurIPS.*

LEACE provably zeroes out *linear* probe accuracy in closed form. Their Theorem 1 / Theorem 2 establish necessary and sufficient conditions for *all* affine classifiers on $Z$ to be at chance. They explicitly motivate the work by the observation that gradient-based / adversarial methods do not provably erase even linearly. **Useful contrast for our paper:** LEACE shows that even within the class of linear probes the adversarial method can underperform a closed-form orthogonalization — a stronger empirical signature of the gap we are formalizing.

### A.9 Ravfogel, Vargas, Goldberg, Cotterell (2022). "Linear Adversarial Concept Erasure." *ICML.*

Theorem 3.1: linear adversarial erasure equals spectral nullspace projection **only at the exact saddle**; gradient training does not in general reach it. This is a *training-dynamics* analog of our gap (saddle quality, not class capacity), worth citing as a parallel failure mode.

### A.10 Zhao, Dan, Aragam, Jaakkola, Gordon, Ravikumar (2022). "Fundamental Limits and Tradeoffs in Invariant Representation Learning." *JMLR.*

Information-theoretic lower bounds on the joint accuracy–invariance tradeoff. Acknowledges (Section 5) that **finite-capacity adversarial discriminators only achieve approximate demographic parity**. They derive lower bounds on prediction error under approximate parity but do not formalize the discriminator-vs-probe gap as such.

### A.11 Elazar & Goldberg (2018). "Adversarial Removal of Demographic Attributes from Text Data." *EMNLP.*

Empirical demonstration of exactly the phenomenon we are formalizing: adversarial removal drives the live discriminator to chance, but a fresh post-hoc classifier recovers the protected attribute well above chance. They do not prove the result; they observe it.

### A.12 Ragonesi, Volpi, Cavazza, Murino (2021). "Learning Unbiased Representations via Mutual Information Backpropagation." *CVPR-W.*

Replaces the adversarial discriminator with a MINE estimator of $I(Z;C)$ to avoid exactly the failure mode we describe. Implicitly assumes the gap exists and is the reason adversarial training fails; does not formalize.

---

## Section B. What is already established

Combining Sections A.1–A.6, the following are **established and citable**:

1. **(Barber–Agakov / Pimentel)** Any probe $q_\theta(C\mid Z)$ gives $I(Z;C) \ge H(C) - \mathbb{E}[-\log q_\theta(C\mid Z)] =: V_{q_\theta}(Z;C)$, with equality iff $q_\theta = p(C\mid Z)$.

2. **(Xu et al. 2020, Proposition 1; Belghazi et al. 2018; Poole et al. 2019)** For nested function classes $\Phi \subseteq \Phi'$,
$$V_\Phi(Z;C) := \sup_{q\in\Phi} V_q(Z;C) \;\le\; V_{\Phi'}(Z;C) \;\le\; I(Z;C),$$
with the right inequality strict whenever $\Phi'$ does not contain the Bayes-optimal posterior.

3. **(Xu et al. 2020, Theorem 3)** The empirical $\mathcal{V}$-information $\hat I_\mathcal{V}(Z\to C)$ converges to $I_\mathcal{V}(Z\to C)$ at PAC rates, given mild capacity conditions (uniform convergence over $\mathcal{V}$ via Rademacher complexity).

4. **(Donsker–Varadhan; Belghazi et al.)** The supremum of the cross-entropy game is achieved at the Bayes posterior, and any minimax saddle of an adversarial training game restricted to $\Phi$ has value equal to $V_\Phi(Z;C)$ (under regularity: convexity in the discriminator, sufficient training, exact saddle).

5. **(Moyer et al. 2018, §2; Elazar & Goldberg 2018)** Empirically, gradient-reversal-trained encoders produce representations on which a post-hoc classifier of higher capacity can recover protected attributes. The phenomenon is documented but not quantified.

6. **(Hewitt & Liang 2019)** The general philosophy of "control tasks" / selectivity — that a *gap* between a richer and a poorer probe is the meaningful diagnostic — is established in the probing literature, though for a different purpose (probe interpretability).

**What is therefore not novel:** The two-inequality chain itself; the assertion that adversarial discriminators are variational MI lower bounds; the qualitative observation that a richer probe can recover what a weaker discriminator missed.

---

## Section C. What is genuinely novel for us

Stripping out the folklore, the following claims survive as legitimately original contributions:

**C.1 (The empirical-equivalence claim).** *The training loss of a gradient-reversal adversarial deconfounder, evaluated at any iterate, is — up to an additive constant — the negative empirical Barber–Agakov lower bound $-\hat V_\Phi(Z;C)$ for the discriminator class $\Phi$ on the current encoder $E_{\phi_t}$.* This is a one-line algebraic identification but it has, to our knowledge, not been written down in the deconfounding literature. It immediately turns "discriminator achieves chance" into "$\hat V_\Phi(Z;C) \le \epsilon$".

**C.2 (The diagnostic-as-estimator claim).** *Let $\hat V_{\Phi'}(Z;C)$ denote the post-hoc plug-in estimator obtained by training a fresh probe in class $\Phi'$ on the frozen encoder. Under standard PAC-Bayesian / Rademacher assumptions on $\Phi'$, $\hat V_{\Phi'}(Z;C) \to V_{\Phi'}(Z;C)$ as the held-out sample grows. Therefore* $\hat V_{\Phi'} - \hat V_\Phi$ *is a consistent estimator of the discriminator–probe gap $V_{\Phi'}(Z;C) - V_\Phi(Z;C) \ge 0$.* This combines Xu et al. 2020 Theorem 3 with claim C.1; the assembly is novel and is the operational content of the diagnostic.

**C.3 (The auditing-failure-mode claim).** *Whenever $\Phi$ is the discriminator class used during training and $\Phi' \supsetneq \Phi$ contains the Bayes-optimal posterior $p(C\mid Z_{\theta^\star})$ at the trained encoder, the diagnostic recovers the entire residual mutual information: $\hat V_{\Phi'} - \hat V_\Phi \to I(Z_{\theta^\star}; C) - V_\Phi(Z_{\theta^\star}; C)$.* This is a sharp characterization of when the diagnostic is **complete** (sees all leakage) versus **partial** (lower-bounds it). To our knowledge no prior paper states this explicitly.

**C.4 (Connection to gradient-reversal training, narrowly).** *In the gradient-reversal optimization, even at exact saddle points $(\phi^\star, \psi^\star)$ of the adversarial game restricted to $\Phi$, $V_{\Phi'}(Z_{\phi^\star}; C) - V_\Phi(Z_{\phi^\star}; C)$ can be made arbitrarily close to $H(C)$ for appropriate choices of $\Phi \subsetneq \Phi'$ and data distribution.* This is a saddle-quality statement and is genuinely new (we have not located a prior version). It is provable in 1–2 pages with a Gaussian construction.

**Honest summary.** Our contribution is **C.1 + C.2 + C.3 + C.4 packaged together as "the frozen-probe diagnostic for adversarial deconfounding,"** with the inequalities themselves attributed to Xu et al. 2020 and Pimentel et al. 2020. The headline theorem should be C.4, the consistency result should be C.2, and the framing should be the empirical-equivalence C.1. **We must not claim the two-inequality chain.**

---

## Section D. Suggested formal statement

Re-statement of Proposition 2, in two parts. Inequalities cited; novelty isolated in the second part.

### Proposition 2 (Frozen-Probe Diagnostic).

Let $E_\phi:\mathcal{X}\to\mathcal{Z}$ be an encoder, $C$ a confounder, and $\Phi, \Phi'$ two parameterized classes of probabilistic classifiers $\mathcal{Z}\to[0,1]$ with $\Phi \subseteq \Phi'$. For any class $\mathcal{F}$, define the **Barber–Agakov variational lower bound**
$$V_\mathcal{F}(Z;C) \;:=\; \sup_{q\in\mathcal{F}} \big[H(C) - \mathbb{E}_{(Z,C)}[-\log q(C\mid Z)]\big].$$

**(a) (Folklore; Xu et al. 2020 Prop. 1; Pimentel et al. 2020.)** $0 \le V_\Phi(Z;C) \le V_{\Phi'}(Z;C) \le I(Z;C)$, with the rightmost inequality strict whenever $\Phi'$ does not contain the Bayes-optimal posterior $p(C\mid Z)$.

**(b) (New: empirical equivalence and consistency.)** Suppose an adversarial encoder is trained by gradient reversal against a discriminator class $\Phi$ until the live discriminator achieves cross-entropy $\hat L_\Phi$. Then:
1. $\hat V_\Phi(Z_{\hat\phi}; C) = H(C) - \hat L_\Phi$ exactly (in expectation over the training distribution), so "discriminator at chance" $\Leftrightarrow$ $\hat V_\Phi \le \epsilon$ for the trained $\hat\phi$.
2. Let $\hat V_{\Phi'}^{\text{post}}$ denote the empirical Barber–Agakov bound obtained by training a fresh post-hoc probe in $\Phi'$ on the *frozen* encoder $E_{\hat\phi}$, evaluated on a held-out sample. Under uniform-convergence conditions on $\Phi'$ (finite Rademacher complexity, bounded log-loss), $\hat V_{\Phi'}^{\text{post}} \to V_{\Phi'}(Z_{\hat\phi};C)$ in probability.
3. Hence $\hat V_{\Phi'}^{\text{post}} - \hat V_\Phi$ is a consistent lower-bound estimator of the residual mutual information $I(Z_{\hat\phi};C) - V_\Phi(Z_{\hat\phi};C)$, exact when $\Phi'$ contains the Bayes-optimal posterior.

**(c) (New: failure of gradient reversal at exact saddle.)** There exists a data distribution over $(X,C)$ and choices $\Phi \subsetneq \Phi'$ such that for every saddle $(\phi^\star,\psi^\star)$ of the gradient-reversal game restricted to $\Phi$, we have $V_\Phi(Z_{\phi^\star};C) = 0$ but $V_{\Phi'}(Z_{\phi^\star};C) \ge H(C) - \delta$ for arbitrary $\delta > 0$ (i.e., near-perfect post-hoc recovery).

### Proof sketch.

*(a)* Restate Xu et al. 2020 Proposition 1, using the equivalence of the Barber–Agakov bound with $\mathcal{V}$-information for the cross-entropy gain function. Strictness is by Donsker–Varadhan: the supremum equals $I(Z;C)$ iff the supremizing $q^\star$ equals $p(C\mid Z)$.

*(b1)* Algebraic identity: for any class $\Phi$, $\sup_{q\in\Phi}\big\{H(C) - \mathbb{E}[-\log q(C\mid Z)]\big\}$ is exactly the gradient-reversal discriminator's converged value (negated and re-centered). Standard.

*(b2)* Apply Xu et al. 2020 Theorem 3 (PAC bound on $\mathcal{V}$-information) to $\mathcal{V} = \Phi'$ on the held-out sample, using that $E_{\hat\phi}$ is fixed.

*(b3)* Difference of two consistent estimators is consistent for the difference; combine with (a).

*(c)* Construction. Take $X = (X_1, X_2)\in\mathbb{R}^2$, $C\in\{0,1\}$, with $X_1\mid C \sim \mathcal{N}(\pm 1, 1)$ and $X_2\mid X_1, C \sim \mathcal{N}(C\cdot X_1, 1)$. Let $\Phi$ = linear classifiers in $Z$ and $\Phi'$ = degree-2 polynomial classifiers in $Z$. For $E_\phi$ in the family of orthogonal projections, there is a saddle at which the linear margin in $Z$ is exactly zero (so $V_\Phi = 0$) yet the quadratic interaction $Z_1 Z_2$ identifies $C$ almost surely (so $V_{\Phi'}\to H(C)$). XOR in disguise.

**Page budget.** Statements (a)–(c) and proof sketch fit comfortably in 1.5 pages of a JBES-style appendix, with the construction in (c) taking a half-page.

### Citation lineage to make explicit in the paper

- "Inequality (a) is well-known and appears as Proposition 1 of Xu et al. (2020) (in the language of $\mathcal{V}$-information) and as the central observation of Pimentel et al. (2020) (in the language of probe capacity); we restate it for self-containedness."
- "Part (b) connects (a) to the gradient-reversal training objective and formalizes the post-hoc adversary used as a heuristic by Moyer et al. (2018) and Elazar & Goldberg (2018) as a consistent statistical estimator."
- "Part (c) is to our knowledge new, although in spirit it parallels saddle-quality results of Ravfogel et al. (2022) for the linear case."

---

## Section E. Suggested numerical experiment

A controlled synthetic verification with three escalating tests.

**E.1 Two-Gaussian linear-vs-MLP sanity check.**
- Data: $X\mid C \sim \mathcal{N}(\mu_C, I_d)$ with $\mu_0 = -\mu_1$, $\|\mu_1\| = 0.5$, $d = 64$. Ground-truth $I(X;C)$ computable in closed form.
- Encoder: 2-layer MLP $E_\phi$ with bottleneck 8.
- Discriminator $\Phi$: linear logistic regression on $Z$. Train with gradient reversal until live discriminator $\hat V_\Phi \le 0.01$.
- Probe $\Phi'$: 3-layer MLP. Train post-hoc on frozen $Z$.
- **Plot:** $\hat V_\Phi$ vs. $\hat V_{\Phi'}^{\text{post}}$ across training iterations of the encoder.
- **Expected:** $\hat V_\Phi$ collapses to 0; $\hat V_{\Phi'}^{\text{post}}$ remains bounded away from 0 (this is the qualitative result). Bracket with CLUB upper bound.

**E.2 XOR-style construction matching the proof of (c).**
- Data: as in proof construction. Ground-truth $I(X;C) = H(C) = \log 2$.
- Discriminator $\Phi$ = linear, probe $\Phi'$ = quadratic kernel SVM or 2-layer MLP with quadratic features.
- **Plot:** Encoder iteration vs. (i) live discriminator accuracy (should go to 50%), (ii) frozen-probe accuracy in $\Phi'$ (should stay near 100%), (iii) CLUB upper bound on residual MI (should also stay near $\log 2$).
- **Success criterion:** All three curves separate cleanly; the gap $\hat V_{\Phi'}^{\text{post}} - \hat V_\Phi$ approximates $\log 2$ within a few percent on a 10k-sample held-out set.

**E.3 Probe-class hierarchy / "escalation curve."**
- Same data as E.1.
- Define $\Phi_k$ = MLP with $k$ hidden units, $k\in\{1,2,4,8,16,32,64,128\}$.
- Train discriminator at $k_0 = 4$, freeze, then run post-hoc probes at every $k$.
- **Plot:** $k$ on $x$-axis, $\hat V_{\Phi_k}^{\text{post}}$ on $y$-axis; horizontal line at $\hat V_{\Phi_{k_0}}$ (live discriminator value at end of training).
- **Expected:** Monotone non-decreasing curve (verifying inequality (a) empirically); plateau approaching CLUB upper bound on $I(Z;C)$, providing visual evidence of the consistency claim (b2).

**E.4 Real-data anchor (optional).** Re-run E.3 on the German Credit / UCI Adult / COMPAS deconfounding setup used by Madras 2018 or Moyer 2018 to confirm the synthetic story transfers.

**Failure modes to budget for and report.**
- Probe $\Phi'$ overfitting $\Rightarrow$ inflated $\hat V_{\Phi'}^{\text{post}}$. Report selectivity (Hewitt & Liang) on a control task with random labels.
- Variance of $\hat V_{\Phi'}^{\text{post}}$ across seeds; report confidence intervals from $\ge 10$ random seeds for the encoder and $\ge 5$ for the probe per encoder.

**What success looks like for the theory paper:** Figure 1 in the paper is the curve from E.3. It is the empirical analog of inequality (a), with the discriminator marked as a single point well below the post-hoc probes. This is a one-figure proof-of-concept that the diagnostic is meaningful, and ties the theoretical proposition to a recipe.

---

## Section F. Final recommendations

1. **Rewrite Proposition 2** along the lines of Section D — with (a) explicitly attributed to Xu et al. 2020 and Pimentel et al. 2020, and (b)–(c) claimed as our contribution.
2. **Lead with the consistency statement (b3)** as the headline operational result. This is the cleanest novel claim. Phrasing: *"The frozen-probe diagnostic is a consistent plug-in estimator of the discriminator–probe gap in $\mathcal{V}$-information."*
3. **Include the Gaussian/XOR construction (c)** as a worked example, not as a separate theorem, to keep the page budget tight.
4. **Reframe the introduction** to credit Moyer et al. (2018) §2 and Elazar & Goldberg (2018) for the qualitative observation, and to position our contribution as the formalization plus the empirical estimator.
5. **Bracket with CLUB** (Cheng et al. 2020) in the experiments to give an upper bound complement, so referees see a two-sided statement rather than a one-sided lower bound.
6. **Cite Choi, Kim, Watanabe (2024) ICASSP** as confirmation that the variational-bounds-of-probes framing is a current research direction, not a 2020 artifact.
7. **Co-author / advisor review (Brandon).** Section D's proof of (c) should be checked carefully; it is the only part where a referee can find a real error.

**One-line reframe of the contribution for the abstract.** *"We formalize the empirical observation that gradient-reversal-trained encoders leak information to richer post-hoc probes by identifying the trained discriminator's loss with a Barber–Agakov variational MI lower bound under its own class, and showing that a frozen-encoder probe of strictly higher capacity is a consistent estimator of the residual gap. The diagnostic is exact when the post-hoc probe contains the Bayes-optimal posterior, and we exhibit gradient-reversal saddles at which the gap is essentially the entire mutual information."*
