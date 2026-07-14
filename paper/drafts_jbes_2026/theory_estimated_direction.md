# Estimated-direction spatial DML: the identification and inference theory

Working derivation for the theory section. Replaces the Davis–Kahan material,
which is demoted to a consistency-rate lemma. Verified symbolically in
`data/scripts/theory_estimated_direction_verify.py`.

## Setup

A listing has a frozen embedding `w ∈ R^d` (sentence-BERT), controls `X`, and log
price `Y`. The scalar treatment is the projection of the embedding on a unit
direction `v`: `T = w'v`. The partially-linear model is

    Y = θ(v) · T + g(X) + ε,   E[ε | X, w] = 0.

DML partialling-out. Let `μ_w(X) = E[w | X]`, `ℓ(X) = E[Y | X]`, and the
residualized objects `w̃ = w − μ_w(X)`, `Ỹ = Y − ℓ(X)`. The Neyman-orthogonal
partialling-out estimand is

    θ(v) = E[T̃ Ỹ] / E[T̃²],   T̃ = w̃'v.

## Result 1 — the estimand is a closed-form function of the direction

Because `T̃ = w̃'v`,

    θ(v) = (v' a) / (v' Σ v),      a := E[w̃ Ỹ],   Σ := E[w̃ w̃'].

`a` is the residualized embedding–price covariance vector; `Σ` the residualized
embedding covariance. Both are population objects of the fixed encoder, estimable
per market. This is the whole family of estimands the analyst chooses among by
picking `v`; the choice of direction is an identification decision, not a modeling
detail.

### Structural specialization (recovers and generalizes the old Theorem 1′)

Suppose `w̃` loads on two orthogonal latent channels, a leaked spatial confounder
`s` (loading `D(v) = v'e_D`) and a non-spatial semantic factor `q`
(loading `E(v) = v'e_E`), with `Var(s)=σ_s²`, `Var(q)=σ_q²`, `s ⟂ q`, and the
outcome `Ỹ = θ q + b s + u`, `u ⟂ (s,q)`. Then `a = (b σ_s², θ σ_q²)`,
`Σ = diag(σ_s², σ_q²)`, and

    θ(v) = ( b σ_s² D(v) + θ σ_q² E(v) ) / ( σ_s² D(v)² + σ_q² E(v)² ).

- **Identified direction.** `D(v) = 0 ⟹ θ(v) = θ`. The effect is identified on
  any direction orthogonal to the leaked spatial channel; the semantic effect θ is
  recovered exactly. This is the closed-form identified direction, obtained by the
  LEACE projection of the embedding onto the concept-orthogonal subspace.

- **Variance-maximal direction and sign reversal.** The default choice, the
  leading principal component, maximizes `v'Σv`. When the spatial channel carries
  the larger residual variance, `σ_s² > σ_q²`, that direction aligns with `e_D`,
  and

        θ(v_PC) → b,

  the coefficient of the confounder on price, not the semantic effect. When
  `sign(b) ≠ sign(θ)` the reported effect is sign-reversed. This is the mechanism
  behind New York: its embedding's variance-maximal axis points at borough and
  neighborhood identity, whose price gradient runs opposite the semantic effect, so
  the naive estimate is a significant negative and the identified estimate a
  significant positive.

The sign reversal is thus not an anomaly but a prediction: the variance-maximal
direction is dangerous exactly to the degree the leaked confounder dominates the
residual variance. Gui & Veitch (2023) observe a naive-vs-adjusted sign flip
empirically (politeness, −0.038 → +0.200) but give no condition; Result 1 supplies
the closed-form condition and identifies which default choice triggers it.

## Result 2 — a valid limiting distribution accounting for the estimated direction

The direction is not known; it is estimated. `v̂` is the leading eigenvector of the
pooled residualized second-moment `Σ̂` (or the LEACE projection direction) on the
pooled sample of size `N ≈ 69,000`; `θ̂_k = θ(v̂; â_k, Σ̂_k)` is estimated per
market on `n_k ∈ [10³, 1.5·10⁴]`.

**The crux, settled honestly.** `∂θ/∂v ≠ 0` at the identified direction (it equals
`(b σ_s²/σ_q², −θ)` in the structural model), so the estimand is first-order
sensitive to `v`. The direction error is therefore **not** annihilated by
Neyman-orthogonality. It is annihilated by **rate domination**: `v̂` converges at
`N^{-1/2}` and `θ̂_k` at `n_k^{-1/2}`, and `N^{-1/2} = o(n_k^{-1/2})` for every
market because `N/n_k → ∞` by construction (the pool is the sum of a dozen markets
of this size). The Taylor term `(∂θ/∂v)'(v̂ − v*) = O_p(N^{-1/2}) = o_p(n_k^{-1/2})`
is asymptotically negligible relative to the per-market sampling error.

**Assumptions.**
- (A1) PLR model with `T = w'v*`, standard DML residualization, cross-fitted
  nuisances `μ_w, ℓ` converging at `o(n^{-1/4})` (Chernozhukov et al. 2018).
- (A2) Spectral gap: the pooled residualized second-moment operator has a bounded
  gap `δ` separating `v*` from the rest of the spectrum. Then by Koltchinskii &
  Lounici (2016), bilinear forms of the estimated spectral projector are
  asymptotically normal at rate `N^{-1/2}`, giving `‖v̂ − v*‖ = O_p(N^{-1/2})`.
  Davis–Kahan / Yu–Wang–Samworth (2015) is used only to certify this rate, never
  for a distribution.
- (A3) Rate domination and independence: `n_k / N → 0`, and `v̂` is formed
  out-of-fold (or leave-market-out) relative to market `k`'s estimation sample, so
  `v̂` and the market-`k` score are independent and no shared-data cross term
  arises.
- (A4) Spatial weak dependence: the per-market field of listings is near-epoch
  dependent on geographic distance (Jenish & Prucha 2009, 2012), with the score
  terms satisfying their CLT conditions.

**Theorem (estimated-direction spatial DML).** Under (A1)–(A4), for each market,

    √n_k ( θ̂_k − θ_k(v*) )  →d  N(0, V_k),

where `θ_k(v*) = v*' a_k / (v*' Σ_k v*)` is the direction-indexed effect of
Result 1 and

    V_k = E[ ψ_i² ] / (E[T̃²])²   under the spatial-HAC weighting,
    ψ_i = T̃_i ( Ỹ_i − θ_k T̃_i ),

with the contribution of `v̂` being `o_p(1)`. `V_k` is consistently estimated by a
Conley (1999) / Kim–Sun (2011) spatial-HAC estimator of the influence-function
sum, implementable via `conleyreg`; zip-clustering (Bester–Conley–Hansen 2011) is
the discrete-support special case.

**Proof structure (four lemmas).**
1. *Spectral-projector CLT* (Koltchinskii–Lounici 2016; Bao–Ding–Wang 2022 for the
   near-degenerate gap): `‖v̂ − v*‖ = O_p(N^{-1/2})` under (A2).
2. *Rate domination* (A3): `(∂θ/∂v)'(v̂ − v*) = o_p(n_k^{-1/2})`; with out-of-fold
   `v̂`, its error is independent of the market score, so no cross term enters `V_k`.
3. *DML expansion* (Chernozhukov et al. 2018): with `v*` fixed, `θ̂_k` admits the
   influence-function representation `√n_k(θ̂_k − θ_k) = n_k^{-1/2} Σ_i ψ_i / E[T̃²]
   + o_p(1)`, the nuisance error entering second-order by Neyman-orthogonality in
   `(μ_w, ℓ)`.
4. *Spatial CLT* (Jenish–Prucha 2009): the normalized spatially-dependent sum of
   `ψ_i` is asymptotically normal with variance consistently estimated by spatial
   HAC.

Composing 1–4 gives the theorem: `v̂`'s error is second-order (lemmas 1–2), the
fixed-direction estimator is asymptotically linear (lemma 3), and its influence
sum obeys a spatial CLT with a spatial-HAC variance (lemma 4).

## What is and is not novel (positioning)

- Novel: the closed-form estimand-as-direction `θ(v) = v'a/v'Σv`; the closed-form
  identified direction (concept-orthogonal projection); the closed-form sign-reversal
  condition tying the variance-maximal direction to confounder-channel dominance;
  and a limiting distribution that accounts for the **estimated direction** as a
  generated regressor under spatial dependence.
- Not claimed: that naive text-treatment estimates can flip sign (Gui & Veitch 2023
  show this empirically — cite it), or that text-treatment DML admits a CLT
  (Gui & Veitch Theorem 2 has one — cite it). Their CLT has no estimated-direction
  term because their treatment is not a direction; ours does, and that term, shown
  second-order by rate domination, is the specific contribution.
- Honest limit: the inference rests on `N/n → ∞` (pooled ≫ per-market) and
  out-of-fold direction construction, not on orthogonality. If a single market
  approached the pooled size, or the direction were formed in-sample, a
  Hahn–Ridder (2013) / Murphy–Topel (1985) correction term would enter; we use the
  pooled, out-of-fold construction precisely to avoid it, and say so.
