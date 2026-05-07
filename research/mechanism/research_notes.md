# Mechanism Analysis — Research Dossier (Item 4)

For Item 4: identifying *what specifically* in language encodes geographic location.

---

## A.1 Stylometric Feature Extraction: Tools and Citations

Three layered toolchains for replicable stylometric features:

### textstat (Python, MIT)
https://github.com/textstat/textstat. Implements Flesch Reading Ease, Flesch-Kincaid Grade, Gunning Fog, SMOG, ARI, Coleman-Liau, Dale-Chall, Linsear Write. Cite originals: Flesch (1948), Kincaid et al. (1975), Gunning (1952), Dale & Chall (1948).

**Use for: 6-feature readability bundle.**

### spaCy linguistic pipeline
https://spacy.io. POS tagging, dependency parsing, NER, morphological analysis. Honnibal et al. (2020), Zenodo. Outputs we need: POS-tag distributions (NOUN/VERB/ADJ/ADV ratios), dependency-arc-length distribution, parse-tree depth, NER frequency by type.

**Use for: 17-dim Universal POS ratio vector + avg dependency-arc length per token.** Robust to lexical ablation.

### Heylighen-Dewaele Formality Index (F-score)
Heylighen & Dewaele (1999, Free University of Brussels, http://pespmc1.vub.ac.be/Papers/Formality.pdf); published version (2002), *Foundations of Science* 7:293-340.

```
F = (noun% + adj% + prep% + article% − pronoun% − verb% − adverb% − interjection% + 100) / 2
```

Higher F = more formal/contextual. Real-estate listings vary enormously: luxury MLS prose (high noun-prep) vs casual SFR descriptions (high pronoun-verb). Expected to load strongly on neighborhood income.

**Use for: single interpretable formality scalar.**

### Burrows-Delta canonical features
Burrows (2002), "Delta: A Measure of Stylistic Difference and a Guide to Likely Authorship," *Literary and Linguistic Computing* 17(3):267-287. Top-N most-frequent-word z-scores, N=150-1000.

Eder, Rybicki & Kestemont (2016), "Stylometry with R," *R Journal* 8(1):107-121, package `stylo`.

Underwood (2019), *Distant Horizons*, U Chicago Press: layer MFW with POS-bigrams and function-word ratios.

**Use for: top-200 MFW z-scores restricted to function-word list (NLTK English stopword ∪ closed-class POS tokens) — textbook "pure stylometric" baseline.**

### Sentiment & surface
Hutto & Gilbert (2014), "VADER," *ICWSM 2014* — lexicon-based compound sentiment. Plus: comma frequency, semicolon frequency, exclamation density, mean sentence length, type-token ratio (Heaps/Herdan corrected).

### Recommended ~230-dim stylometric vector
- 6 readability indices
- 17 POS ratios
- 1 Heylighen-Dewaele F-score
- 2 syntactic (mean dep-arc, parse-depth)
- 4 lexical-richness (TTR, corrected TTR, mean sentence length, mean word length)
- 4 punctuation density
- 2 sentiment (VADER compound + variance)
- ~200 MFW function-word z-scores

Train logistic regression / GBM on this vector → predict zip-code-bin. **Predicted finding: stylometric-only accuracy ≈ 4-8× random** (vs full-embedding 26×). Proves style alone carries location signal but most encoding is lexical-semantic. Clean decomposition.

---

## A.2 Mutual Information: Word ↔ Zip-Code

### Right estimator for top-100 ranking
Yang & Pedersen (1997), "A Comparative Study on Feature Selection in Text Categorization," *ICML 1997* (https://dl.acm.org/doi/10.5555/645526.657137):

```
IG(w) = − Σ_c P(c) log P(c)
       + P(w)  Σ_c P(c|w)  log P(c|w)
       + P(w̄) Σ_c P(c|w̄) log P(c|w̄)
```

**Their finding: pure PMI penalizes rare words too heavily; IG/MI does not.**

Implemented as `sklearn.feature_selection.mutual_info_classif` (binary feature → discrete class reduces to plug-in MLE). Sanity-check via `sklearn.feature_extraction.text.CountVectorizer` + `chi2`.

### Recipe
1. Bin zip codes into K classes (K ≈ 50-200).
2. Document-term binary indicator matrix.
3. Compute IG per term against zip-bin label.
4. Rank, take top-100, manually code: {place name, architectural term, amenity, aesthetic/quality, formality marker, other}.

### Coding rubric citation
Hovy (2018), "The Social and the Neural Network," NLP+CSS Workshop — argues for transparent thematic coding of lexical findings.

### Bin-cardinality
Eirola & Lendasse (2014), "Gaussian Mixture Models for Mutual Information Estimation," *ESANN* — bias when one side has many classes. **Report both K=50 and K=200; show rankings stable.**

---

## A.3 Integrated Gradients on Transformer Embeddings

### Foundation
Sundararajan, Taly & Yan (2017), "Axiomatic Attribution for Deep Networks," *ICML 2017* (http://proceedings.mlr.press/v70/sundararajan17a.html). IG axioms: sensitivity, implementation invariance.

### Library
Captum (Kokhlikyan et al. 2020, https://arxiv.org/abs/2009.07896, https://captum.ai). Use `captum.attr.LayerIntegratedGradients` at the embedding layer (input is non-differentiable token ids).

### Pitfalls
1. **Baseline:** Mudrakarta et al. (2018), ACL — recommend `[PAD]`-embedding baseline. Zero baseline lands off-manifold.
2. **Sub-token aggregation:** "Brookline" → `["Brook", "##line"]`. Sum sub-token attributions within word boundary.
3. **Steps:** default 50; transformer embeddings need 100-300. Always report completeness delta.
4. **Position embeddings:** hold fixed at input values. Captum's `LayerIntegratedGradients` does this; rolling your own is error-prone.

### Working recipe
```python
from captum.attr import LayerIntegratedGradients
lig = LayerIntegratedGradients(forward_fn, model.bert.embeddings)
attributions, delta = lig.attribute(
    inputs=input_ids,
    baselines=baseline_ids,        # [PAD] tokens
    target=zip_class_id,
    n_steps=200,
    return_convergence_delta=True,
)
# Aggregate sub-tokens to words; rank by attribution magnitude.
```

**Cross-tabulate top-20 IG-attributed tokens against IG-MI top-100. Agreement validates both analyses.**

---

## A.4 Causal Mediation in Transformers — Skip for Now

Vig et al. (2020), "Investigating Gender Bias in Language Models Using Causal Mediation Analysis," *NeurIPS 2020*.

**Verdict: overkill.** Mediation answers "which internal units carry the location signal" — beyond what reviewers will demand. Reviewer's concern is "what about language encodes location" — answered by stylometric + MI + IG.

**Recommendation: cite Vig et al. (2020) in discussion as follow-up work; do not run.** Implementation would add 2-3 weeks.

---

## A.5 Recent Stylometric vs Semantic Decomposition Literature

### Central anchor
Hovy & Yang (2021), "The Importance of Modeling Social Factors of Language," *NAACL 2021* (https://aclanthology.org/2021.naacl-main.49/). Argues language carries social/demographic signal orthogonal to topical content.

**Cite as theoretical anchor for our decomposition. Their framing — demographic signal encoded at multiple layers (lexical, syntactic, stylistic) — is exactly what we operationalize for geography.**

### Other key citations
- Eisenstein, O'Connor, Smith & Xing (2010), "A Latent Variable Model for Geographic Lexical Variation," *EMNLP* (https://aclanthology.org/D10-1124/) — classic precedent.
- Bamman, Eisenstein & Schnoebelen (2014), "Gender identity and lexical variation in social media," *J. Sociolinguistics* 18(2):135-160.
- Pavalanathan & Eisenstein (2015), "Audience-modulated variation in online social media," *American Speech* 90(2):187-213.
- Demszky et al. (2019), "Analyzing Polarization in Social Media," *NAACL* — practical lexical-vs-topical decomposition recipe.

### Paper framing
"Following Hovy & Yang (2021), we decompose the location-encoding signal in property-description embeddings into a lexical-semantic component (operationalized via word-level mutual information, §X.X) and a stylometric component (operationalized via the feature set of Burrows 2002 / Eder et al. 2016, §X.X)."

---
