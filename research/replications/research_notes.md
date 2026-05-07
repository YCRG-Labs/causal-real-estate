# Replicating Baur / Shen / Zhang — Research Dossier (Item 3)

For Item 3: replicate the exact pipelines from the three papers we're rebutting; show their reported gains evaporate under our causal tests.

---

## ⚠ Important verification finding

The Baur paper title in my original brief ("Automated Valuation Modeling using Multi-modal Inputs") **does not match the actual title**. The actual paper is:

> Baur, K., Rosenfelder, M. & Lutz, B. (2023). "Automated real estate valuation with machine learning models using **property descriptions**." *Expert Systems with Applications* 213, 119147.

This is **text-only ML, not multi-modal** image+text fusion. If we want a true multi-modal AVM paper, the closer hit is "Artificial Intelligence and Real Estate Valuation: The Design and Implementation of a Multimodal Model", *Information* 16(12):1049 (MDPI, 2025). But Baur et al. 2023 is the right rebuttal target — it's a leading text-only AVM paper and our SCM directly addresses it.

---

## B1. Shen & Ross (2021) — *J. Urban Economics*

### Citation
Shen, L. & Ross, S. L. (2021). "Information value of property description: A Machine learning approach." *Journal of Urban Economics* 122, 103299. doi:10.1016/j.jue.2020.103299.
- SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3281221
- ScienceDirect: https://www.sciencedirect.com/science/article/abs/pii/S009411902030070X
- Working paper PDF (UConn): https://finance.business.uconn.edu/wp-content/uploads/sites/723/2019/02/Information-Value-Of-Property-Description-A-Machine-Learning-Approach.pdf

### Method
Unsupervised NLP (LDA-style topic / doc2vec-style embedding) → one-dimensional **"uniqueness" score** (semantic deviation of listing from neighbors). Plugged into hedonic OLS and repeat-sales models. **Pre-BERT NLP** — not deep transformer pipeline.

### Dataset
~40,000 single-family homes, Atlanta, GA, 2010–2017 (advertised + transacted via MLS).

### Key reported result
- One-SD increase in uniqueness ⇒ **+15% sale price** (hedonic) and **+10%** (repeat sales)
- Annual hedonic price indices ignoring uniqueness overstate post-2009 price recovery by **10–23%**
- Headline is **coefficient magnitude on uniqueness variable**, not ΔR²

### Code
**No public release.** SSRN posting text-only. No GitHub on Lily Shen's page (Clemson: https://sites.google.com/g.clemson.edu/lily-shen) or Stephen Ross's (UConn: https://econ.uconn.edu/person/stephen-ross/). Reproducible from MLS data + verbal recipe but not via released code.

### Engineering complexity
**~1–2 person-weeks** to re-implement on our data. NLP is shallow (TF-IDF / topic-model uniqueness, not transformers); most time in matching MLS variable conventions and sample restrictions.

---

## B2. Baur, Rosenfelder & Lutz (2023) — *Expert Systems with Applications*

### Citation
Baur, K., Rosenfelder, M. & Lutz, B. (2023). "Automated real estate valuation with machine learning models using property descriptions." *Expert Systems with Applications* 213, 119147. doi:10.1016/j.eswa.2022.119147.
- ACM mirror: https://dl.acm.org/doi/10.1016/j.eswa.2022.119147

### Method
Compares Random Forest / Gradient Boosting / DNN baselines on structured features only versus same models augmented with **BERT embeddings** of German/English listing description. Best configuration: **Gradient Boosting + BERT** for purchase prices.

### Dataset
- Berlin (German listings, ImmobilienScout24-style)
- Los Angeles (English)
- Both rental and purchase splits

### Key reported result
- Adding BERT-embedded description features to GBM **reduces prediction error** on purchase prices vs structured-only baselines
- Reported in MAE / MAPE terms; lowest error from GBM+BERT
- Precise ΔMAPE in their Table 4 (typically a few percentage points)
- Abstract: win is statistically meaningful but modest

### Code
**No public GitHub repository advertised** in paper or on authors' pages. Data are proprietary scrapes; replication requires independent scraping plus re-implementation.

### Engineering complexity
**~2–3 person-weeks** to re-implement on our data:
- BERT featurization is well-trodden (HuggingFace `bert-base-uncased` or `bert-base-german-cased`)
- GBM baselines are LightGBM/XGBoost stock
- Only fiddle: matched purchase- and rental-price splits + reproducing their feature taxonomy

If we already have DistilBERT pipeline assets, drops to ~1 person-week.

---

## B3. Zhang et al. (2021) — *KDD*

### Citation
Zhang, W., Liu, H., Zha, L., Zhu, H., Liu, J., Dou, D. & Xiong, H. (2021). "MugRep: A Multi-Task Hierarchical Graph Representation Learning Framework for Real Estate Appraisal." *KDD '21*, pp. 3937-3947.
- arXiv: https://arxiv.org/abs/2107.05180
- ar5iv HTML: https://ar5iv.labs.arxiv.org/html/2107.05180
- USTC PDF: https://bigdata.ustc.edu.cn/paper_pdf/2021/Weijia-Zhang-KDD.pdf

### Method
Three components on top of structured + multi-source urban features:
1. **Evolving real-estate transaction graph** with event graph convolution module (spatiotemporal transaction dependencies)
2. **Hierarchical heterogeneous community graph** convolution module (community–community correlations)
3. **Urban-district-partitioned multi-task** head (district-conditional value estimates)

Baselines: HA, LR, SVR, GBRT, DNN, PDVM (Peer-Dependence Valuation Model, AAAI 2019). All ANN methods (DNN, PDVM, MugRep) outperform non-ANN.

### Dataset
- Beijing (~185k transactions, 6,267 communities)
- Chengdu (smaller; exact size in Table 1)
- Multi-source urban data: POIs, mobility traces, demographics
- **Proprietary** (Lianjia/Anjuke-style agency feed); not released

### Key reported result
- MAE, MAPE, RMSE on test transactions
- Table 2: MugRep attains lower MAPE than PDVM on both cities
- Single-digit absolute MAPE; improvements ~0.5–1.5 percentage points over PDVM

### Code
**No public release.** Searched corresponding author's GitHub (https://github.com/willzhang3) — only repos for *other* papers (SHARE parking, MASTER EV charging). Hao Liu's group page (https://raymondhliu.github.io/publications/) lists paper without code link. No supplementary at ACM DL entry.

### Engineering complexity
**~6–10 person-weeks** to re-implement from scratch. Bottleneck of the three replications. Sub-tasks:
- Multi-source urban feature engineering (POIs, mobility, demographics) — substitute OpenStreetMap POI counts + ACS demographics (~1.5 weeks)
- Evolving transaction graph + event graph convolution (custom GNN) — ~2 weeks in PyTorch Geometric
- Hierarchical heterogeneous community graph convolution — ~2 weeks (heterogeneous GNN, leverage DGL `HeteroGraphConv`)
- District-partitioned multi-task head + training loop — ~1 week
- Hyperparameter search to match reported MAPE — ~1–2 weeks

Because data are proprietary and code is closed, replication will be **from-spec reimplementation**; expect to reproduce qualitative gain (graph beats GBRT) but not exact numbers.

---

## B4. Reproducibility Summary

| Paper | Code | Data | Re-impl on our data | Headline metric to reproduce |
|---|---|---|---|---|
| Shen & Ross 2021 (JUE) | No | No (MLS, proprietary) | **1–2 weeks** | +15% price coefficient on uniqueness; price-index distortion |
| Baur et al. 2023 (ESWA) | No | No (Berlin/LA scrapes) | **1–3 weeks** | ΔMAPE of GBM+BERT vs GBM-only on purchase prices |
| Zhang et al. 2021 (KDD MugRep) | **No** | No (Beijing/Chengdu agency) | **6–10 weeks** | ΔMAPE of MugRep vs PDVM on both cities |

### Order of attack
**Shen → Baur → MugRep**, mirroring engineering cost. All three reproducible from spec; none ship code. Once reproduced, applying our DML + NCO/NCE battery follows identical template — marginal cost per paper after the first is mostly data-prep.

---

## Verification Status (all three papers confirmed real)
- ✓ Shen JUE 2021
- ✓ Baur ESWA 2023 (article 119147)
- ✓ Zhang KDD '21 (arXiv:2107.05180)

**Caveat:** original brief had wrong title for Baur. Correct title is "...using property descriptions." Paper is text-only, not multi-modal.
