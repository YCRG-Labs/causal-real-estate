# Does Listing Language Add Value Beyond Location?

**Deconfounding text and geography in automated home valuation.**

Replication code and data for the working paper *Does Listing Language Add Value
Beyond Location? Deconfounding Text and Geography in Automated Home Valuation*
(Jacob Crainic, 2026).

---

## The question

Automated valuation models increasingly read a listing's prose alongside its
structured attributes, and the language reliably sharpens their price predictions.
But better prediction cannot say *why*: does the text carry genuine information
about a property, or does it merely relabel the neighborhood an agent's description
inevitably encodes? A model that prices laundered geography offers nothing beyond
location and inherits its biases; one that prices semantic content adds real
information. Predictive accuracy alone cannot tell the two apart.

This project puts the question to a design-based test. We summarize each description
by the leading direction of its sentence embedding, treat that direction as a
regressor in a partially linear hedonic model, and measure how much of its effect on
price survives once location is removed from the picture.

## What we find

Across **twelve metropolitan markets** and **69,173 deduplicated listings**, the
price effect of listing language is almost unmoved by spatial adjustment in eleven of
them. The exception is New York, whose text is the most geographic of the twelve, so
the exception illuminates the mechanism rather than undercutting it. The effect does
not depend on the encoder, it is economically sizable (a one-standard-deviation move
along the text direction is worth roughly 8–17% of price, several times the value of
an added bedroom), and in the four markets that publish recorded transactions it
replicates on realized sale prices. In most markets, a listing's language is priced
for its content, not its geography.

## Method, in brief

| Step | Tool |
|---|---|
| Estimand | partially linear hedonic model; leading embedding direction as treatment |
| Effect | Robinson partialling-out + double machine learning (cross-fitted nuisances) |
| Headline test | Gelbach decomposition — text held fixed, geography moved on/off the controls |
| Spatial control | thin-plate regression spline over coordinates (robustness) |
| Does text encode location? | concept-erasure diagnostic (held-out probe + projection) |
| Inference | generated-regressor correction (Pagan) + full-pipeline bootstrap |
| Robustness | Cinelli–Hazlett sensitivity bounds; recorded-sale-price validity check |

## Repository layout

```
paper/          LaTeX source and compiled PDF (causalrealestate.tex)
data/scripts/   acquisition, confounder construction, and the replication pipeline
  replications/   the estimators: hedonic, DML, Gelbach, sensitivity, meta-analysis
release/        ToS-compliant data release + datasheet (Hugging Face card)
verification/   computational verification artifact for every analytic claim
research/        design notes and the revision report
results/         per-market estimates and intermediate outputs
```

## Reproducing the analysis

```bash
pip install -r requirements.txt
```

The released package ships the derived embeddings, structured covariates, and
public-record sale prices, which are enough to rerun the estimators end to end:

```bash
# example: the headline deconfounding decomposition, all twelve markets
python3 data/scripts/replications/leace_price_decomposition.py --all_12 --fast

# sale-price validity check (the four markets with public transactions)
python3 data/scripts/compare_saleprice.py
```

See `verification/README.md` for the claim-to-script map that checks each result.

## Data availability

The structured covariates come from public sources — county assessor and recorder
open data, the U.S. Census Bureau (ACS), and OpenStreetMap — and are redistributed
here, together with the sentence-transformer embeddings derived from listing text and
the recorded sale prices used in the validity check.

The **raw listing descriptions and asking prices are not redistributed**: they are
scraped under a commercial platform's terms of service. The collection code that
reconstructs them from your own access is included, so the corpus can be rebuilt
rather than only read. See `release/DATASHEET.md` and `release/JAE_DATA_README.md`
for the full provenance and the confidential-source path.

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)). Released data
follow the licenses of their public sources, documented in `release/DATASHEET.md`.
