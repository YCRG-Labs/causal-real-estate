---
license: cc-by-4.0
language: en
size_categories:
  - 10K<n<100K
task_categories:
  - tabular-regression
  - feature-extraction
tags:
  - causal-inference
  - real-estate
  - spatial-confounding
  - text-as-data
  - urban-analytics
pretty_name: Listing Language & Home Valuation (12 U.S. markets)
---

# Listing Language & Home Valuation

Listing-level text embeddings and structured covariates for studying whether the
language of a property description carries genuine information about a home or
merely relabels its location. Released alongside the working paper *Does Listing
Language Add Value Beyond Location? Deconfounding Text and Geography in Automated
Home Valuation* (Crainic, 2026).

The package covers **twelve U.S. metropolitan markets** and roughly **69,000
listings** (69,173 in the paper's analysis sample). Each listing ships with a
sentence-transformer embedding of its description, structured property attributes,
parcel coordinates, and a recovered sale date. For the four markets that publish
recorded transactions (Philadelphia, Chicago, DC, New York), the realized sale
price is included as well.

## Contents

```
data_12/
  <city>.parquet     one row per listing, for each of the twelve markets
  MANIFEST.md        authoritative per-market row counts and sale-price coverage
scripts/             collection + pipeline code (rebuilds the corpus from source)
DATASHEET.md         full provenance, collection process, and limitations
JAE_DATA_README.md   deposit notes for the Journal of Applied Econometrics archive
```

Markets: Boston, New York, San Francisco, Washington DC, Philadelphia, Chicago,
Seattle, Denver, Atlanta, Portland, Phoenix, Dallas.

| Market | Listings | Recorded sale price |
|---|---:|---|
| Dallas | 8,009 | reconstruct via scraper |
| Phoenix | 7,089 | reconstruct via scraper |
| Philadelphia | 7,033 | ✓ ~94% |
| Atlanta | 6,139 | reconstruct via scraper |
| Chicago | 5,606 | ✓ ~86% |
| Denver | 5,283 | reconstruct via scraper |
| Portland | 4,347 | reconstruct via scraper |
| Washington DC | 4,028 | ✓ ~53% |
| Seattle | 2,887 | reconstruct via scraper |
| Boston | 2,630 | reconstruct via scraper |
| New York | 15,240 | ✓ ~42% |
| San Francisco | 987 | reconstruct via scraper |

Counts are as built; see `data_12/MANIFEST.md` for the authoritative figures.

## Schema

`data_12/<city>.parquet`, one row per listing:

- **Identifier:** `id` (stable anonymized listing key)
- **Location:** `latitude`, `longitude`, `zip` (parcel-centroid precision)
- **Property:** `beds`, `baths`, `sqft`, `year_built`, `lot_size`, `property_type`
- **Timing:** `sale_year`, `sale_quarter` (recovered from the listing page)
- **Outcome (4 markets only):** `sale_price`, `sale_date` — recorded transactions
  for Philadelphia, Chicago, DC, and New York
- **Text:** `emb_0` … `emb_767` — 768-dimensional sentence-transformer (mpnet)
  embedding of the listing description

The census, crime, and amenity confounders used in the paper are **not** shipped
here; they are regenerated from their public sources by the pipeline scripts
(`attach_census.py`, `attach_crime.py`, `attach_amenities.py`).

## Loading

```python
import pandas as pd
nyc = pd.read_parquet("hf://datasets/jcrainic2/causal-real-estate/data_12/nyc.parquet")
```

## What is *not* included, and why

**Raw listing descriptions and asking prices are not redistributed.** Both are
scraped content: descriptions carry MLS copyright and asking prices are retrieved
under Redfin's Terms of Service, which prohibit redistribution. The embeddings
shipped here are derivative representations released under CC BY 4.0. To rebuild the
raw text and asking price from your own access, run the collection code in
`scripts/`.

**Street addresses are not included.** The analysis uses lat/lon at parcel-centroid
precision; addresses are unnecessary for replication.

## License

- **Structured features and embeddings:** [CC BY 4.0](LICENSE)
- **Code (`scripts/`):** MIT
- **Recorded sale prices:** public county assessor/recorder open data
- **Source attributions:** see `DATASHEET.md` § Source Datasets

## Contact

Questions and corrections via the
[GitHub repository](https://github.com/human-vc/causal-real-estate) issues.
