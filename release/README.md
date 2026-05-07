---
license: cc-by-4.0
language: en
size_categories:
  - 100K<n<1M
task_categories:
  - tabular-regression
  - feature-extraction
tags:
  - causal-inference
  - real-estate
  - spatial-confounding
  - urban-analytics
pretty_name: Causal Real Estate (Boston / NYC / SF)
---

# Causal Real Estate

Multi-city parcel-level dataset and text embeddings for studying spatial confounding
in real estate valuation, released alongside *Causal Disentanglement of Location and
Semantic Signals in Real Estate Valuation* (Yee & Crainic, 2026).

The dataset combines property assessments, census demographics, geocoded crime
incidents, and points-of-interest amenities for **556,636 parcels** across Boston,
New York (Manhattan + Brooklyn), and San Francisco. A held-out subset of ~3,000
parcels also ships with sentence-transformer embeddings of listing descriptions.

## Contents

```
data/
  boston/  parcels.parquet  embeddings_mpnet.parquet  embeddings_minilm.parquet
  nyc/     parcels.parquet  embeddings_mpnet.parquet  embeddings_minilm.parquet
  sf/      parcels.parquet  embeddings_mpnet.parquet  embeddings_minilm.parquet
splits/temporal_splits.csv     parcel_id → train / val / test (NYC, SF only)
scripts/make_panel.py          rebuilds these parquets from raw sources
scripts/fetch_descriptions.py  re-derives raw text from your own Redfin access
scripts/regenerate_embeddings.py
```

| City | Parcels | With description / embeddings | Median price |
|---|---:|---:|---:|
| Boston | 87,340 | 997 | $812,800 (assessed) |
| NYC (Manhattan + Brooklyn) | 242,062 | 990 | $995,000 (sale) |
| San Francisco | 227,677 | 995 | $1,283,160 (inferred sale) |

## Schema

`parcels.parquet` per city. Common 34-covariate confounder set used in the paper:

- **Property:** `bedrooms`, `lot_area_sqft`, `bldg_area_sqft`, `year_built`, `assessed_*`
- **Location:** `latitude`, `longitude`, `GEOID` (block group)
- **Census (ACS 5yr 2022):** `median_household_income`, `median_home_value`, `median_gross_rent`, `pct_white`, `pct_black`, `pct_asian`, `pct_hispanic`, `pct_bachelors`, `labor_force_participation`, `pct_under_25`, `pct_over_60`
- **Crime (500m radius):** `crime_violent`, `crime_property`, `crime_quality_of_life`, `crime_other`, `crime_total`
- **Amenities (counts and densities):** `amenity_food_dining`, `amenity_retail`, `amenity_services`, `amenity_recreation`, `amenity_transportation`, `amenity_education`, `amenity_total`, `amenity_diversity`, plus `*_density` versions
- **Micro-geography:** `dist_park_m`, `dist_transit_m`, `dist_school_m`, `dist_restaurant_m`, `dist_retail_m`, `dist_medical_m`

City-specific extras (NYC sale dates, SF assessor metadata) are documented in `DATASHEET.md`.

`embeddings_*.parquet` per city: `latitude`, `longitude`, `zip`, `price`, plus `emb_0`..`emb_N`
where N=767 (mpnet, 768-dim) or N=383 (MiniLM, 384-dim).

## Loading

```python
from datasets import load_dataset
ds = load_dataset("jcrainic2/causal-real-estate", "nyc-parcels")
# or directly:
import pandas as pd
parcels = pd.read_parquet("hf://datasets/jcrainic2/causal-real-estate/data/nyc/parcels.parquet")
```

## What is *not* included, and why

**Raw listing descriptions are not redistributed.** The descriptions used to
produce the embeddings were retrieved from Redfin, whose Terms of Service prohibit
redistribution; the remarks themselves carry MLS copyright. To reproduce the
text-side analysis in full, run `scripts/fetch_descriptions.py` against your own
Redfin access. The pre-computed embeddings shipped here are derivative
representations and are released under CC BY 4.0.

**Street addresses are not included** even where they appear in source assessor
rolls (notably SF). The paper's analysis uses lat/lon at parcel-centroid
precision; addresses are unnecessary for replication.

## License

- **Structured features and embeddings:** [CC BY 4.0](LICENSE)
- **Code (`scripts/`):** MIT
- **Source attributions:** see `DATASHEET.md` § Source Datasets

If you use this dataset, please cite:

```bibtex
@article{yee2026causal,
  title={Causal Disentanglement of Location and Semantic Signals in Real Estate Valuation},
  author={Yee, Brandon and Crainic, Jacob},
  journal={Journal of Business and Economic Statistics},
  year={2026},
  note={Under review}
}
```

## Reproducing the paper

The companion code repository contains the full pipeline (`run_pipeline.py`) including DML, doubly-robust estimation,
adversarial deconfounding, and the confounder escalation test. This dataset is
the input.

## Contact

`{b.yee, j.crainic}@ycrg-labs.org`
