# Datasheet: Causal Real Estate

Following the schema of Gebru et al., *Datasheets for Datasets* (CACM 2021).

## Motivation

**Purpose.** Released to support and enable independent replication of *Causal
Disentanglement of Location and Semantic Signals in Real Estate Valuation* (Yee
& Crainic, 2026), which evaluates whether semantic features extracted from
property descriptions add predictive value beyond geographic location, or
whether observed gains arise from spatial confounding.

**Created by.** Yee Collins Research Group, Management Sciences Lab. Funded
through internal research budget; no external sponsor.

## Composition

**Instances.** 556,636 residential parcels across three U.S. metropolitan areas:

- Boston (87,340 parcels, Boston Open Data + Assessing Department)
- New York Manhattan & Brooklyn (242,062 parcels, NYC PLUTO + DOF Rolling Sales)
- San Francisco (227,677 parcels, DataSF + Assessor Historical Secured Roll)

A subset of 2,982 parcels (~995 per city) additionally carries a sentence-transformer
embedding of a Redfin listing description; raw descriptions themselves are not
redistributed (see Distribution).

**Features per instance.** 34-covariate confounder set covering property
characteristics, parcel location (lat/lon), block-group census demographics
(ACS 5-Year 2022), 500m-radius crime counts (municipal police open data), and
points-of-interest amenities (OpenStreetMap, by category). Full list in `README.md`.

**Labels.** Sale price (NYC, SF) or assessed total value (Boston). NYC sale
prices come from DOF Rolling Sales 2018–2024 filtered to $50k–$20M. SF prices
are inferred via Proposition 13: a parcel is treated as sold when
`current_sales_date` changes between fiscal years and assessed value shifts
>5%, with the post-sale assessed value as the price proxy.

**Sampling.** All available residential parcels in each jurisdiction's open
data, restricted to land use codes 1–3 (single-family, two-family, walk-up
multi-family) where applicable.

**Missingness.** ACS spatial-join match rates: Boston 99.4%, NYC 100%, SF 100%.
Crime and amenity counts are zero where no incidents/POIs fall within radius;
this is recorded data, not missingness.

## Collection

**Property assessments and parcels.** Public open-data portals: Boston Open
Data, NYC Open Data (PLUTO + DOF), DataSF. Vintages match each city's most
recent published roll at extraction (FY2026 Boston, 2024 PLUTO, FY2024 SF).

**Census.** U.S. Census Bureau ACS 5-Year Estimates 2022, queried via
`api.census.gov`. Spatial join against TIGER/Line block group polygons.

**Crime.** Municipal police open data: Boston PD Crime Incident Reports
(2020–2023), NYPD Complaint Data YTD, SF PD Incident Reports (2018–present).
Cross-city offense crosswalk maps each agency's codes into violent, property,
and quality-of-life categories.

**Amenities.** OpenStreetMap via Overpass API, queried by city bounding box and
bucketed into six categories (food_dining, retail, services, recreation,
transportation, education).

**Listing descriptions.** Retrieved from Redfin's public listing pages via the
`stingray/api/gis` search endpoint and individual listing HTML
(`scripts/fetch_descriptions.py`). Spatial join of listing lat/lon to parcels
within 50m. **Not redistributed** in this dataset.

## Preprocessing

- Outliers removed: prices below $50k or above $20M (7,814 NYC, 968 SF).
- SF sale inference: see Composition § Labels.
- Embeddings: `sentence-transformers/all-mpnet-base-v2` (768-dim) and
  `all-MiniLM-L6-v2` (384-dim), mean-pooled across cleaned description tokens.
- Splits: temporal 70/15/15 by sale date for NYC and SF; Boston has no sale
  date, so all Boston parcels are training-only.

## Uses

**Intended.** Replication of the paper's causal-inference analysis; benchmarking
spatial-confounding diagnostics; teaching examples for DML, doubly-robust
estimation, and adversarial deconfounding on a real dataset where the SCM is
well-justified.

**Out of scope / discouraged.**

- *Production AVMs.* The labels include assessed values (Boston) and inferred
  sales (SF) which are not directly comparable to transacted prices.
- *Re-identification.* Lat/lon are at parcel-centroid precision. Combined with
  external data this could re-identify owner-occupants. Do not redistribute
  joined to identifying records.
- *Transfer to other markets.* The paper's results are specific to these three
  cities and do not generalize without re-fitting.

## Distribution

**Format.** Apache Parquet (Snappy compression). One `parcels.parquet` per
city plus two embedding parquets per city.

**Where.** Hugging Face Hub (`jcrainic2/causal-real-estate`) and Kaggle mirror.
DOI minted via Zenodo on first stable release.

**License.**

- Derived structured features and embeddings: CC BY 4.0.
- Code in `scripts/`: MIT.
- Underlying source datasets remain governed by their original terms (city
  open-data portals are public-domain or CC0; ACS is public-domain; OSM is
  ODbL — note that derived counts/densities are aggregations, not redistribution
  of OSM features).
- Raw Redfin listing descriptions are **not** redistributed. Retrieval against
  Redfin remains subject to Redfin's Terms of Service and MLS copyright.

## Maintenance

Maintained by the authors. Issues and corrections via the Hugging Face dataset
repo's discussion tab. No scheduled refresh; updates pinned to
paper revisions.
