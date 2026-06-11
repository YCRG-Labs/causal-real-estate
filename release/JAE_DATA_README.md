# Data and code for the JAE Data Archive

Prepared to the Journal of Applied Econometrics replication policy, which since
1994 requires a complete set of non-confidential data, and for confidential data
a readme describing the source in enough detail that others can apply to obtain
access. Listing text and asking prices are scraped under a commercial platform's
terms and are confidential in that sense; everything derived from them, and all
public-record data, is deposited here.

## What is deposited (redistributable)

- `data_12/<city>.parquet` — for each of the twelve markets, one row per
  anonymized listing id with: the 768-dimensional sentence-transformer embedding
  of the description (a derived representation, not the text), structured property
  attributes (beds, baths, sqft, year built, lot size, property type), parcel
  latitude/longitude and ZIP, the recovered sale year/quarter, and, for the four
  markets that publish recorded transactions (Philadelphia, Chicago, DC, NYC), the
  recorded sale price and date.
- `data_12/MANIFEST.md` — per-market row counts and sale-price coverage.
- All analysis code (`data/scripts/`), the scraper, and the computational
  verification suite (`verification/`, reproducible via `make verify`).

## What is NOT deposited, and how to obtain it (confidential source)

- **Listing descriptions and asking (listing) prices.** Source: sold-listing
  pages on Redfin (`redfin.com`), retrieved via the GIS/listing endpoints in
  `data/scripts/scrape_descriptions.py` and `scrape_redfin_async.py`. These are
  scraped content subject to Redfin's Terms of Service and MLS copyright and are
  not redistributed. A researcher can reconstruct the corpus by running the
  provided collection code against Redfin (the embeddings deposited here are the
  derived representation used in every regression, so the embedding-based results
  replicate without re-scraping the raw text).
- **Recorded sale prices** for the four disclosure markets come from public county
  assessor/recorder open data and are obtainable directly: Philadelphia OPA
  (`phl.carto.com`), Cook County Assessor (`datacatalog.cookcountyil.gov`), DC CAMA
  (`maps2.dcgis.dc.gov`), NYC DOF Rolling Sales. The fetchers are in
  `data/scripts/download_sales.py`; the join to listings in
  `join_sales_to_listings.py`.

## Confounders regenerated from public sources

Census (ACS block-group), crime (municipal open data), and amenity (OpenStreetMap)
covariates are not shipped as files; they are regenerated from their public
sources by `attach_census.py`, `attach_crime.py`, and `attach_amenities.py`.

## Reproducing the results

The embedding-based estimates (the headline analysis) reproduce from the deposited
`data_12/` files plus the code, with no access to Redfin required. The asking-price
outcome requires reconstructing the listing prices via the scraper; the recorded
sale-price validity check reproduces from the deposited public sale prices.
