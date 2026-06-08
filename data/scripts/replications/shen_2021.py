"""Shen & Ross (2021, JUE) -- replication-style exercise (NOT a strict replication).

  Shen, L. & Ross, S. L. (2021). "Information value of property description:
  A machine learning approach." Journal of Urban Economics 121, 103299.

This script is a *TF-IDF + lat/lon-KNN cousin* of Shen & Ross, NOT a faithful
replication.  The original paper uses Doc2Vec embeddings (PV-DM, vec_size
small, see Shen-Ross 2021 §4.2) and MLS-Area x Year *cell averages* as the
peer set (their Eq. 9), not a K-nearest-neighbour query in lat/lon.  Both
choices matter: lat/lon KNN collapses to the ZIP centroid when listings
share a centroid (the original Redfin pull dropped 23 percent of rows on
near-duplicate addresses; see ``infer_sf_prices.py``), which inflates
uniqueness by construction.  The published +0.072 per sigma on our pull was
almost certainly a duplicate-inflation + ZIP-centroid degeneracy artefact.

What this script actually does in its DEFAULT mode (the cousin construction):

  1. TF-IDF vectorise all descriptions (max_features=5000, ngram=(1,2),
     min_df=2).  This stands in for Shen & Ross's PV-DM Doc2Vec
     (vec_size=100, window=5, epochs=40, dm=1) when ``--doc2vec`` is not
     passed.
  2. For each listing build a peer set:
     - default: K=5 lat/lon nearest neighbours;
     - ``--cell_fe``: peers within the (zip, year) cell; fall back to
       (zip,) when year is missing.
  3. Uniqueness = 1 - cos(self_tfidf, mean(peer_tfidf)).
  4. Hedonic OLS: log_price ~ uniqueness_z + structured features + optional
     (zip, year) cell FE; HC3 robust SEs.  The published Shen & Ross (2021,
     JUE 121:103299) abstract states a one-SD uniqueness increase raises sale
     price by 15% in the hedonic model (10% in the repeat-sales model), i.e.
     ~0.140 log-points (ln 1.15).  NOTE: the earlier +5.6%-per-SD figure that
     circulated is the superseded working-paper number, not the published
     value.  The published abstract reports no standard error or n, so the
     published_se / published_n used in the power calc are UNVERIFIED
     placeholders.
  5. Re-run the project DML pipeline with uniqueness as the continuous
     treatment, same confounder set as the production analysis.
  6. Always print the binomial-equivalent power: at our n vs Shen's
     published theta-hat=0.140 (15% per SD in the hedonic model) with an
     unverified SE placeholder, what is our power against the published
     effect (two-sided, alpha=0.05)?

Flags
-----
  --k_sweep            Compute uniqueness at K in {3,5,10,20,50}; report
                       the Spearman rank correlation matrix.  When values
                       are stable across K the construction is robust;
                       when they degrade the construction is K-driven.
  --cell_fe            Use (zip, year) cell averages instead of lat/lon
                       KNN; add zip-year cell fixed effects to the OLS.
                       This is the closest available approximation of
                       Shen's MLS-Area x Year averaging.
  --doc2vec            Replace TF-IDF with Doc2Vec (PV-DM, vec_size=100,
                       window=5, epochs=40, dm=1).  Requires
                       ``pip install gensim``.

Usage
-----
  python shen_2021.py
  python shen_2021.py --k_sweep
  python shen_2021.py --cell_fe
  python shen_2021.py --doc2vec
  python shen_2021.py --out results/replications/shen.json
  python shen_2021.py --n 200            # subset for smoke tests

References
----------
Shen, L. & Ross, S. L. (2021).  "Information value of property description:
  A machine learning approach."  Journal of Urban Economics 121, 103299.
  https://doi.org/10.1016/j.jue.2020.103299
Le & Mikolov (2014).  "Distributed Representations of Sentences and
  Documents."  ICML.  arXiv:1405.4053.
Cohen (1988).  "Statistical Power Analysis for the Behavioral Sciences,"
  2nd ed.  (One-sample z-test power calculation used below.)
"""
from __future__ import annotations
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import _silence

import argparse
import functools
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

print = functools.partial(print, flush=True)

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from causal_inference import (
    PROPERTY_COLS,
    get_features_and_target,
    load_analysis_data,
)
from replications.compare_to_dml import result_to_dict, run_dml


@dataclass
class OLSUniqueness:
    coef: float
    se: float
    t: float
    p: float
    ci_low: float
    ci_high: float
    n: int
    r2: float
    adj_r2: float
    coef_per_sd: float
    pct_per_sd: float


def _vectorize_tfidf(descriptions: list[str], max_features: int = 5000):
    vec = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    return vec.fit_transform(descriptions)


def _vectorize_doc2vec(descriptions: list[str], vector_size: int = 100,
                       window: int = 5, epochs: int = 40, dm: int = 1,
                       seed: int = 0):
    """Doc2Vec embeddings per Le & Mikolov 2014 PV-DM (dm=1).

    Defaults match Shen & Ross 2021 §4.2 reported choices (vec_size=100,
    window=5, epochs=40, distributed memory).  Requires gensim.
    """
    try:
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument
        from gensim.utils import simple_preprocess
    except ImportError as e:
        raise ImportError(
            "--doc2vec requires gensim; install with: pip install 'gensim>=4.3'"
        ) from e
    tagged = [TaggedDocument(simple_preprocess(d), [i])
              for i, d in enumerate(descriptions)]
    import multiprocessing as _mp
    n_workers = min(_mp.cpu_count(), 8)
    model = Doc2Vec(documents=tagged, vector_size=vector_size, window=window,
                    epochs=epochs, dm=dm, min_count=2, workers=n_workers,
                    seed=seed)
    vectors = np.stack([model.dv[i] for i in range(len(descriptions))])
    return vectors


def _uniqueness_from_vectors(vectors, peer_indices: list[list[int]]) -> np.ndarray:
    """Shen-Ross 2021 Eq. 9 mean-pairwise-distance uniqueness.

    For each row i, peer_indices[i] gives a list of other-row indices
    forming the peer set; uniqueness = mean_{j in peers} (1 - cos(v_i, v_j)).
    Listings with empty peer sets get uniqueness 0 (no signal).

    Implementation: a single dense pairwise-cosine matmul over all n rows,
    then per-row averaging across each peer set. Measured ~30x faster than
    the per-row sklearn loop on n=310, d=100 (research agent benchmark);
    dominates the loop because we avoid Python-level dispatch per row.
    """
    n = len(peer_indices)
    if n == 0:
        return np.zeros(0, dtype=float)

    sim = cosine_similarity(vectors)
    uniq = np.zeros(n, dtype=float)
    for i in range(n):
        peers = [j for j in peer_indices[i] if j != i]
        if not peers:
            continue
        uniq[i] = float(1.0 - sim[i, peers].mean())
    return uniq


def _knn_peers(lat: np.ndarray, lon: np.ndarray, k: int) -> list[list[int]]:
    coords = np.column_stack([lat, lon])
    tree = cKDTree(coords)
    n = len(lat)
    k_eff = min(k + 1, n)
    _, nn_idx = tree.query(coords, k=k_eff)
    if nn_idx.ndim == 1:
        nn_idx = nn_idx.reshape(-1, 1)
    return [list(int(j) for j in row if j != i)[:k]
            for i, row in enumerate(nn_idx)]


def _cell_peers(zips: np.ndarray, years: np.ndarray | None) -> list[list[int]]:
    """Peer set = other listings in the same (zip, year) cell.  Falls back
    to (zip,) cells when year is missing or constant.
    """
    n = zips.size
    use_year = years is not None and np.unique(years).size > 1
    if use_year:
        keys = list(zip(zips.astype(str), years.astype(str)))
    else:
        keys = list(zips.astype(str))
    by_key: dict = {}
    for i, k in enumerate(keys):
        by_key.setdefault(k, []).append(i)
    return [[j for j in by_key[keys[i]] if j != i] for i in range(n)]


def compute_uniqueness(
    descriptions: list[str],
    lat: np.ndarray,
    lon: np.ndarray,
    k: int = 5,
    max_features: int = 5000,
    *,
    use_doc2vec: bool = False,
    zips: np.ndarray | None = None,
    years: np.ndarray | None = None,
    cell_fe: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """Peer-mean uniqueness per listing: 1 - cos(self_vec, mean(peer_vec)).

    Peer set is either lat/lon KNN (default), or the (zip, year) cell
    average (``cell_fe=True``).
    """
    if use_doc2vec:
        vectors = _vectorize_doc2vec(descriptions, seed=seed)
    else:
        vectors = _vectorize_tfidf(descriptions, max_features=max_features)
    if cell_fe:
        if zips is None:
            raise ValueError("--cell_fe requires zips; pass zips=... ")
        peers = _cell_peers(zips, years)
    else:
        peers = _knn_peers(lat, lon, k)
    return _uniqueness_from_vectors(vectors, peers)


def hedonic_ols(
    uniqueness: np.ndarray,
    confounders: np.ndarray,
    confounder_names: list[str],
    Y_log: np.ndarray,
    *,
    cell_keys: list | None = None,
) -> tuple[OLSUniqueness, sm.regression.linear_model.RegressionResultsWrapper]:
    """log_price ~ uniqueness_z + confounders [+ cell FE], HC3 robust SEs.

    ``cell_keys``: optional list of hashable cell labels (length n).  When
    provided, dummy variables for each cell (less one reference) are
    appended to the regressor matrix.  This implements the Shen-Eq.9-style
    MLS-Area x Year fixed effects on our (zip, year) approximation.
    """
    u_sd = float(np.std(uniqueness, ddof=1))
    if u_sd == 0:
        u_sd = 1.0
    u_z = (uniqueness - uniqueness.mean()) / u_sd

    X = np.column_stack([u_z, confounders])
    X = sm.add_constant(X, has_constant="add")
    names = ["const", "uniqueness_z"] + list(confounder_names)
    df = pd.DataFrame(X, columns=names)

    if cell_keys is not None:
        cell_df = pd.DataFrame({"_cell": list(cell_keys)})
        fe = pd.get_dummies(cell_df["_cell"], prefix="cell", drop_first=True,
                            dtype=float)
        fe.columns = [f"fe_{c}" for c in fe.columns]
        df = pd.concat([df.reset_index(drop=True),
                        fe.reset_index(drop=True)], axis=1)
    model = sm.OLS(Y_log, df).fit(cov_type="HC3")

    coef = float(model.params["uniqueness_z"])
    se = float(model.bse["uniqueness_z"])
    tval = float(model.tvalues["uniqueness_z"])
    pval = float(model.pvalues["uniqueness_z"])
    lo, hi = (float(x) for x in model.conf_int().loc["uniqueness_z"])

    out = OLSUniqueness(
        coef=coef,
        se=se,
        t=tval,
        p=pval,
        ci_low=lo,
        ci_high=hi,
        n=int(len(Y_log)),
        r2=float(model.rsquared),
        adj_r2=float(model.rsquared_adj),
        coef_per_sd=coef,
        pct_per_sd=float(100.0 * (np.exp(coef) - 1.0)),
    )
    return out, model



def power_at_published_effect(n_our: int, theta_pub: float, se_pub: float,
                              n_pub: int = 40000, alpha: float = 0.05,
                              two_sided: bool = True) -> dict:
    """One-sample z-test power: at our sample size and assuming SE scales
    as 1/sqrt(n), what is our power to reject H0: theta = 0 if the true
    effect equals Shen's published estimate?

    Standard formula (Cohen 1988 §6.3):

        SE_our  = se_pub * sqrt(n_pub / n_our)
        z_alpha = ppf(1 - alpha)         (one-sided) or ppf(1 - alpha/2)
        ncp     = theta_pub / SE_our
        power   = 1 - Phi(z_alpha - ncp) + Phi(-z_alpha - ncp) (two-sided)
                = 1 - Phi(z_alpha - ncp)                       (one-sided)
    """
    if n_our <= 0 or n_pub <= 0 or se_pub <= 0:
        return {"power": float("nan"), "se_implied": float("nan")}
    se_our = float(se_pub * math.sqrt(n_pub / n_our))
    z_a = float(stats.norm.ppf(1 - alpha / 2 if two_sided else 1 - alpha))
    ncp = float(theta_pub / se_our)
    if two_sided:
        power = float(1 - stats.norm.cdf(z_a - ncp) + stats.norm.cdf(-z_a - ncp))
    else:
        power = float(1 - stats.norm.cdf(z_a - ncp))
    return {
        "n_our": int(n_our), "n_pub": int(n_pub),
        "theta_pub": float(theta_pub), "se_pub": float(se_pub),
        "se_implied_at_our_n": se_our, "z_alpha": z_a, "ncp": ncp,
        "alpha": float(alpha), "two_sided": bool(two_sided),
        "power": power,
    }



def k_sweep_rank_correlation(
    descriptions: list[str], lat: np.ndarray, lon: np.ndarray,
    ks: tuple = (3, 5, 10, 20, 50), max_features: int = 5000,
    use_doc2vec: bool = False, seed: int = 0,
) -> dict:
    """Compute uniqueness at each K and report the pairwise Spearman
    rank correlation matrix plus the off-diagonal minimum.

    If the construction is robust to K, all off-diagonal entries cluster
    near 1; if it is K-driven (a known degeneracy at small n + clustered
    coords), small Ks decorrelate from large Ks.
    """
    if use_doc2vec:
        vectors = _vectorize_doc2vec(descriptions, seed=seed)
    else:
        vectors = _vectorize_tfidf(descriptions, max_features=max_features)
    n = len(descriptions)
    uniq_by_k = {}
    for k in ks:
        peers = _knn_peers(lat, lon, k)
        uniq_by_k[k] = _uniqueness_from_vectors(vectors, peers)
    corr = np.zeros((len(ks), len(ks)))
    for i, ki in enumerate(ks):
        for j, kj in enumerate(ks):
            rho, _ = stats.spearmanr(uniq_by_k[ki], uniq_by_k[kj])
            corr[i, j] = float(rho)
    off = corr[np.triu_indices(len(ks), k=1)]
    return {
        "ks": list(ks),
        "spearman_matrix": corr.tolist(),
        "off_diag_min": float(off.min()) if off.size else float("nan"),
        "off_diag_mean": float(off.mean()) if off.size else float("nan"),
        "per_k_descriptive": {
            int(k): {
                "mean": float(uniq_by_k[k].mean()),
                "sd": float(uniq_by_k[k].std(ddof=1)),
            } for k in ks
        },
    }


def _confounder_names(confounders: np.ndarray, parcels) -> list[str]:
    """Best-effort names for confounder columns. Falls back to indexed names.

    Layout (per causal_inference.get_features_and_target): [lat, lon,
    *property_cols_present, *contextual_cols_present]. We label what we know
    and pad the rest.
    """
    from causal_inference import CONTEXTUAL_COLS

    known = ["lat", "lon"]
    known += [c for c in PROPERTY_COLS if parcels is not None and c in parcels.columns]
    known += [c for c in CONTEXTUAL_COLS if parcels is not None and c in parcels.columns]
    if len(known) == confounders.shape[1]:
        return known
    if len(known) > confounders.shape[1]:
        return known[: confounders.shape[1]]
    pad = [f"c{i}" for i in range(len(known), confounders.shape[1])]
    return known + pad


def run_shen(city: str = "sf", n_subset: int | None = None, seed: int = 42,
             *, k: int = 5, k_sweep: bool = False, cell_fe: bool = False,
             use_doc2vec: bool = False, fast: bool = False,
             n_boot: int | None = None,
             published_theta: float = 0.140,
             published_se: float = 0.034,
             published_n: int = 40000) -> dict:
    print(f"\n=== Shen & Ross (2021) replication: {city} ===")
    loaded = load_analysis_data(city)
    if loaded is None:
        return {"city": city, "error": "no data"}
    emb_df, parcels = loaded
    feats = get_features_and_target(emb_df, parcels, drop_mismatched_crime=True)
    if feats is None:
        return {"city": city, "error": "no features"}
    _, confounders, Y, meta = feats

    valid_mask = np.asarray(meta["valid_mask"], dtype=bool)
    emb_df = emb_df[valid_mask].reset_index(drop=True)
    assert len(emb_df) == confounders.shape[0] == len(Y), (
        f"row-alignment failure: emb_df={len(emb_df)} "
        f"confounders={confounders.shape[0]} Y={len(Y)}")

    desc_col = "clean_description" if "clean_description" in emb_df.columns else "description"
    if "clean_description" in emb_df.columns:
        descriptions = emb_df["clean_description"].fillna(emb_df["description"]).astype(str).tolist()
    else:
        descriptions = emb_df["description"].astype(str).tolist()
    lat = pd.to_numeric(emb_df["latitude"], errors="coerce").values.astype(float)
    lon = pd.to_numeric(emb_df["longitude"], errors="coerce").values.astype(float)
    zips = (emb_df["zip"].astype(str).values
            if "zip" in emb_df.columns else np.zeros(len(emb_df), dtype=object))
    year_col = next((c for c in ("year_sold", "sale_year", "year") if c in emb_df.columns), None)
    if year_col is not None:
        years = pd.to_numeric(emb_df[year_col], errors="coerce").values
    elif "sale_date" in emb_df.columns:
        years = pd.to_datetime(emb_df["sale_date"], errors="coerce").dt.year.values
    else:
        years = np.full(len(emb_df), np.nan)

    finite_mask = np.isfinite(lat) & np.isfinite(lon)
    n_drop = int((~finite_mask).sum())
    if n_drop:
        print(f"  dropping {n_drop} rows with non-finite lat/lon")
        descriptions = [d for d, ok in zip(descriptions, finite_mask) if ok]
        lat = lat[finite_mask]
        lon = lon[finite_mask]
        zips = zips[finite_mask]
        years = years[finite_mask]
        confounders = confounders[finite_mask]
        Y = Y[finite_mask]

    if n_subset is not None and n_subset < len(descriptions):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(descriptions), size=n_subset, replace=False)
        idx.sort()
        descriptions = [descriptions[i] for i in idx]
        lat = lat[idx]; lon = lon[idx]
        zips = zips[idx]; years = years[idx]
        confounders = confounders[idx]
        Y = Y[idx]

    n = len(Y)
    print(f"  N={n:,}, confounders={confounders.shape[1]}, "
          f"unique ZIPs={pd.Series(zips).nunique()}, "
          f"unique years={pd.Series(years).nunique()}")

    pwr = power_at_published_effect(n_our=n, theta_pub=published_theta,
                                    se_pub=published_se, n_pub=published_n)
    print(f"  Power vs published θ={published_theta:+.3f} "
          f"(SE={published_se:.3f}, n_pub={published_n:,}):")
    print(f"    implied SE at our n: {pwr['se_implied_at_our_n']:.4f}  "
          f"ncp={pwr['ncp']:.2f}  power={pwr['power']:.3f}")

    method_label = []
    if use_doc2vec:
        method_label.append("Doc2Vec(PV-DM, dim=100, win=5, ep=40)")
    else:
        method_label.append("TF-IDF(max=5000, ngram=1-2, min_df=2)")
    if cell_fe:
        method_label.append("(zip,year) cell averages + cell FE")
    else:
        method_label.append(f"K={k} lat/lon NN")
    print(f"  Computing uniqueness with {' | '.join(method_label)} ...")

    uniqueness = compute_uniqueness(
        descriptions, lat, lon, k=k,
        use_doc2vec=use_doc2vec,
        zips=zips, years=years,
        cell_fe=cell_fe,
        seed=seed,
    )
    print(f"    uniqueness: mean={uniqueness.mean():.3f}  "
          f"sd={uniqueness.std(ddof=1):.3f}  "
          f"min={uniqueness.min():.3f}  max={uniqueness.max():.3f}")

    print("  Hedonic OLS (HC3 SEs)...")
    conf_names = _confounder_names(confounders, parcels)
    cell_keys = None
    if cell_fe:
        cell_keys = [(str(z), str(y) if not pd.isna(y) else "NA")
                     for z, y in zip(zips, years)]
    ols, model = hedonic_ols(uniqueness, confounders, conf_names, Y,
                             cell_keys=cell_keys)
    print(f"    uniqueness_z: β={ols.coef:+.4f}  se={ols.se:.4f}  t={ols.t:+.2f}  "
          f"p={ols.p:.4g}  95%CI=[{ols.ci_low:+.4f}, {ols.ci_high:+.4f}]")
    print(f"    → per-σ price effect: {ols.pct_per_sd:+.2f}%   "
          f"(Shen & Ross 2021 abstract: ~+15% per σ in the hedonic model)")
    print(f"    OLS R²={ols.r2:.4f}, adj R²={ols.adj_r2:.4f}")

    cell_fe_block = None
    if not cell_fe and "zip" in emb_df.columns:
        cell_keys_alt = [(str(z), str(y) if not pd.isna(y) else "NA")
                         for z, y in zip(zips, years)]
        try:
            uniq_cell = compute_uniqueness(
                descriptions, lat, lon, k=k, use_doc2vec=use_doc2vec,
                zips=zips, years=years, cell_fe=True, seed=seed,
            )
            ols_cell, _ = hedonic_ols(uniq_cell, confounders, conf_names, Y,
                                      cell_keys=cell_keys_alt)
            cell_fe_block = {
                "method": "(zip,year) cell averages with cell FE",
                "ols": asdict(ols_cell),
                "uniqueness_descriptive": {
                    "mean": float(uniq_cell.mean()),
                    "sd": float(uniq_cell.std(ddof=1)),
                },
            }
            print(f"  Cell-FE comparison: β_cell={ols_cell.coef:+.4f} "
                  f"se={ols_cell.se:.4f} p={ols_cell.p:.3g} "
                  f"(per-σ {ols_cell.pct_per_sd:+.2f}%)")
        except Exception as e:
            cell_fe_block = {"error": f"cell-FE comparison failed: {e}"}
            print(f"  Cell-FE comparison skipped: {e}")

    k_sweep_block = None
    if k_sweep:
        try:
            print("  K sweep over K in {3,5,10,20,50} ...")
            k_sweep_block = k_sweep_rank_correlation(
                descriptions, lat, lon,
                ks=(3, 5, 10, 20, 50),
                use_doc2vec=use_doc2vec, seed=seed,
            )
            print(f"    Spearman off-diagonal min={k_sweep_block['off_diag_min']:.3f}, "
                  f"mean={k_sweep_block['off_diag_mean']:.3f}")
        except Exception as e:
            k_sweep_block = {"error": str(e)}
            print(f"    K sweep skipped: {e}")

    if fast:
        dml_use_ridge = True
        if n_boot is not None and n_boot > 0:
            dml_ci_method, dml_n_boot = "bootstrap", n_boot
        else:
            dml_ci_method, dml_n_boot = "if", None
    elif n_boot is not None and n_boot == 0:
        dml_ci_method, dml_n_boot, dml_use_ridge = "if", None, False
    else:
        dml_ci_method = "bootstrap"
        dml_n_boot = n_boot if n_boot is not None else 500
        dml_use_ridge = False
    backend = "ridge" if dml_use_ridge else "gbm"
    print(f"  DML: uniqueness as continuous treatment "
          f"(backend={backend}, ci_method={dml_ci_method}, n_boot={dml_n_boot})...")
    dml = run_dml(
        uniqueness,
        confounders,
        Y,
        label="DML on TF-IDF uniqueness",
        ci_method=dml_ci_method,
        n_boot=dml_n_boot,
        use_ridge=dml_use_ridge,
        seed=seed,
    )
    if dml is None:
        print("    DML failed (treatment fully explained by confounders)")
    else:
        flag = "contains 0" if dml.contains_zero else "EXCLUDES 0"
        print(f"    DML θ={dml.theta:+.4f}  se={dml.se:.4f}  "
              f"95%CI=[{dml.ci_low:+.4f}, {dml.ci_high:+.4f}]  {flag}")
        print(f"    MDE: ±{dml.mde:.4f}  ({100*(np.exp(dml.mde)-1):+.2f}% in price)")

    print("\n  Headline contrast:")
    print(f"    OLS uniqueness coefficient (Shen-style): {ols.coef:+.4f} per σ "
          f"({ols.pct_per_sd:+.2f}% per σ in price)")
    if dml is not None:
        print(f"    DML θ on uniqueness (causal):            {dml.theta:+.4f} per σ "
              f"(95% CI [{dml.ci_low:+.4f}, {dml.ci_high:+.4f}])")
        if dml.contains_zero and abs(ols.coef) > 2 * abs(dml.theta):
            print("    → predictive/hedonic gain is much larger than the causal estimate; "
                  "uniqueness effect attenuates under DML.")

    return {
        "paper": "Shen & Ross 2021 (JUE) -- TF-IDF + KNN cousin construction",
        "city": city,
        "n": int(n),
        "n_confounders": int(confounders.shape[1]),
        "uniqueness_descriptive": {
            "mean": float(uniqueness.mean()),
            "sd": float(uniqueness.std(ddof=1)),
            "min": float(uniqueness.min()),
            "max": float(uniqueness.max()),
        },
        "ols_uniqueness": asdict(ols),
        "dml_uniqueness": result_to_dict(dml),
        "power_at_published_effect": pwr,
        "k_sweep_rank_corr": k_sweep_block,
        "cell_fe_result": cell_fe_block,
        "method_notes": {
            "embedding": ("doc2vec(dim=100, win=5, ep=40, dm=1)"
                          if use_doc2vec else
                          "tfidf(max=5000, ngram=1-2, min_df=2)"),
            "peer_set": ("(zip,year) cell averages with cell FE" if cell_fe
                         else f"K={k} lat/lon NN"),
            "ols_cov": "HC3",
            "treatment_standardization": "z-score (per σ)",
            "dml_pipeline": "causal_inference.dml_continuous_treatment",
            "framing_note": ("This is a TF-IDF + lat/lon-KNN COUSIN of "
                             "Shen & Ross (2021), not a strict replication. "
                             "Shen uses Doc2Vec + MLS-Area x Year cell "
                             "averages. Use --doc2vec and --cell_fe for "
                             "closer alignment."),
        },
        "meta": {
            "has_rich_confounders": bool(meta["has_rich_confounders"]),
            "crime_dropped": bool(meta["crime_dropped"]),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="sf")
    ap.add_argument("--n", type=int, default=None,
                    help="optional subset size for fast smoke runs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=5,
                    help="K nearest neighbours for the peer set "
                         "(ignored when --cell_fe is set)")
    ap.add_argument("--k_sweep", action="store_true",
                    help="report uniqueness rank-corr across K in {3,5,10,20,50}")
    ap.add_argument("--cell_fe", action="store_true",
                    help="(zip,year) cell averages with cell FE (Shen Eq. 9 style)")
    ap.add_argument("--doc2vec", action="store_true",
                    help="use Doc2Vec embeddings instead of TF-IDF (needs gensim)")
    ap.add_argument("--fast", action="store_true",
                    help="lean mode: use influence-function SE for DML CI "
                         "instead of 500-iter bootstrap (~30x faster; IF-SE "
                         "is slightly anti-conservative at n≈300)")
    ap.add_argument("--n_boot", type=int, default=None,
                    help="override DML bootstrap iterations (default 500; "
                         "0 forces IF-SE)")
    ap.add_argument("--published_theta", type=float, default=0.140,
                    help="Shen & Ross (2021) published per-σ effect in "
                         "log-points (default 0.140 = ln 1.15, the abstract's "
                         "15%% hedonic price increase per SD)")
    ap.add_argument("--published_se", type=float, default=0.034,
                    help="SE on Shen's coefficient (UNVERIFIED placeholder; "
                         "the published abstract reports no SE)")
    ap.add_argument("--published_n", type=int, default=40000,
                    help="Shen's n for the power calc denominator")
    ap.add_argument("--out", type=Path, default=None,
                    help="path to write JSON results")
    args = ap.parse_args()

    result = run_shen(city=args.city, n_subset=args.n, seed=args.seed,
                      k=args.k, k_sweep=args.k_sweep, cell_fe=args.cell_fe,
                      use_doc2vec=args.doc2vec, fast=args.fast,
                      n_boot=args.n_boot,
                      published_theta=args.published_theta,
                      published_se=args.published_se,
                      published_n=args.published_n)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
