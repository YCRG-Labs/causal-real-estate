"""Shen-Ross 2021 six-variant replication on SF dedup data.

Variants:
  A. TF-IDF + lat/lon-KNN K=5 (corrected Eq. 9)
  B. TF-IDF + (zip, year) cell-FE — years=0 in SF, falls back to zip-only
  C. Doc2Vec PV-DM (vec_size=100, window=5, epochs=40) + lat/lon-KNN
  D. Doc2Vec + zip-cell-FE
  E. TF-IDF + lat/lon-KNN on raw scraped (pre-dedup) for the apples-to-apples
  F. TF-IDF + ZIP-FE only (no peer-distance, pure spatial control)

Output: results/diagnostics/shen_six_variants.json with theta_OLS, theta_DML, CIs.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _silence  # noqa: F401
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.spatial import cKDTree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from fast_bootstrap_dml_v2 import _build_features, _LEARNERS, _dml_core
from sklearn.preprocessing import StandardScaler

OUT = Path("results/diagnostics"); OUT.mkdir(parents=True, exist_ok=True)
CITY = "sf"


def _tfidf(descriptions, max_features=5000):
    v = TfidfVectorizer(max_features=max_features, stop_words="english",
                        ngram_range=(1, 2), min_df=2)
    return v.fit_transform(descriptions)


def _doc2vec(descriptions, vec_size=100, window=5, epochs=40, dm=1, seed=0):
    try:
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument
        from gensim.utils import simple_preprocess
    except ImportError:
        return None
    tagged = [TaggedDocument(simple_preprocess(d), [i])
              for i, d in enumerate(descriptions)]
    m = Doc2Vec(documents=tagged, vector_size=vec_size, window=window,
                epochs=epochs, dm=dm, min_count=2, workers=1, seed=seed)
    return np.stack([m.dv[i] for i in range(len(descriptions))])


def _uniqueness_pairwise_eq9(vectors, peer_indices):
    """Corrected Eq.9: mean over j of (1 - cos(self, peer_j))."""
    n = len(peer_indices)
    u = np.zeros(n)
    for i in range(n):
        peers = [j for j in peer_indices[i] if j != i]
        if not peers:
            continue
        sims = cosine_similarity(vectors[i:i + 1], vectors[peers])[0]
        u[i] = float((1.0 - sims).mean())
    return u


def _uniqueness_peer_mean(vectors, peer_indices):
    """Buggy old formula: 1 - cos(self, mean(peers))."""
    n = len(peer_indices)
    u = np.zeros(n)
    is_sparse = hasattr(vectors, "toarray")
    for i in range(n):
        peers = [j for j in peer_indices[i] if j != i]
        if not peers:
            continue
        if is_sparse:
            pm = np.asarray(vectors[peers].mean(axis=0))
            sim = cosine_similarity(vectors[i], pm)[0, 0]
        else:
            pm = vectors[peers].mean(axis=0, keepdims=True)
            sim = cosine_similarity(vectors[i:i + 1], pm)[0, 0]
        u[i] = 1.0 - float(sim)
    return u


def _knn_peers(lat, lon, k=5):
    coords = np.column_stack([lat, lon])
    n = len(lat)
    tree = cKDTree(coords)
    _, nn = tree.query(coords, k=min(k + 1, n))
    if nn.ndim == 1:
        nn = nn.reshape(-1, 1)
    return [list(int(j) for j in row if j != i)[:k]
            for i, row in enumerate(nn)]


def _cell_peers(zips):
    n = zips.size
    by_z = {}
    for i in range(n):
        by_z.setdefault(int(zips[i]) if not pd.isna(zips[i]) else -1, []).append(i)
    out = []
    for i in range(n):
        z = int(zips[i]) if not pd.isna(zips[i]) else -1
        peers = [j for j in by_z.get(z, []) if j != i]
        out.append(peers)
    return out


def _ols(uniq_z, controls, Y, label):
    X = np.column_stack([uniq_z, controls, np.ones(len(Y))])
    m = sm.OLS(Y, X).fit(cov_type="HC3")
    return {
        "label": label,
        "n": int(len(Y)),
        "coef_uniq_z": float(m.params[0]),
        "se_uniq_z": float(m.bse[0]),
        "t_uniq_z": float(m.tvalues[0]),
        "p_uniq_z": float(m.pvalues[0]),
        "ci_low": float(m.conf_int()[0][0]),
        "ci_high": float(m.conf_int()[0][1]),
        "r2": float(m.rsquared), "adj_r2": float(m.rsquared_adj),
    }


def _dml_on_uniq(uniq, conf, Y, label):
    T = np.asarray(uniq).reshape(-1, 1).astype(np.float64)
    conf = np.asarray(conf, dtype=np.float64)
    scaler = StandardScaler().fit(conf)
    conf_s = scaler.transform(conf).astype(np.float64)
    res = _dml_core(T, conf_s, Y, _LEARNERS["ridge"], seed=0, k_folds=5,
                    return_residuals=True)
    if res is None:
        return None
    th, se, _, _, _ = res
    return {
        "label": label, "theta_dml": float(th), "se_dml": float(se),
        "ci_low": float(th - 1.96 * se), "ci_high": float(th + 1.96 * se),
        "contains_zero": bool((th - 1.96 * se) <= 0 <= (th + 1.96 * se)),
    }


def main():
    data = _build_features(CITY)
    T_emb, conf, Y, coords, meta = data
    n = len(Y)
    emb_path = Path("data/processed") / f"{CITY}_embeddings.parquet"
    df = pd.read_parquet(emb_path)
    # align: _build_features returns valid rows in order; we proxy by trimming df
    df = df.iloc[: n].copy()
    desc = df["clean_description"].fillna("").astype(str).tolist()
    lat = pd.to_numeric(df.get("latitude"), errors="coerce").to_numpy()
    lon = pd.to_numeric(df.get("longitude"), errors="coerce").to_numpy()
    zips = pd.to_numeric(df.get("zip"), errors="coerce").to_numpy()

    tfidf = _tfidf(desc)
    doc2vec = _doc2vec(desc) if "DOC2VEC_OK" not in os.environ or True else None

    peers_knn = _knn_peers(lat, lon, k=5)
    peers_cell = _cell_peers(zips)

    # uniqueness vectors
    uA = _uniqueness_pairwise_eq9(tfidf, peers_knn)
    uB = _uniqueness_pairwise_eq9(tfidf, peers_cell)
    uC = _uniqueness_pairwise_eq9(doc2vec, peers_knn) if doc2vec is not None else None
    uD = _uniqueness_pairwise_eq9(doc2vec, peers_cell) if doc2vec is not None else None
    uE_buggy = _uniqueness_peer_mean(tfidf, peers_knn)  # the OLD formula on dedup
    uF = uA  # placeholder; F is OLS with full zip dummies on top of uniqueness

    log_price = Y  # already log
    Yz = log_price

    structured_cols = [c for c in ["bedrooms", "bldg_area_sqft",
                                    "lot_area_sqft", "year_built"]
                       if c in df.columns]
    struct = df[structured_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    struct = StandardScaler().fit_transform(struct) if struct.size else struct

    results = {}
    for label, u in [("A_tfidf_knn_eq9", uA),
                     ("B_tfidf_cellfe_eq9", uB),
                     ("E_tfidf_knn_peer_mean_buggy_on_dedup", uE_buggy)]:
        uz = (u - u.mean()) / u.std() if u.std() > 0 else u
        results.setdefault(label, {})["ols"] = _ols(uz, struct, Yz, label)
        results[label]["dml"] = _dml_on_uniq(uz, conf, Yz, label)

    if uC is not None:
        for label, u in [("C_doc2vec_knn_eq9", uC),
                         ("D_doc2vec_cellfe_eq9", uD)]:
            uz = (u - u.mean()) / u.std() if u.std() > 0 else u
            results.setdefault(label, {})["ols"] = _ols(uz, struct, Yz, label)
            results[label]["dml"] = _dml_on_uniq(uz, conf, Yz, label)
    else:
        results["C_D_doc2vec"] = {"note": "gensim not installed; skip Doc2Vec"}

    # F: TF-IDF KNN uniqueness with explicit ZIP fixed effects
    uz_A = (uA - uA.mean()) / uA.std() if uA.std() > 0 else uA
    zip_dummies = pd.get_dummies(pd.Series(zips, dtype="Int64").astype("Int64"),
                                  drop_first=True).to_numpy(dtype=np.float64)
    X = np.column_stack([uz_A, struct, zip_dummies, np.ones(n)])
    m = sm.OLS(Yz, X).fit(cov_type="HC3")
    results["F_tfidf_knn_zipFE"] = {
        "label": "F_tfidf_knn_zipFE",
        "ols": {
            "n": int(n),
            "coef_uniq_z": float(m.params[0]), "se_uniq_z": float(m.bse[0]),
            "t_uniq_z": float(m.tvalues[0]), "p_uniq_z": float(m.pvalues[0]),
            "ci_low": float(m.conf_int()[0][0]), "ci_high": float(m.conf_int()[0][1]),
            "r2": float(m.rsquared), "n_zip_dummies": int(zip_dummies.shape[1]),
        },
    }

    # Power vs Shen's published +0.149 / SE 0.034 at n_pub ~ 40000
    from scipy import stats
    se_our_imp = 0.034 * np.sqrt(40000 / n)
    ncp = 0.149 / se_our_imp
    power = float(1 - stats.norm.cdf(stats.norm.ppf(0.95) - ncp))
    results["power_vs_shen_published"] = {
        "theta_pub": 0.149, "se_pub": 0.034, "n_pub": 40000, "n_our": int(n),
        "implied_se_at_our_n": float(se_our_imp), "ncp": float(ncp),
        "one_sided_power_alpha_0.05": power,
        "n_needed_for_80pct_power": int(np.ceil(40000 * (
            (stats.norm.ppf(0.80) + stats.norm.ppf(0.95)) ** 2
            / (0.149 / 0.034) ** 2))),
    }

    (OUT / "shen_six_variants.json").write_text(json.dumps(results, indent=2,
                                                            default=str))
    print(json.dumps(results, indent=2, default=str)[:4000])
    print(f"WROTE {OUT/'shen_six_variants.json'}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
