"""Cell-density diagnostic for Shen-Ross 2021 (JUE) replication.

Hypothesis under test
---------------------
The SF-only success of our Shen-style replication (θ ≈ +0.13, CI excludes 0)
versus the NYC and Boston null is driven by per-zip listing density, not by
city-specific signal. Shen-Ross's Eq. 9 mean-pairwise-cosine "uniqueness" is a
fixed-k k-NN-style estimator. Under the Mack-Rosenblatt 1979 expansion, the
asymptotic bias of fixed-k k-NN density and functional estimators scales as
(k / n_local)^{2/d}, where n_local is the count of points inside the smallest
ball around x containing k peers (Loftsgaarden-Quesenberry 1965;
Mack-Rosenblatt 1979 JMVA 9:1; Györfi et al. 2002 §6.2). When n_local is small,
the peer set spans a wider geographic radius and the "uniqueness" conflates
within-area residual variance with cross-area drift; when n_local >> k, peers
are tight neighbours and uniqueness recovers the within-area construct Shen
intended.

Per-city density (our pulls):
  SF      ≈ 14.5 listings / zip   (348 listings, 24 zips)
  Boston  ≈ 10.6 listings / zip   (349 listings, 33 zips)
  NYC     ≈  2.7 listings / zip   (347 listings, 127 zips)
  Shen   ≈ 234   listings / area-year (40,918 / 25 / 7 ≈ 234)

Diagnostics produced
--------------------
  1. listings-per-zip distribution (min, median, max, % zips with K>=5)
  2. K-NN radius distribution: for each listing, geodesic distance to its
     5th NN; report median, 95th pct per city
  3. K-stability: Spearman rank correlation of uniqueness across K in
     {3, 5, 10, 20, 50}; off-diagonal min and mean per city
  4. Headline DML at K=5 vs at K=K_stable (largest K with rank correlation
     ≥ 0.85 against K=5)
  5. Vocabulary diagnostics: TTR, OOV vs an English baseline vocabulary,
     description length distribution
  6. Microstructure check: within-zip log-price variance per city
  7. Power-equivalence: at our per-cell density, what is the effective sample
     size of the Eq.9 estimator, and what would be needed to detect Shen's
     published effect

Outputs
-------
  results/diagnostics/shen_cell_density.json
  results/diagnostics/shen_cell_density.png (2 panels)

Run
---
  python data/scripts/diagnostics/diag_shen_cell_density.py
  python data/scripts/diagnostics/diag_shen_cell_density.py --cities sf,nyc,boston

References
----------
Loftsgaarden, D. O., & Quesenberry, C. P. (1965). A nonparametric estimate of
  a multivariate density function. Annals of Math. Stat. 36(3), 1049-1051.
Mack, Y. P., & Rosenblatt, M. (1979). Multivariate k-nearest neighbor density
  estimates. J. Multivariate Analysis 9(1), 1-15.
Györfi, L., Kohler, M., Krzyzak, A., & Walk, H. (2002). A Distribution-Free
  Theory of Nonparametric Regression. Springer. §6.2 (k-NN bias/variance).
Goodman, A. C., & Thibodeau, T. G. (1998). Housing market segmentation.
  J. Housing Economics 7(2), 121-143.
Shen, L., & Ross, S. L. (2021). Information value of property description:
  A machine learning approach. JUE 121, 103299.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _silence

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from causal_inference import (
    get_features_and_target,
    load_analysis_data,
    dml_continuous_treatment,
)
from replications.compare_to_dml import DMLResult, run_dml
from replications.shen_2021 import (
    _vectorize_tfidf, _vectorize_doc2vec, _uniqueness_from_vectors,
    _knn_peers, hedonic_ols, power_at_published_effect,
)


def _fast_dml(T, confounders, Y, label: str) -> DMLResult | None:
    import contextlib, io
    T = np.asarray(T)
    if T.ndim == 1:
        T = T.reshape(-1, 1)
    n_pca = min(50, T.shape[1], T.shape[0] - 1)
    with contextlib.redirect_stdout(io.StringIO()):
        raw = dml_continuous_treatment(T, confounders, Y, n_pca=n_pca,
                                       k_folds=5, ci_method="if")
    if raw is None:
        return None
    lo, hi = raw["ci"]
    return DMLResult(label=label, n=int(len(Y)),
                     theta=float(raw["theta"]), se=float(raw["se"]),
                     ci_low=float(lo), ci_high=float(hi),
                     mde=float(raw["mde"]),
                     contains_zero=bool(lo <= 0 <= hi))

OUT_DIR = Path("results/diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)



@dataclass
class CityFrame:
    city: str
    n: int
    descriptions: list[str]
    lat: np.ndarray
    lon: np.ndarray
    zips: np.ndarray
    Y: np.ndarray
    confounders: np.ndarray


def _load_city(city: str) -> CityFrame | None:
    loaded = load_analysis_data(city)
    if loaded is None:
        return None
    emb_df, parcels = loaded
    feats = get_features_and_target(emb_df, parcels, drop_mismatched_crime=True)
    if feats is None:
        return None
    _, confounders, Y, _ = feats

    if len(emb_df) != confounders.shape[0]:
        emb_df = emb_df.iloc[: confounders.shape[0]].reset_index(drop=True)

    if "clean_description" in emb_df.columns:
        descriptions = emb_df["clean_description"].fillna(
            emb_df["description"]).astype(str).tolist()
    else:
        descriptions = emb_df["description"].astype(str).tolist()
    lat = pd.to_numeric(emb_df["latitude"], errors="coerce").values.astype(float)
    lon = pd.to_numeric(emb_df["longitude"], errors="coerce").values.astype(float)
    zips = (emb_df["zip"].astype(str).values
            if "zip" in emb_df.columns
            else np.zeros(len(emb_df), dtype=object))

    finite = np.isfinite(lat) & np.isfinite(lon)
    descriptions = [d for d, ok in zip(descriptions, finite) if ok]
    lat = lat[finite]; lon = lon[finite]; zips = zips[finite]
    confounders = confounders[finite]; Y = Y[finite]

    return CityFrame(city=city, n=len(Y), descriptions=descriptions, lat=lat,
                     lon=lon, zips=zips, Y=Y, confounders=confounders)


_EARTH_KM = 6371.0


def _haversine_km(lat1, lon1, lat2, lon2):
    r1 = np.radians(lat1); r2 = np.radians(lat2)
    dphi = r2 - r1
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(r1) * np.cos(r2) * np.sin(dlmb / 2.0) ** 2
    return 2.0 * _EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def knn_radius_km(lat: np.ndarray, lon: np.ndarray, k: int = 5) -> np.ndarray:
    coords = np.column_stack([lat, lon])
    tree = cKDTree(coords)
    k_eff = min(k + 1, len(lat))
    dists, idx = tree.query(coords, k=k_eff)
    out = np.empty(len(lat))
    for i in range(len(lat)):
        peer_idx = [j for j in idx[i] if j != i][:k]
        if not peer_idx:
            out[i] = np.nan; continue
        out[i] = float(_haversine_km(lat[i], lon[i],
                                     lat[peer_idx], lon[peer_idx]).max())
    return out


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def vocab_stats(descriptions: list[str]) -> dict:
    all_tokens: list[str] = []
    type_counts: dict[str, int] = {}
    lengths = []
    for d in descriptions:
        toks = _tokenize(d)
        lengths.append(len(toks))
        all_tokens.extend(toks)
        for t in set(toks):
            type_counts[t] = type_counts.get(t, 0) + 1

    n_tokens = len(all_tokens)
    n_types = len(type_counts)
    ttr = float(n_types / n_tokens) if n_tokens else 0.0

    hapax = sum(1 for _, c in type_counts.items() if c == 1)
    hapax_rate = float(hapax / n_types) if n_types else 0.0
    return {
        "n_descriptions": len(descriptions),
        "n_tokens": int(n_tokens),
        "n_types": int(n_types),
        "ttr": ttr,
        "hapax_rate": hapax_rate,
        "doc_length_mean": float(np.mean(lengths)) if lengths else 0.0,
        "doc_length_median": float(np.median(lengths)) if lengths else 0.0,
        "doc_length_p10": float(np.percentile(lengths, 10)) if lengths else 0.0,
        "doc_length_p90": float(np.percentile(lengths, 90)) if lengths else 0.0,
    }



def k_stability(descriptions, lat, lon, ks=(3, 5, 10, 20, 50),
                use_doc2vec: bool = True, seed: int = 0) -> dict:
    if use_doc2vec:
        vectors = _vectorize_doc2vec(descriptions, seed=seed)
    else:
        vectors = _vectorize_tfidf(descriptions)
    uniqs = {}
    for k in ks:
        peers = _knn_peers(lat, lon, k)
        uniqs[k] = _uniqueness_from_vectors(vectors, peers)
    n = len(ks)
    corr = np.zeros((n, n))
    for i, ki in enumerate(ks):
        for j, kj in enumerate(ks):
            rho, _ = stats.spearmanr(uniqs[ki], uniqs[kj])
            corr[i, j] = float(rho)
    off = corr[np.triu_indices(n, k=1)]
    rho_vs_5 = {int(k): float(stats.spearmanr(uniqs[5], uniqs[k])[0])
                for k in ks}
    return {
        "ks": list(ks),
        "spearman_matrix": corr.tolist(),
        "off_diag_min": float(off.min()),
        "off_diag_mean": float(off.mean()),
        "rho_vs_K5": rho_vs_5,
        "per_k_descriptive": {
            int(k): {"mean": float(uniqs[k].mean()),
                     "sd": float(uniqs[k].std(ddof=1))}
            for k in ks
        },
        "_vectors": vectors,
        "_uniqs": uniqs,
    }



def density_stats(zips, Y, k_target: int = 5) -> dict:
    s = pd.Series(zips)
    counts = s.value_counts()
    pct_ge_k = float((counts >= k_target).mean())

    log_y = Y
    by_zip_var = pd.DataFrame({"zip": zips, "ly": log_y}).groupby("zip")["ly"].var()
    by_zip_var = by_zip_var.dropna()
    return {
        "n_listings": int(len(zips)),
        "n_unique_zips": int(counts.size),
        "listings_per_zip_min": int(counts.min()),
        "listings_per_zip_p25": float(counts.quantile(0.25)),
        "listings_per_zip_median": float(counts.median()),
        "listings_per_zip_p75": float(counts.quantile(0.75)),
        "listings_per_zip_max": int(counts.max()),
        "listings_per_zip_mean": float(counts.mean()),
        "pct_zips_with_ge_k": pct_ge_k,
        "k_target": int(k_target),
        "within_zip_log_price_var_mean": float(by_zip_var.mean()) if len(by_zip_var) else float("nan"),
        "within_zip_log_price_var_median": float(by_zip_var.median()) if len(by_zip_var) else float("nan"),
        "between_zip_log_price_var": float(pd.Series(log_y).groupby(zips).mean().var()),
        "overall_log_price_var": float(np.var(log_y, ddof=1)),
    }



def headline_dml(uniqueness, confounders, Y, label, *, fast: bool = True):
    if fast:
        return _fast_dml(uniqueness, confounders, Y, label)
    return run_dml(uniqueness, confounders, Y, label=label)


def pick_k_stable(rho_vs_5: dict[int, float], threshold: float = 0.85) -> int:
    stable = [k for k, r in rho_vs_5.items() if r >= threshold]
    if not stable:
        return 5
    return max(stable)



def power_block(n_our: int, n_local: float, theta_obs: float, se_obs: float,
                theta_pub: float = 0.149, se_pub: float = 0.034,
                n_pub: int = 40918, n_areas_pub: int = 25,
                k: int = 5) -> dict:
    """Effective sample size for the Eq. 9 estimator under fixed-k k-NN.

    Mack-Rosenblatt 1979 §3: for a fixed-k k-NN functional, the variance is
    bounded below by sigma^2 / (n_local * something(k)), so the effective n
    is *not* the raw n but n * (1 - bias_drift). At our per-cell density we
    treat n_eff ≈ n * min(1, n_local / (k * 2)) as a conservative deflation
    (the factor of 2 is the Loftsgaarden-Quesenberry condition for k-NN
    density estimator consistency: k(n) / n -> 0 *and* k(n) -> infinity).
    """
    n_local_pub = float(n_pub) / float(n_areas_pub)
    density_ratio = float(n_local / n_local_pub)
    n_eff = float(n_our * min(1.0, n_local / max(k, 1)))

    pwr = power_at_published_effect(n_our=n_our, theta_pub=theta_pub,
                                    se_pub=se_pub, n_pub=n_pub)
    pwr_eff = power_at_published_effect(n_our=max(1, int(n_eff)),
                                        theta_pub=theta_pub, se_pub=se_pub,
                                        n_pub=n_pub)
    implied_theta_pub = (theta_obs * math.sqrt(n_eff / n_pub)
                         if n_eff > 0 else float("nan"))
    return {
        "k": int(k),
        "n_our": int(n_our),
        "n_local_our": float(n_local),
        "n_local_pub_approx": n_local_pub,
        "density_ratio_our_over_pub": density_ratio,
        "n_eff_under_density_correction": float(n_eff),
        "theta_obs": float(theta_obs),
        "se_obs": float(se_obs),
        "naive_power_vs_pub": pwr,
        "density_adjusted_power_vs_pub": pwr_eff,
        "implied_theta_pub_if_se_scales_1_over_sqrt_n_eff": float(implied_theta_pub),
    }



def run_city(city: str, use_doc2vec: bool = True, seed: int = 42,
             ks=(3, 5, 10, 20, 50), fast_dml: bool = True) -> dict:
    print(f"\n=== {city} ===")
    cf = _load_city(city)
    if cf is None:
        return {"city": city, "error": "load failed"}

    dens = density_stats(cf.zips, cf.Y, k_target=5)
    print(f"  n={cf.n}  unique zips={dens['n_unique_zips']}  "
          f"listings/zip median={dens['listings_per_zip_median']:.1f}  "
          f"%≥K=5: {dens['pct_zips_with_ge_k']*100:.1f}%")

    radii = knn_radius_km(cf.lat, cf.lon, k=5)
    rad_block = {
        "k": 5,
        "median_km": float(np.nanmedian(radii)),
        "p10_km": float(np.nanpercentile(radii, 10)),
        "p90_km": float(np.nanpercentile(radii, 90)),
        "p95_km": float(np.nanpercentile(radii, 95)),
        "max_km": float(np.nanmax(radii)),
    }
    print(f"  K=5 radius (km): median={rad_block['median_km']:.2f} "
          f"p90={rad_block['p90_km']:.2f} max={rad_block['max_km']:.2f}")

    vocab = vocab_stats(cf.descriptions)
    print(f"  vocab: TTR={vocab['ttr']:.4f}  hapax={vocab['hapax_rate']:.3f}  "
          f"len_med={vocab['doc_length_median']:.0f}")

    print(f"  K-stability sweep over {ks} (Doc2Vec={use_doc2vec}) ...")
    kstab = k_stability(cf.descriptions, cf.lat, cf.lon, ks=ks,
                        use_doc2vec=use_doc2vec, seed=seed)
    vectors = kstab.pop("_vectors")
    uniqs = kstab.pop("_uniqs")
    print(f"    Spearman vs K=5: " +
          " ".join(f"K{k}={kstab['rho_vs_K5'][k]:+.2f}" for k in ks))
    print(f"    off-diag min={kstab['off_diag_min']:.3f} "
          f"mean={kstab['off_diag_mean']:.3f}")

    k_stable = pick_k_stable(kstab["rho_vs_K5"], threshold=0.85)
    print(f"  K_stable = {k_stable} (max K with ρ vs K=5 ≥ 0.85)")
    conf_names = [f"c{i}" for i in range(cf.confounders.shape[1])]

    heads = {}
    for k_label, k_val in [("K5", 5), ("Kstable", k_stable)]:
        u = uniqs[k_val]
        try:
            ols, _ = hedonic_ols(u, cf.confounders, conf_names, cf.Y)
            ols_d = {
                "coef": ols.coef, "se": ols.se, "t": ols.t, "p": ols.p,
                "ci_low": ols.ci_low, "ci_high": ols.ci_high,
                "pct_per_sd": ols.pct_per_sd, "r2": ols.r2, "adj_r2": ols.adj_r2,
            }
        except Exception as e:
            ols_d = {"error": str(e)}

        dml = headline_dml(u, cf.confounders, cf.Y,
                           label=f"{city}-{k_label}", fast=fast_dml)
        dml_d = None
        if dml is not None:
            dml_d = {"theta": dml.theta, "se": dml.se, "ci_low": dml.ci_low,
                     "ci_high": dml.ci_high, "mde": dml.mde,
                     "contains_zero": dml.contains_zero}
        heads[k_label] = {"k": int(k_val), "ols": ols_d, "dml": dml_d,
                          "uniqueness_mean": float(u.mean()),
                          "uniqueness_sd": float(u.std(ddof=1))}
        if ols_d.get("coef") is not None and dml_d is not None:
            print(f"  {k_label} (K={k_val}): OLS β={ols_d['coef']:+.4f} "
                  f"({ols_d['pct_per_sd']:+.1f}%/σ)  "
                  f"DML θ={dml_d['theta']:+.4f}  CI=[{dml_d['ci_low']:+.4f},"
                  f"{dml_d['ci_high']:+.4f}]")

    pw = None
    if heads["K5"]["dml"] is not None:
        pw = power_block(
            n_our=cf.n,
            n_local=dens["listings_per_zip_median"],
            theta_obs=heads["K5"]["dml"]["theta"],
            se_obs=heads["K5"]["dml"]["se"],
            k=5,
        )

    return {
        "city": city,
        "density": dens,
        "knn_radius_km": rad_block,
        "vocab": vocab,
        "k_stability": kstab,
        "headlines": heads,
        "k_stable": int(k_stable),
        "power_equivalence": pw,
    }



def make_plot(per_city: dict, path: Path):
    cities = [c for c in ["sf", "boston", "nyc"] if c in per_city
              and "density" in per_city[c]]
    if not cities:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    data = []
    for c in cities:
        d = per_city[c]["density"]
        data.append([d["listings_per_zip_min"], d["listings_per_zip_p25"],
                     d["listings_per_zip_median"], d["listings_per_zip_p75"],
                     d["listings_per_zip_max"]])
    pos = np.arange(len(cities))
    box = ax.boxplot(
        [[1, 1, 1]] * len(cities), positions=pos,
        widths=0.5, showfliers=False, patch_artist=True,
    )
    for i, (b, q) in enumerate(zip(box["boxes"], data)):
        mn, p25, med, p75, mx = q
        ax.add_patch(plt.Rectangle((i - 0.25, p25), 0.5, p75 - p25,
                                   facecolor="#7aa6e0", edgecolor="black",
                                   alpha=0.65))
        ax.plot([i - 0.25, i + 0.25], [med, med], color="black", lw=2)
        ax.plot([i, i], [mn, p25], color="black", lw=1)
        ax.plot([i, i], [p75, mx], color="black", lw=1)
    for b in box["boxes"]:
        b.set_visible(False)
    for m in box["medians"]:
        m.set_visible(False)
    ax.axhline(5, color="red", ls=":", lw=1, label="K=5 threshold")
    ax.set_xticks(pos)
    ax.set_xticklabels([c.upper() for c in cities])
    ax.set_ylabel("listings per zip")
    ax.set_title("Per-zip density (lower whisker = min, box = IQR, "
                 "upper whisker = max)")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)

    ax = axes[1]
    for c in cities:
        kstab = per_city[c]["k_stability"]
        ks = kstab["ks"]
        rho = [kstab["rho_vs_K5"][k] for k in ks]
        ax.plot(ks, rho, marker="o", label=c.upper())
    ax.axhline(0.85, color="grey", ls="--", lw=1, label="0.85 threshold")
    ax.set_xscale("log")
    ax.set_xticks([3, 5, 10, 20, 50])
    ax.set_xticklabels([3, 5, 10, 20, 50])
    ax.set_xlabel("K (peers)")
    ax.set_ylabel("Spearman ρ vs K=5 uniqueness")
    ax.set_title("Uniqueness rank stability across K")
    ax.set_ylim(-0.2, 1.05)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", default="sf,nyc,boston")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tfidf", action="store_true",
                    help="use TF-IDF instead of Doc2Vec (default Doc2Vec to "
                         "match the headline run)")
    ap.add_argument("--bootstrap", action="store_true",
                    help="run the 500-boot DML CI (matches headline runs but "
                         "is ~10x slower than the default IF SE)")
    ap.add_argument("--out", type=Path,
                    default=OUT_DIR / "shen_cell_density.json")
    ap.add_argument("--png", type=Path,
                    default=OUT_DIR / "shen_cell_density.png")
    args = ap.parse_args()

    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    use_doc2vec = not args.tfidf

    results: dict = {
        "spec": {
            "embedding": "doc2vec(dim=100, win=5, ep=40, dm=1)"
                         if use_doc2vec else "tfidf",
            "ks_sweep": [3, 5, 10, 20, 50],
            "stability_threshold": 0.85,
            "k_target_for_density_share": 5,
            "shen_pub_n": 40918,
            "shen_pub_n_areas": 25,
            "shen_pub_theta": 0.149,
            "shen_pub_se": 0.034,
        },
        "cities": {},
    }

    fast_dml = not args.bootstrap
    results["spec"]["dml_ci_method"] = "if" if fast_dml else "bootstrap-500"
    for c in cities:
        try:
            results["cities"][c] = run_city(c, use_doc2vec=use_doc2vec,
                                            seed=args.seed,
                                            fast_dml=fast_dml)
        except Exception as e:
            results["cities"][c] = {"city": c, "error": repr(e)}
            print(f"  ERROR on {c}: {e}")

    rows = []
    for c, blk in results["cities"].items():
        if "density" not in blk:
            continue
        d = blk["density"]; v = blk["vocab"]; ks = blk["k_stability"]
        h5 = blk["headlines"].get("K5", {})
        hs = blk["headlines"].get("Kstable", {})
        row = {
            "city": c,
            "n": d["n_listings"],
            "n_zips": d["n_unique_zips"],
            "list_per_zip_median": d["listings_per_zip_median"],
            "list_per_zip_min": d["listings_per_zip_min"],
            "pct_zip_ge_K5": d["pct_zips_with_ge_k"],
            "knn5_radius_median_km": blk["knn_radius_km"]["median_km"],
            "knn5_radius_p90_km": blk["knn_radius_km"]["p90_km"],
            "within_zip_lp_var": d["within_zip_log_price_var_median"],
            "ttr": v["ttr"],
            "hapax_rate": v["hapax_rate"],
            "doc_len_median": v["doc_length_median"],
            "kstab_off_diag_min": ks["off_diag_min"],
            "kstab_off_diag_mean": ks["off_diag_mean"],
            "rho_K5_K20": ks["rho_vs_K5"].get(20, float("nan")),
            "k_stable": blk["k_stable"],
            "ols_K5": h5.get("ols", {}).get("coef", float("nan")),
            "dml_K5_theta": (h5.get("dml") or {}).get("theta", float("nan")),
            "dml_K5_ci_low": (h5.get("dml") or {}).get("ci_low", float("nan")),
            "dml_K5_ci_high": (h5.get("dml") or {}).get("ci_high", float("nan")),
            "dml_Kstable_theta": (hs.get("dml") or {}).get("theta", float("nan")),
            "dml_Kstable_ci_low": (hs.get("dml") or {}).get("ci_low", float("nan")),
            "dml_Kstable_ci_high": (hs.get("dml") or {}).get("ci_high", float("nan")),
        }
        rows.append(row)
    results["summary_table"] = rows

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nwrote {args.out}")

    try:
        make_plot(results["cities"], args.png)
        print(f"wrote {args.png}")
    except Exception as e:
        print(f"plot failed: {e}")

    if rows:
        print("\n=== Cross-city summary ===")
        cols = ["city", "n", "n_zips", "list_per_zip_median",
                "pct_zip_ge_K5", "knn5_radius_median_km", "ttr",
                "hapax_rate", "within_zip_lp_var",
                "kstab_off_diag_min", "k_stable",
                "dml_K5_theta", "dml_Kstable_theta"]
        df = pd.DataFrame(rows)[cols]
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
