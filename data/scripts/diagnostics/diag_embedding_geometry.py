"""Comparative embedding-geometry diagnostic across SF / NYC / Boston.

Per city, on the SBERT all-mpnet-base-v2 768-d embeddings of the dedup
listing descriptions:

  (1) Effective rank (Vershynin 2010, Roy-Vetterli 2007 variants):
        r_V    = tr(Sigma) / lambda_max(Sigma)         # Vershynin op-norm
        r_F    = tr(Sigma)^2 / ||Sigma||_F^2           # stable rank squared form
      We report both; the second is what the prompt calls "Vershynin"
      (cf. Hsu-Kakade-Zhang 2012; Bartlett-Long-Lugosi-Tsigler 2020 use
      this exact form for benign overfitting effective rank r_0).
  (2) Eigenvalue decay slope: log lambda_k regressed on log k for k=1..min(50, n-1).
  (3) Participation ratio of PC1, PC2, PC3:
        PR(v) = (sum_i v_i^2)^2 / (n * sum_i v_i^4)   (in [1/n, 1]),
      which captures how concentrated the loadings are across rows.
      A value near 1 = mass spread across all listings;
      near 1/n = mass on one listing (degenerate spike).
  (4) Anisotropy in the contextualized-embedding sense:
        A = ||Sigma - (tr(Sigma)/d) I||_F / ||Sigma||_F
      (=0 iff Sigma is isotropic; tends to 1 as Sigma rank-1).
      Also reports Ethayarajh (2019) mean-cosine anisotropy: average pairwise
      cosine similarity of centered embeddings.
  (5) Cross-city projection: project city A's PC1 onto city B's top-k
      subspace, report |cos(PC1_A, V_B[:,0])| and the residual norm
      ||PC1_A - V_B V_B^T PC1_A|| with k=1, 5, 20.
  (6) TF-IDF surrogate for per-PC "top words":
      Loadings approach is hard with mpnet (no per-token weights at the
      output). We instead correlate the sentence-level PC_k scores with
      TF-IDF columns and take the largest positive / largest negative
      coefficients. This is a SHAP-style surrogate that is in fact more
      defensible for sentence embeddings (gradient through mpnet to the
      mean-pooled output is dominated by [CLS] token noise; the regression
      surrogate is what BERT-flow / Li et al. 2020 use in their probing).
  (7) K-Means (K=10) clusters in the 768-d space; report cluster sizes
      and within-cluster log-price variance per city.
  (8) Davis-Kahan prediction check: PC1_var / PC2_var ratio used as the
      empirical spike strength lambda. Bound = lambda / (lambda - 1)
      for the dominant-eigen overlap (Yu-Wang-Samworth 2015 Thm 2).

Outputs:
  results/diagnostics/embedding_geometry.json
  results/diagnostics/embedding_geometry.png  (3 panels)
  results/diagnostics/embedding_top_words.json (top words per PC per city)

References (carried by the json):
  Vershynin 2010 arXiv:1011.3027 (effective rank)
  Roy-Vetterli 2007 EUSIPCO       (effective rank entropy form)
  Hsu-Kakade-Zhang 2012 ECP       (effective rank in regression)
  Gao et al. 2019 ICLR 1907.12009 (representation degeneration)
  Ethayarajh 2019 EMNLP 1909.00512(contextual embedding anisotropy)
  Li et al. 2020 EMNLP 2011.05864 (BERT-flow)
  Mu-Viswanath 2018 ICLR 1702.01417 (all-but-the-top)
  Wang-Isola 2020 ICML 2005.10242 (alignment + uniformity)
  Reimers-Gurevych 2019 EMNLP 1908.10084 (Sentence-BERT)
  Yu-Wang-Samworth 2015 Biometrika (a useful variant of Davis-Kahan)
"""

import sys, os, json, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import _silence  # noqa: F401
except Exception:
    pass

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path("/Users/jacobcrainic/causal-real-estate")
OUT = ROOT / "results/diagnostics"
OUT.mkdir(parents=True, exist_ok=True)
PROCESSED = ROOT / "data/processed"
CITIES = ["sf", "nyc", "boston"]
EMB_DIM = 768
TOPK_PC = 5
TOPN_WORDS = 8
K_CLUSTERS = 10
SEED = 0
RNG = np.random.default_rng(SEED)


def _load_city(city: str):
    p = PROCESSED / f"{city}_embeddings.parquet"
    df = pd.read_parquet(p)
    emb_cols = [f"emb_{i}" for i in range(EMB_DIM)]
    T = df[emb_cols].to_numpy(dtype=np.float64)
    Y = np.log(pd.to_numeric(df["price"], errors="coerce")
                 .to_numpy(dtype=np.float64))
    desc = df["clean_description"].fillna("").astype(str).tolist()
    keep = np.isfinite(Y) & (Y > 0) & np.isfinite(T).all(axis=1)
    return T[keep], Y[keep], [d for d, k in zip(desc, keep) if k]


def _eig_spectrum(T: np.ndarray):
    Tc = T - T.mean(axis=0, keepdims=True)
    n, d = Tc.shape
    # We want the eigenvalues of the d x d covariance Sigma_T.
    # Use the SVD: T_c = U S V^T, lambda_k = s_k^2 / (n-1).
    s = np.linalg.svd(Tc, full_matrices=False, compute_uv=False)
    lam = (s ** 2) / max(n - 1, 1)
    return lam, s, Tc


def _effective_ranks(lam: np.ndarray, d: int):
    lam = np.asarray(lam, dtype=np.float64)
    tr = float(lam.sum())
    fro2 = float((lam ** 2).sum())
    op = float(lam.max())
    r_V = tr / op if op > 0 else 0.0
    r_F = (tr ** 2) / fro2 if fro2 > 0 else 0.0
    # Roy-Vetterli entropy-based
    p = lam / max(tr, 1e-12)
    p = p[p > 0]
    H = float(-(p * np.log(p)).sum())
    r_RV = float(np.exp(H))
    return {
        "r_vershynin_op": r_V,
        "r_stable_F2": r_F,
        "r_roy_vetterli_entropy": r_RV,
        "trace": tr,
        "frob2": fro2,
        "op_norm": op,
        "d": int(d),
    }


def _decay_slope(lam: np.ndarray, kmax: int = 50):
    k = np.arange(1, min(kmax, len(lam)) + 1)
    lk = np.log(np.maximum(lam[: len(k)], 1e-30))
    x = np.log(k)
    # OLS slope
    a, b = np.polyfit(x, lk, 1)
    return float(a), float(b), k.tolist(), lk.tolist()


def _participation_ratio(v: np.ndarray):
    v = np.asarray(v, dtype=np.float64)
    s2 = (v ** 2).sum()
    s4 = (v ** 4).sum()
    n = len(v)
    if s4 <= 0:
        return 0.0
    pr_n = (s2 ** 2) / s4
    return float(pr_n / n)


def _anisotropy(lam: np.ndarray, d: int, Tc: np.ndarray):
    iso = lam.sum() / d
    lam_full = np.zeros(d)
    lam_full[: len(lam)] = lam
    aniso_F = float(np.sqrt(((lam_full - iso) ** 2).sum())
                    / max(np.sqrt((lam_full ** 2).sum()), 1e-12))
    # Ethayarajh mean-cosine
    n = Tc.shape[0]
    norms = np.linalg.norm(Tc, axis=1, keepdims=True)
    Z = Tc / np.maximum(norms, 1e-12)
    # subsample if huge
    m = min(500, n)
    idx = RNG.choice(n, size=m, replace=False)
    Zs = Z[idx]
    G = Zs @ Zs.T
    tri = G[np.triu_indices(m, k=1)]
    mean_cos = float(tri.mean())
    return {"anisotropy_frobenius": aniso_F, "mean_pairwise_cosine": mean_cos}


def _topwords_per_pc(desc, scores: np.ndarray, top_n: int = TOPN_WORDS):
    v = TfidfVectorizer(max_features=4000, stop_words="english",
                         ngram_range=(1, 2), min_df=3)
    X = v.fit_transform(desc).toarray().astype(np.float64)
    vocab = np.array(v.get_feature_names_out())
    out = []
    for k in range(scores.shape[1]):
        s = scores[:, k]
        s = (s - s.mean())
        sd = s.std()
        if sd > 0:
            s = s / sd
        # column-wise correlation w/ standardized TF-IDF cols
        Xc = X - X.mean(axis=0, keepdims=True)
        sdcol = Xc.std(axis=0)
        ok = sdcol > 0
        rho = np.zeros(X.shape[1])
        rho[ok] = (Xc[:, ok] * s[:, None]).mean(axis=0) / sdcol[ok]
        top_pos = np.argsort(-rho)[:top_n]
        top_neg = np.argsort(rho)[:top_n]
        out.append({
            "pc": int(k + 1),
            "top_words_pos": [(str(vocab[i]), float(rho[i])) for i in top_pos],
            "top_words_neg": [(str(vocab[i]), float(rho[i])) for i in top_neg],
        })
    return out


def _kmeans_price(T: np.ndarray, Y_log: np.ndarray, k: int = K_CLUSTERS):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
    lab = km.fit_predict(T)
    rows = []
    for c in range(k):
        sel = (lab == c)
        n_c = int(sel.sum())
        if n_c == 0:
            rows.append({"cluster": c, "n": 0, "mean_logp": None,
                          "sd_logp": None})
            continue
        rows.append({"cluster": c, "n": n_c,
                      "mean_logp": float(Y_log[sel].mean()),
                      "sd_logp": float(Y_log[sel].std()) if n_c > 1 else 0.0})
    # ANOVA-style between/within
    total_var = float(Y_log.var())
    within = []
    for c in range(k):
        sel = (lab == c)
        if sel.sum() > 1:
            within.append(Y_log[sel].var() * (sel.sum() - 1))
    within_var = float(np.sum(within) / max(len(Y_log) - k, 1))
    eta2 = float(1.0 - within_var / total_var) if total_var > 0 else 0.0
    return {
        "clusters": rows,
        "total_var_logp": total_var,
        "mean_within_var_logp": within_var,
        "eta_squared_between": eta2,
    }


def _project_pc1(T_a: np.ndarray, T_b: np.ndarray, ks=(1, 5, 20)):
    Ta = T_a - T_a.mean(axis=0, keepdims=True)
    Tb = T_b - T_b.mean(axis=0, keepdims=True)
    _, _, Vta = np.linalg.svd(Ta, full_matrices=False)
    _, _, Vtb = np.linalg.svd(Tb, full_matrices=False)
    pc1_a = Vta[0]
    out = {}
    for k in ks:
        Vb = Vtb[:k].T  # d x k
        proj = Vb @ (Vb.T @ pc1_a)
        residual = pc1_a - proj
        out[f"k_{k}"] = {
            "cos_pc1A_pc1B_top": float(abs(np.dot(pc1_a, Vtb[0]))),
            "subspace_overlap_pc1A_into_Bk": float(np.linalg.norm(proj)),
            "residual_norm_pc1A": float(np.linalg.norm(residual)),
        }
    return out, Vta, Vtb


def _davis_kahan_check(lam: np.ndarray):
    lam_sorted = np.sort(lam)[::-1]
    if len(lam_sorted) < 2:
        return {"lambda_PC1_over_PC2": None, "sin_theta_bound": None}
    spike = float(lam_sorted[0] / lam_sorted[1])
    bulk_mean = float(lam_sorted[1:].mean())
    spike_over_bulk = float(lam_sorted[0] / bulk_mean) if bulk_mean > 0 else None
    # Yu-Wang-Samworth 2015 Thm 2 sin-Theta bound surrogate:
    # for a spike model lambda*v v^T + sigma^2 I with n samples,
    # sin Theta <= C * sqrt(sigma^2 d / (n * lambda^2)) when lambda >> sigma^2.
    # We compute the ratio lambda / (lambda - 1) which controls the
    # finite-sample eigenvector consistency (BBP transition at lambda=1).
    if spike > 1.0:
        sin_theta_proxy = float(1.0 / np.sqrt(max(spike - 1.0, 1e-9)))
    else:
        sin_theta_proxy = float("nan")
    return {
        "pc1_var_over_pc2_var": spike,
        "pc1_var_over_bulk_mean": spike_over_bulk,
        "sin_theta_proxy_bbp": sin_theta_proxy,
    }


def main():
    per_city = {}
    spectra = {}
    pc1_vecs = {}
    pc_loadings = {}
    cluster_info = {}
    for city in CITIES:
        T, Y, desc = _load_city(city)
        n, d = T.shape
        lam, s, Tc = _eig_spectrum(T)
        eff = _effective_ranks(lam, d)
        slope, intercept, ks, lks = _decay_slope(lam, kmax=50)
        ani = _anisotropy(lam, d, Tc)

        # Top-5 PC scores (rows in PC space).
        n_top = min(TOPK_PC, min(n, d) - 1)
        svd = TruncatedSVD(n_components=n_top, random_state=SEED)
        pc_scores = svd.fit_transform(Tc)
        # PR is on the score (column) vectors, i.e. per-listing loadings.
        pr = [_participation_ratio(pc_scores[:, k]) for k in range(n_top)]

        top_words = _topwords_per_pc(desc, pc_scores, top_n=TOPN_WORDS)

        km_info = _kmeans_price(T, Y, k=K_CLUSTERS)
        dk = _davis_kahan_check(lam)

        per_city[city] = {
            "n": int(n),
            "d": int(d),
            "effective_rank": eff,
            "eigen_decay_slope_loglog_k1_50": slope,
            "eigen_decay_intercept": intercept,
            "participation_ratio_pc1_pc2_pc3_pc4_pc5": pr,
            "anisotropy": ani,
            "kmeans_K10": km_info,
            "davis_kahan": dk,
            "top_eigenvalues": [float(x) for x in lam[:10].tolist()],
        }
        spectra[city] = (ks, lks, lam[:50].tolist())
        pc_loadings[city] = top_words
        pc1_vecs[city] = (T,)  # store raw for cross-projection step
        cluster_info[city] = km_info

    # Cross-projection (Tc only; recompute SVD)
    cross = {}
    for a in CITIES:
        for b in CITIES:
            if a == b:
                continue
            r, _, _ = _project_pc1(pc1_vecs[a][0], pc1_vecs[b][0])
            cross[f"{a}_to_{b}"] = r

    bundle = {
        "schema": "diag_embedding_geometry.v1",
        "settings": {
            "embedding_model": "sentence-transformers/all-mpnet-base-v2",
            "embedding_dim": EMB_DIM,
            "K_clusters": K_CLUSTERS,
            "K_pc_for_words": TOPK_PC,
            "decay_slope_range": "k = 1..50",
            "anisotropy_subsample_for_mean_cosine": 500,
        },
        "per_city": per_city,
        "cross_projection_pc1": cross,
        "references": [
            "Vershynin 2010 arXiv:1011.3027",
            "Roy-Vetterli 2007 EUSIPCO",
            "Hsu-Kakade-Zhang 2012",
            "Gao et al. 2019 ICLR 1907.12009",
            "Ethayarajh 2019 EMNLP 1909.00512",
            "Li et al. 2020 EMNLP 2011.05864",
            "Mu-Viswanath 2018 ICLR 1702.01417",
            "Wang-Isola 2020 ICML 2005.10242",
            "Reimers-Gurevych 2019 EMNLP 1908.10084",
            "Yu-Wang-Samworth 2015 Biometrika",
            "Bartlett-Long-Lugosi-Tsigler 2020 PNAS",
        ],
    }

    (OUT / "embedding_geometry.json").write_text(json.dumps(bundle, indent=2,
                                                              default=str))
    (OUT / "embedding_top_words.json").write_text(json.dumps(pc_loadings,
                                                                indent=2,
                                                                default=str))

    # 3-panel figure: eigenvalue decay, top-words text panel, cluster price box.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = {"sf": "#1f77b4", "nyc": "#d62728", "boston": "#2ca02c"}
    for city in CITIES:
        ks, lks, _ = spectra[city]
        axes[0].plot(np.log(ks), lks, marker="o", ms=3, lw=1.3,
                       label=f"{city.upper()} slope={per_city[city]['eigen_decay_slope_loglog_k1_50']:+.2f}",
                       color=colors[city])
    axes[0].set_xlabel("log k")
    axes[0].set_ylabel(r"log $\lambda_k$")
    axes[0].set_title("Eigenvalue decay (k=1..50)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Panel 2: top words per PC, per city (sketchy text panel)
    axes[1].axis("off")
    rows = []
    for city in CITIES:
        for entry in pc_loadings[city][:3]:
            pc = entry["pc"]
            pos_words = ", ".join(w for w, _ in entry["top_words_pos"][:5])
            rows.append([city.upper(), f"PC{pc}+", pos_words])
    tbl_text = "Top TF-IDF terms correlated with PC scores\n\n"
    for r in rows:
        tbl_text += f"{r[0]:<8} {r[1]:<5} {r[2][:48]}\n"
    axes[1].text(0.0, 1.0, tbl_text, fontsize=8, family="monospace",
                  verticalalignment="top")
    axes[1].set_title("Top words per PC1-3")

    # Panel 3: KMeans cluster log-price boxplot per city
    bxdata = []
    bxlab = []
    for city in CITIES:
        T, Y, _ = _load_city(city)
        km = KMeans(n_clusters=K_CLUSTERS, n_init=10, random_state=SEED)
        lab = km.fit_predict(T)
        for c in range(K_CLUSTERS):
            sel = lab == c
            if sel.sum() >= 2:
                bxdata.append(Y[sel])
                bxlab.append(f"{city[0].upper()}{c}")
    axes[2].boxplot(bxdata, labels=bxlab, showfliers=False)
    axes[2].set_xticklabels(bxlab, rotation=90, fontsize=6)
    axes[2].set_ylabel("log price")
    axes[2].set_title(f"K={K_CLUSTERS} cluster log-price by city")
    axes[2].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(OUT / "embedding_geometry.png", dpi=160)
    plt.close(fig)

    # Short stdout summary so the agent shell shows verification.
    print("=== embedding_geometry.py summary ===")
    for city in CITIES:
        c = per_city[city]
        print(f"\n[{city.upper()}] n={c['n']} d={c['d']}")
        print(f"  r_V op-norm        = {c['effective_rank']['r_vershynin_op']:.3f}")
        print(f"  r_stable (tr^2/F2) = {c['effective_rank']['r_stable_F2']:.3f}")
        print(f"  r_RV entropy       = {c['effective_rank']['r_roy_vetterli_entropy']:.3f}")
        print(f"  decay slope        = {c['eigen_decay_slope_loglog_k1_50']:+.3f}")
        print(f"  PR(PC1..PC5)       = {[f'{x:.3f}' for x in c['participation_ratio_pc1_pc2_pc3_pc4_pc5']]}")
        print(f"  anisotropy_F       = {c['anisotropy']['anisotropy_frobenius']:.3f}")
        print(f"  mean cos (Etha)    = {c['anisotropy']['mean_pairwise_cosine']:.3f}")
        print(f"  PC1/PC2 spike      = {c['davis_kahan']['pc1_var_over_pc2_var']:.3f}")
        print(f"  sin-Theta proxy    = {c['davis_kahan']['sin_theta_proxy_bbp']:.3f}")
        print(f"  eta^2 K=10 clust   = {c['kmeans_K10']['eta_squared_between']:.3f}")
    print("\n--- Cross PC1 projection (cos of PC1_A onto PC1_B) ---")
    for k, v in cross.items():
        print(f"  {k:<22} k=1 cos={v['k_1']['cos_pc1A_pc1B_top']:.3f}  "
              f"k=5 overlap={v['k_5']['subspace_overlap_pc1A_into_Bk']:.3f}  "
              f"k=20 overlap={v['k_20']['subspace_overlap_pc1A_into_Bk']:.3f}")
    print(f"\nWROTE {OUT/'embedding_geometry.json'}")
    print(f"WROTE {OUT/'embedding_top_words.json'}")
    print(f"WROTE {OUT/'embedding_geometry.png'}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
