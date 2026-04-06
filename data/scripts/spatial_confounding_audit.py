"""
Spatial Confounding Audit Toolkit

Standalone diagnostic suite for detecting geographic confounding in text
embeddings derived from spatially situated data. Takes any (embeddings,
coordinates, prices) triple and produces a structured audit report.

Usage:
    from spatial_confounding_audit import audit_embeddings
    report = audit_embeddings(embeddings, latitudes, longitudes, prices)
    report.print_summary()
    report.to_dict()

Or from command line:
    python spatial_confounding_audit.py --embeddings emb.npy --coords coords.csv --prices prices.csv
"""

import sys
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Optional
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import spearmanr, bootstrap


@dataclass
class AuditReport:
    n_samples: int = 0
    embedding_dim: int = 0
    n_location_classes: int = 0

    nmi: float = 0.0
    location_accuracy: float = 0.0
    location_random_baseline: float = 0.0
    location_accuracy_ratio: float = 0.0

    spatial_correlation: float = 0.0
    spatial_correlation_pvalue: float = 0.0

    backdoor_delta_r2: float = 0.0
    dr_ate: float = 0.0
    dr_ci_low: float = 0.0
    dr_ci_high: float = 0.0
    dr_contains_zero: bool = True

    confounding_severity: str = ""
    recommendation: str = ""

    def print_summary(self):
        print(f"\n{'='*60}")
        print("SPATIAL CONFOUNDING AUDIT REPORT")
        print(f"{'='*60}")
        print(f"  Samples: {self.n_samples}, Embedding dim: {self.embedding_dim}")
        print(f"  Location classes: {self.n_location_classes}")
        print(f"\n  CONFOUNDING METRICS:")
        print(f"    NMI(embeddings, location):    {self.nmi:.4f}")
        print(f"    Location classifier accuracy: {self.location_accuracy:.4f} "
              f"(random: {self.location_random_baseline:.4f}, "
              f"ratio: {self.location_accuracy_ratio:.1f}x)")
        print(f"    Spatial autocorrelation:      {self.spatial_correlation:.4f} "
              f"(p={self.spatial_correlation_pvalue:.2e})")
        print(f"\n  CAUSAL ESTIMATES:")
        print(f"    Backdoor delta R2:            {self.backdoor_delta_r2:.4f}")
        print(f"    Doubly-robust ATE:            {self.dr_ate:.4f}")
        print(f"    95% CI:                       [{self.dr_ci_low:.4f}, {self.dr_ci_high:.4f}]")
        print(f"    CI contains zero:             {self.dr_contains_zero}")
        print(f"\n  VERDICT: {self.confounding_severity}")
        print(f"  {self.recommendation}")
        print(f"{'='*60}")

    def to_dict(self):
        return asdict(self)

    def to_json(self, path=None):
        d = self.to_dict()
        if path:
            with open(path, "w") as f:
                json.dump(d, f, indent=2)
        return json.dumps(d, indent=2)


def _discretize_locations(lat, lon, n_bins=20):
    lat_bins = pd.cut(lat, bins=n_bins, labels=False)
    lon_bins = pd.cut(lon, bins=n_bins, labels=False)
    labels = lat_bins * n_bins + lon_bins
    le = LabelEncoder()
    return le.fit_transform(np.nan_to_num(labels, nan=-1).astype(int))


def audit_embeddings(
    embeddings: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    prices: np.ndarray,
    location_labels: Optional[np.ndarray] = None,
    n_pca: int = 50,
    n_bootstrap: int = 500,
) -> AuditReport:
    report = AuditReport()

    embeddings = np.asarray(embeddings, dtype=float)
    latitudes = np.asarray(latitudes, dtype=float)
    longitudes = np.asarray(longitudes, dtype=float)
    prices = np.asarray(prices, dtype=float)

    valid = (
        ~np.any(np.isnan(embeddings), axis=1)
        & ~np.isnan(latitudes)
        & ~np.isnan(longitudes)
        & ~np.isnan(prices)
        & (prices > 0)
    )
    embeddings = embeddings[valid]
    latitudes = latitudes[valid]
    longitudes = longitudes[valid]
    prices = prices[valid]
    if location_labels is not None:
        location_labels = np.asarray(location_labels)[valid]

    n = len(embeddings)
    report.n_samples = n
    report.embedding_dim = embeddings.shape[1]

    if n < 30:
        report.confounding_severity = "INSUFFICIENT DATA"
        report.recommendation = "Need at least 30 samples for meaningful analysis."
        return report

    Y = np.log(prices)

    if location_labels is None:
        location_labels = _discretize_locations(latitudes, longitudes)

    le = LabelEncoder()
    loc_enc = le.fit_transform(location_labels)
    report.n_location_classes = len(le.classes_)

    n_clust = min(50, n)
    km = KMeans(n_clusters=n_clust, random_state=42, n_init=10).fit_predict(embeddings)
    report.nmi = normalized_mutual_info_score(loc_enc, km)

    unique_counts = np.bincount(loc_enc)
    valid_classes = np.where(unique_counts >= 3)[0]
    if len(valid_classes) >= 2:
        mask = np.isin(loc_enc, valid_classes)
        le2 = LabelEncoder()
        labels_f = le2.fit_transform(loc_enc[mask])
        n_comp = min(n_pca, embeddings.shape[1], mask.sum() - 1)
        pca = PCA(n_components=max(1, n_comp), random_state=42)
        emb_pca = pca.fit_transform(embeddings[mask])
        clf = LogisticRegression(max_iter=1000, random_state=42)
        n_cv = min(5, len(np.unique(labels_f)))
        scores = cross_val_score(clf, emb_pca, labels_f, cv=n_cv, scoring="accuracy")
        report.location_accuracy = scores.mean()
        report.location_random_baseline = 1.0 / len(np.unique(labels_f))
        report.location_accuracy_ratio = report.location_accuracy / max(report.location_random_baseline, 1e-10)

    from scipy.spatial.distance import pdist
    sample_n = min(n, 5000)
    idx = np.random.RandomState(42).choice(n, sample_n, replace=False) if n > sample_n else np.arange(n)
    geo_d = pdist(np.column_stack([latitudes[idx], longitudes[idx]]))
    emb_d = pdist(embeddings[idx], metric="cosine")
    rho, pval = spearmanr(geo_d, emb_d)
    report.spatial_correlation = rho
    report.spatial_correlation_pvalue = pval

    confounders = np.column_stack([latitudes, longitudes])
    scaler_c = StandardScaler()
    conf_s = scaler_c.fit_transform(confounders)

    n_comp = min(n_pca, embeddings.shape[1], n - 1)
    pca_full = PCA(n_components=max(1, n_comp), random_state=42)
    T_pca = pca_full.fit_transform(embeddings)
    scaler_t = StandardScaler()
    T_s = scaler_t.fit_transform(T_pca)

    full_feat = np.hstack([T_s, conf_s])
    kf = KFold(n_splits=min(5, n), shuffle=True, random_state=42)
    r2_full, r2_conf = [], []
    for tr, te in kf.split(Y):
        m1 = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
        m1.fit(full_feat[tr], Y[tr])
        r2_full.append(m1.score(full_feat[te], Y[te]))
        m2 = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
        m2.fit(conf_s[tr], Y[tr])
        r2_conf.append(m2.score(conf_s[te], Y[te]))
    report.backdoor_delta_r2 = np.mean(r2_full) - np.mean(r2_conf)

    T_norm = np.linalg.norm(T_pca, axis=1)
    treatment = (T_norm > np.median(T_norm)).astype(float)

    outcome = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    outcome.fit(np.hstack([treatment.reshape(-1, 1), conf_s]), Y)

    propensity = LogisticRegression(max_iter=1000, random_state=42)
    propensity.fit(conf_s, treatment)
    e = np.clip(propensity.predict_proba(conf_s)[:, 1], 0.05, 0.95)

    mu1 = outcome.predict(np.hstack([np.ones((n, 1)), conf_s]))
    mu0 = outcome.predict(np.hstack([np.zeros((n, 1)), conf_s]))

    report.dr_ate = float(np.mean(
        mu1 - mu0 + treatment * (Y - mu1) / e - (1 - treatment) * (Y - mu0) / (1 - e)
    ))

    def stat(idx, t=treatment, y=Y, m1=mu1, m0=mu0, ps=e):
        i = idx[0]
        return np.mean(m1[i] - m0[i] + t[i]*(y[i]-m1[i])/ps[i] - (1-t[i])*(y[i]-m0[i])/(1-ps[i]))

    rng = np.random.default_rng(42)
    ci = bootstrap((np.arange(n),), stat, n_resamples=n_bootstrap, random_state=rng, method="percentile")
    report.dr_ci_low = float(ci.confidence_interval.low)
    report.dr_ci_high = float(ci.confidence_interval.high)
    report.dr_contains_zero = report.dr_ci_low <= 0 <= report.dr_ci_high

    if report.nmi > 0.4 and report.location_accuracy_ratio > 5 and report.dr_contains_zero:
        report.confounding_severity = "SEVERE SPATIAL CONFOUNDING"
        report.recommendation = (
            "Embeddings encode geography, not independent semantics. "
            "Do not attribute predictive performance to semantic content "
            "without controlling for location."
        )
    elif report.nmi > 0.2 and report.location_accuracy_ratio > 2:
        report.confounding_severity = "MODERATE SPATIAL CONFOUNDING"
        report.recommendation = (
            "Substantial location information in embeddings. "
            "Causal claims about semantic content require confounding adjustment."
        )
    elif report.nmi > 0.1:
        report.confounding_severity = "MILD SPATIAL CONFOUNDING"
        report.recommendation = (
            "Some location signal present. Report confounding metrics "
            "alongside predictive performance."
        )
    else:
        report.confounding_severity = "LOW SPATIAL CONFOUNDING"
        report.recommendation = (
            "Embeddings appear largely location-independent. "
            "Semantic signal claims are plausible but should still be validated."
        )

    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Spatial Confounding Audit for Text Embeddings")
    parser.add_argument("--embeddings", required=True, help="Path to embeddings (.npy or .csv)")
    parser.add_argument("--coords", required=True, help="CSV with latitude,longitude columns")
    parser.add_argument("--prices", required=True, help="CSV with price column")
    parser.add_argument("--locations", help="Optional CSV with location labels")
    parser.add_argument("--output", help="Path to save JSON report")
    args = parser.parse_args()

    if args.embeddings.endswith(".npy"):
        emb = np.load(args.embeddings)
    else:
        emb = pd.read_csv(args.embeddings).values

    coords = pd.read_csv(args.coords)
    lat = coords["latitude"].values
    lon = coords["longitude"].values

    prices_df = pd.read_csv(args.prices)
    price_col = [c for c in prices_df.columns if "price" in c.lower()][0]
    prices = prices_df[price_col].values

    loc_labels = None
    if args.locations:
        loc_df = pd.read_csv(args.locations)
        loc_labels = loc_df.iloc[:, 0].values

    report = audit_embeddings(emb, lat, lon, prices, location_labels=loc_labels)
    report.print_summary()

    if args.output:
        report.to_json(args.output)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
