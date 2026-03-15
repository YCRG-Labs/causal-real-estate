import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from config import PROCESSED_DIR, CITIES, EMBEDDING_DIM


def load_embeddings(city):
    path = PROCESSED_DIR / f"{city}_embeddings.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    emb_cols = [f"emb_{i}" for i in range(EMBEDDING_DIM)]
    available = [c for c in emb_cols if c in df.columns]
    embeddings = df[available].values
    return df, embeddings


def compute_nmi(embeddings, locations, n_clusters=50):
    n_clusters = min(n_clusters, len(embeddings))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    emb_clusters = kmeans.fit_predict(embeddings)

    le = LabelEncoder()
    loc_labels = le.fit_transform(locations)

    nmi = normalized_mutual_info_score(loc_labels, emb_clusters)
    return nmi


def compute_location_classifier(embeddings, locations, n_components=50):
    le = LabelEncoder()
    loc_labels = le.fit_transform(locations)

    unique_counts = np.bincount(loc_labels)
    valid_classes = np.where(unique_counts >= 5)[0]
    if len(valid_classes) < 2:
        return None, None

    mask = np.isin(loc_labels, valid_classes)
    embeddings_filtered = embeddings[mask]
    labels_filtered = loc_labels[mask]

    le2 = LabelEncoder()
    labels_filtered = le2.fit_transform(labels_filtered)

    n_components = min(n_components, embeddings_filtered.shape[1], embeddings_filtered.shape[0])
    pca = PCA(n_components=n_components, random_state=42)
    emb_pca = pca.fit_transform(embeddings_filtered)

    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    scores = cross_val_score(clf, emb_pca, labels_filtered, cv=5, scoring="accuracy")

    n_classes = len(np.unique(labels_filtered))
    random_baseline = 1.0 / n_classes

    return scores.mean(), random_baseline


def compute_spatial_autocorrelation(embeddings, lats, lons, max_pairs=10000):
    n = len(embeddings)
    if n > max_pairs:
        idx = np.random.RandomState(42).choice(n, max_pairs, replace=False)
        embeddings = embeddings[idx]
        lats = lats[idx]
        lons = lons[idx]

    geo_dists = pdist(np.column_stack([lats, lons]), metric="euclidean")

    emb_dists_vec = pdist(embeddings, metric="cosine")

    rho, pval = spearmanr(geo_dists, emb_dists_vec)
    return rho, pval


def build_location_labels(df):
    if "zip" in df.columns:
        return df["zip"].astype(str).values

    if "latitude" in df.columns and "longitude" in df.columns:
        lats = df["latitude"].values
        lons = df["longitude"].values
        lat_bins = pd.cut(lats, bins=20, labels=False)
        lon_bins = pd.cut(lons, bins=20, labels=False)
        return np.array([f"{la}_{lo}" for la, lo in zip(lat_bins, lon_bins)])

    return None


def run_metrics(city):
    result = load_embeddings(city)
    if result is None:
        print(f"{city}: no embeddings found, skipping")
        return

    df, embeddings = result
    print(f"\n{city}: {len(df)} embeddings ({embeddings.shape[1]}d)")

    locations = build_location_labels(df)
    if locations is None:
        print("  No location labels available, skipping NMI and classifier")
        return

    nmi = compute_nmi(embeddings, locations)
    print(f"\n  NMI(T; L) = {nmi:.4f}")
    if nmi > 0.3:
        print("    → HIGH: embeddings strongly encode location")
    elif nmi > 0.1:
        print("    → MODERATE: some location information leaked")
    else:
        print("    → LOW: embeddings appear location-independent")

    acc, baseline = compute_location_classifier(embeddings, locations)
    if acc is not None:
        print(f"\n  Location classifier accuracy = {acc:.4f} (random baseline = {baseline:.4f})")
        ratio = acc / baseline
        print(f"    → {ratio:.1f}x above random")
        if acc > 0.85:
            print("    → CRITICAL: embeddings nearly deterministically encode location")
        elif acc > 0.5:
            print("    → SUBSTANTIAL: significant location signal in embeddings")
        else:
            print("    → MODERATE: some location signal present")

    lats = df["latitude"].values.astype(float)
    lons = df["longitude"].values.astype(float)
    valid = ~(np.isnan(lats) | np.isnan(lons))
    if valid.sum() > 100:
        rho, pval = compute_spatial_autocorrelation(
            embeddings[valid], lats[valid], lons[valid]
        )
        print(f"\n  Spatial autocorrelation (Spearman ρ) = {rho:.4f} (p = {pval:.2e})")
        if rho > 0.3:
            print("    → STRONG: nearby properties have similar embeddings")
        elif rho > 0.1:
            print("    → MODERATE: some spatial clustering in embedding space")
        else:
            print("    → WEAK: embeddings not spatially structured")

    acc_str = f"{acc:.4f}" if acc is not None else "N/A"
    print(f"\n  Summary: NMI={nmi:.4f}, Acc={acc_str}")


def main():
    cities = sys.argv[1:] if len(sys.argv) > 1 else list(CITIES.keys())
    for city in cities:
        run_metrics(city)


if __name__ == "__main__":
    main()
