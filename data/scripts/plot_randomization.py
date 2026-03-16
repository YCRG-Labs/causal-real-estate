import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from config import PROCESSED_DIR, EMBEDDING_DIM


def run_and_plot(city, n_permutations=100):
    emb_df = pd.read_parquet(PROCESSED_DIR / f"{city}_embeddings.parquet")
    emb_cols = [f"emb_{i}" for i in range(EMBEDDING_DIM)]
    available = [c for c in emb_cols if c in emb_df.columns]

    emb_df["price"] = pd.to_numeric(emb_df["price"], errors="coerce")
    emb_df = emb_df.dropna(subset=["price", "zip"])
    emb_df = emb_df[emb_df["price"] > 0]

    T = emb_df[available].values
    Y = np.log(emb_df["price"].values.astype(float))

    zips = emb_df["zip"].astype(float).astype(int).astype(str)
    le = LabelEncoder()
    zip_encoded = le.fit_transform(zips)
    n_bins = min(len(le.classes_), 20)
    L = np.zeros((len(T), n_bins))
    for i, z in enumerate(zip_encoded):
        L[i, z % n_bins] = 1.0

    n_pca = min(30, T.shape[1], len(T) - 2)
    pca = PCA(n_components=n_pca, random_state=42)
    T_pca = pca.fit_transform(T)
    scaler = StandardScaler()
    T_s = scaler.fit_transform(T_pca)

    features_orig = np.hstack([T_s, L])
    n = len(Y)
    train_n = int(n * 0.7)
    idx = np.random.RandomState(42).permutation(n)
    train_idx, test_idx = idx[:train_n], idx[train_n:]

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    model.fit(features_orig[train_idx], Y[train_idx])
    r2_original = model.score(features_orig[test_idx], Y[test_idx])

    r2_permuted = []
    for p in range(n_permutations):
        perm = np.random.RandomState(p).permutation(n)
        L_perm = L[perm]
        features_perm = np.hstack([T_s, L_perm])

        model_p = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=p,
        )
        model_p.fit(features_perm[train_idx], Y[train_idx])
        r2_permuted.append(model_p.score(features_perm[test_idx], Y[test_idx]))

    r2_permuted = np.array(r2_permuted)
    p_value = np.mean(r2_permuted >= r2_original)

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    fig, ax = plt.subplots(figsize=(5.5, 4))

    ax.hist(r2_permuted, bins=20, color="#abd9e9", edgecolor="#2c7bb6",
            linewidth=0.6, alpha=0.85, label="Permuted locations")

    ax.axvline(x=r2_original, color="#d7191c", linewidth=1.5, linestyle="-",
               label=f"Original ($R^2 = {r2_original:.4f}$)")

    ax.axvline(x=np.mean(r2_permuted), color="#333333", linewidth=1.0, linestyle=":",
               label=f"Permuted mean ($R^2 = {np.mean(r2_permuted):.4f}$)")

    pass

    ax.set_xlabel(r"$R^2$ on held-out test set")
    ax.set_ylabel("Count")
    ax.set_title(f"Randomization Test ({city.upper()})", fontweight="normal")
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    fig.savefig(PROCESSED_DIR / f"{city}_randomization.png", dpi=600)
    fig.savefig(PROCESSED_DIR / f"{city}_randomization.pdf")
    print(f"Saved to {PROCESSED_DIR / f'{city}_randomization.pdf'}")
    print(f"Original R2: {r2_original:.4f}, Permuted mean: {np.mean(r2_permuted):.4f}, p={p_value:.2f}")


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "nyc"
    run_and_plot(city)


if __name__ == "__main__":
    main()
