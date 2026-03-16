import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from umap import UMAP
from config import PROCESSED_DIR, EMBEDDING_DIM


def plot_umap(city):
    emb_df = pd.read_parquet(PROCESSED_DIR / f"{city}_embeddings.parquet")
    emb_cols = [f"emb_{i}" for i in range(EMBEDDING_DIM)]
    available = [c for c in emb_cols if c in emb_df.columns]
    T = emb_df[available].values

    if "zip" in emb_df.columns:
        labels = emb_df["zip"].fillna(0).astype(float).astype(int).astype(str).values
    else:
        return

    pca = PCA(n_components=min(30, T.shape[1], len(T) - 2), random_state=42)
    T_pca = pca.fit_transform(T)

    reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.3, random_state=42)
    coords = reducer.fit_transform(T_pca)

    unique_labels = sorted(set(labels))
    n_labels = len(unique_labels)

    if n_labels <= 10:
        cmap = plt.cm.tab10
    elif n_labels <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.nipy_spectral

    label_to_int = {l: i for i, l in enumerate(unique_labels)}
    colors = [label_to_int[l] for l in labels]

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

    fig, ax = plt.subplots(figsize=(6, 5))

    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=colors, cmap=cmap,
        s=18, alpha=0.7, edgecolors="white", linewidths=0.3,
        vmin=0, vmax=max(n_labels - 1, 1),
    )

    if n_labels <= 12:
        handles = []
        for label in unique_labels:
            idx = label_to_int[label]
            color = cmap(idx / max(n_labels - 1, 1))
            handles.append(plt.scatter([], [], c=[color], s=20, label=f"Zip {label}"))
        ax.legend(handles=handles, frameon=False, fontsize=8, loc="best",
                  markerscale=1.2, handletextpad=0.3)
    else:
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Zip code", fontsize=10)
        cbar.outline.set_linewidth(0.4)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"Property Embeddings by Location ({city.upper()})", fontweight="normal")
    ax.tick_params(length=3, width=0.5)

    fig.savefig(PROCESSED_DIR / f"{city}_umap.png", dpi=600)
    fig.savefig(PROCESSED_DIR / f"{city}_umap.pdf")
    print(f"Saved UMAP figure to {PROCESSED_DIR / f'{city}_umap.pdf'}")


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "sf"
    plot_umap(city)


if __name__ == "__main__":
    main()
