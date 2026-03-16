import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import PROCESSED_DIR


def plot_corr_vs_causal():
    cities = ["Boston", "New York", "San Francisco"]
    delta_r2 = [0.859, 0.884, 0.589]
    ate = [-0.001, -0.138, -0.067]
    ci_low = [-3.24, -3.65, -1.13]
    ci_high = [2.63, 2.87, 1.39]

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    x = np.arange(len(cities))
    width = 0.55

    bars = ax1.bar(x, delta_r2, width, color="#2c7bb6", edgecolor="white", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cities, fontsize=10)
    ax1.set_ylabel(r"$\Delta R^2$ (text beyond location)")
    ax1.set_title("(a) Correlational Signal", fontweight="normal")
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=0, color="black", linewidth=0.4)

    for bar, val in zip(bars, delta_r2):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    bars2 = ax2.bar(x, ate, width, color="#d7191c", edgecolor="white", linewidth=0.5)

    for i in range(len(cities)):
        ax2.plot([x[i], x[i]], [ci_low[i], ci_high[i]],
                 color="#333333", linewidth=1.2, zorder=5)
        ax2.plot([x[i] - 0.08, x[i] + 0.08], [ci_low[i], ci_low[i]],
                 color="#333333", linewidth=1.2, zorder=5)
        ax2.plot([x[i] - 0.08, x[i] + 0.08], [ci_high[i], ci_high[i]],
                 color="#333333", linewidth=1.2, zorder=5)

    ax2.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
    ax2.set_xticks(x)
    ax2.set_xticklabels(cities, fontsize=10)
    ax2.set_ylabel("Causal ATE (log-price)")
    ax2.set_title("(b) Causal Effect (DR Estimate)", fontweight="normal")

    for bar, val in zip(bars2, ate):
        y_pos = val - 0.3 if val < 0 else val + 0.1
        ax2.text(bar.get_x() + bar.get_width() / 2, y_pos,
                 f"{val:.3f}", ha="center", va="top" if val < 0 else "bottom", fontsize=9)

    plt.tight_layout(w_pad=3)

    fig.savefig(PROCESSED_DIR / f"correlation_vs_causation.png", dpi=600)
    fig.savefig(PROCESSED_DIR / f"correlation_vs_causation.pdf")
    print(f"Saved to {PROCESSED_DIR / 'correlation_vs_causation.pdf'}")


if __name__ == "__main__":
    plot_corr_vs_causal()
