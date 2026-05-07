"""JBES-style figures from the saved coverage / power tables.

Produces two PNGs in `results/simulation/`:

  power_curves.png    -- one panel per N, x-axis = beta_direct (or
                         effect-size eta), y-axis = empirical power,
                         one line per estimator. Standard JBES "size-power"
                         display (Wager & Athey 2018).
  coverage_bars.png   -- grouped bar chart, x-axis = estimator, one bar per
                         (N, dgp) cell, y-axis = empirical coverage,
                         horizontal reference at 0.95 with shaded
                         95%-binomial-acceptance band [0.93, 0.97]
                         (the band Belloni et al. 2014 bold against).

Both figures read the CSVs run_simulation.py wrote; no recomputation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parents[3] / "results" / "simulation"
)

ESTIMATOR_ORDER = ["DR", "DML", "Adversarial", "Randomization"]
ESTIMATOR_COLORS = {
    "DR": "#1f77b4",
    "DML": "#d62728",
    "Adversarial": "#2ca02c",
    "Randomization": "#9467bd",
}


def _eta_from_dgp(dgp: str) -> float:
    """Map dgp label to a numeric effect-size eta."""
    if dgp == "scm0":
        return 0.0
    # 'scm1_0.05' -> 0.05
    return float(dgp.split("_")[1])


def plot_power_curves(power_df: pd.DataFrame, out_path: Path) -> None:
    Ns = sorted(power_df["N"].unique())
    n_panels = len(Ns)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(5 * n_panels, 4.2),
        sharey=True, squeeze=False,
    )
    axes = axes[0]

    power_df = power_df.assign(eta=power_df["dgp"].map(_eta_from_dgp))

    for ax, N in zip(axes, Ns):
        sub = power_df[power_df["N"] == N]
        for est in ESTIMATOR_ORDER:
            line = sub[sub["estimator"] == est].sort_values("eta")
            if line.empty:
                continue
            ax.plot(
                line["eta"], line["power"],
                marker="o", label=est, color=ESTIMATOR_COLORS.get(est, None),
                linewidth=2, markersize=6,
            )
        ax.axhline(0.05, color="gray", linestyle=":", linewidth=1,
                   label="size = 0.05" if ax is axes[0] else None)
        ax.axhline(0.80, color="gray", linestyle="--", linewidth=1,
                   label="80% power" if ax is axes[0] else None)
        ax.set_xlabel("effect size  Var(direct) / Var(Y)")
        ax.set_title(f"N = {N}")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("empirical power")
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("Power curves: rejection of H0: theta = 0", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_coverage_bars(coverage_df: pd.DataFrame, out_path: Path) -> None:
    coverage_df = coverage_df.assign(eta=coverage_df["dgp"].map(_eta_from_dgp))
    Ns = sorted(coverage_df["N"].unique())
    etas = sorted(coverage_df["eta"].unique())
    estimators_present = [e for e in ESTIMATOR_ORDER
                          if e in coverage_df["estimator"].unique()]
    n_groups = len(estimators_present)
    n_bars = len(Ns) * len(etas)

    fig, ax = plt.subplots(figsize=(max(8, 1.4 * n_groups * len(Ns)), 5))

    bar_width = 0.8 / max(n_bars, 1)
    x = np.arange(n_groups)
    legend_labels = []

    cmap = plt.colormaps.get_cmap("viridis")
    color_idx = 0
    for N in Ns:
        for eta in etas:
            heights = []
            for est in estimators_present:
                sel = coverage_df[
                    (coverage_df["estimator"] == est)
                    & (coverage_df["N"] == N)
                    & (coverage_df["eta"] == eta)
                ]
                heights.append(sel["coverage"].mean() if not sel.empty else np.nan)
            offset = (color_idx - n_bars / 2 + 0.5) * bar_width
            color = cmap(color_idx / max(n_bars - 1, 1))
            ax.bar(x + offset, heights, width=bar_width, color=color,
                   label=f"N={N}, eta={eta:.2f}")
            color_idx += 1

    ax.axhspan(0.93, 0.97, color="lightgreen", alpha=0.3, zorder=0,
               label="95% binomial band [0.93, 0.97]")
    ax.axhline(0.95, color="black", linewidth=1, linestyle="--", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(estimators_present)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("empirical 95% CI coverage")
    ax.set_title("Coverage of nominal 95% CIs across estimator x N x effect-size cells")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--out_dir", type=Path, default=None)
    args = ap.parse_args()
    in_dir = args.in_dir
    out_dir = args.out_dir or in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cov_path = in_dir / "coverage_table.csv"
    pow_path = in_dir / "power_table.csv"
    if not cov_path.exists() or not pow_path.exists():
        raise SystemExit(
            f"Missing input CSVs in {in_dir}. Run run_simulation.py first."
        )
    coverage_df = pd.read_csv(cov_path)
    power_df = pd.read_csv(pow_path)

    plot_power_curves(power_df, out_dir / "power_curves.png")
    plot_coverage_bars(coverage_df, out_dir / "coverage_bars.png")


if __name__ == "__main__":
    main()
