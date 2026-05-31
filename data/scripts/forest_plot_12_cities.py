"""Cleveland forest plot of per-city DML theta with IF and bootstrap CIs.

Parses /tmp/dml_b1000_v2.log (or the path passed via --log) which contains
one block per city of the form:

    === <city> ===
    ...
    <city> n text_dim n_confounders has_rich_confounders learner mode drop_crime B theta se_if se_cb ci_low ci_high mde t_pivot elapsed_sec load_sec boot_sec ...

Extracts (city, n, theta, se_if, se_cb, ci_low, ci_high) for each city and
renders a Cleveland forest plot:
  - one row per city, sorted by theta
  - thick whisker = +/- 1.96 * se_if (IF-based CI, the underestimate)
  - thin whisker = se_cb-based 95% CI from the bootstrap (the wider, valid one)
  - point estimate as filled circle
  - vertical dashed line at theta = 0
  - bottom row: random-effects pooled estimate (DerSimonian-Laird)

The two-whisker construction is the visual analogue of the JBES finding: in
moderate-n cities the IF SE undercovers, and the bootstrap whisker extends
well beyond the IF whisker. In well-conditioned cities the two whiskers
nearly coincide.

Output: paper/figures/forest_12cities.pdf
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

CITY_DISPLAY = {
    "boston": "Boston", "nyc": "New York", "sf": "San Francisco",
    "dc": "Washington DC", "philadelphia": "Philadelphia",
    "chicago": "Chicago", "seattle": "Seattle", "denver": "Denver",
    "atlanta": "Atlanta", "portland": "Portland", "phoenix": "Phoenix",
    "dallas": "Dallas",
}
ORDER = list(CITY_DISPLAY.keys())


def parse_log(log_path: Path) -> pd.DataFrame:
    text = log_path.read_text()
    pattern = re.compile(
        r"^\s*(?P<city>[a-z]+)\s+(?P<n>\d+)\s+\d+\s+\d+\s+(?:True|False)\s+\w+\s+\w+\s+(?:True|False)\s+"
        r"(?P<B>\d+)\s+(?P<theta>-?\d+\.\d+)\s+(?P<se_if>-?\d+\.\d+)\s+(?P<se_cb>-?\d+\.\d+)\s+"
        r"(?P<ci_low>-?\d+\.\d+)\s+(?P<ci_high>-?\d+\.\d+)",
        flags=re.MULTILINE,
    )
    rows = []
    for m in pattern.finditer(text):
        d = m.groupdict()
        if d["city"] not in CITY_DISPLAY:
            continue
        rows.append({
            "city": d["city"], "n": int(d["n"]), "B": int(d["B"]),
            "theta": float(d["theta"]), "se_if": float(d["se_if"]),
            "se_cb": float(d["se_cb"]),
            "ci_low": float(d["ci_low"]), "ci_high": float(d["ci_high"]),
        })
    if not rows:
        raise RuntimeError(f"no city blocks parsed from {log_path}")
    df = pd.DataFrame(rows).drop_duplicates(subset=["city"], keep="last")
    return df.set_index("city").reindex([c for c in ORDER if c in df["city"].tolist() or c in df.index]).reset_index()


def random_effects_pool(theta: np.ndarray, se: np.ndarray) -> dict:
    w_fe = 1.0 / np.maximum(se ** 2, 1e-12)
    theta_fe = float(np.sum(w_fe * theta) / np.sum(w_fe))
    Q = float(np.sum(w_fe * (theta - theta_fe) ** 2))
    k = len(theta)
    if k <= 1:
        tau2 = 0.0
    else:
        c = float(np.sum(w_fe) - np.sum(w_fe ** 2) / np.sum(w_fe))
        tau2 = max(0.0, (Q - (k - 1)) / max(c, 1e-12))
    w_re = 1.0 / (se ** 2 + tau2)
    theta_re = float(np.sum(w_re * theta) / np.sum(w_re))
    var_re = 1.0 / np.sum(w_re)
    se_re = float(np.sqrt(var_re))
    I2 = max(0.0, (Q - (k - 1)) / max(Q, 1e-12)) * 100.0
    return {"theta_re": theta_re, "se_re": se_re, "tau2": tau2,
            "Q": Q, "df": k - 1, "I2_pct": I2,
            "theta_fe": theta_fe, "se_fe": float(np.sqrt(1.0 / np.sum(w_fe)))}


def plot_forest(df: pd.DataFrame, out_path: Path, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm",
        "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
        "axes.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    df_sorted = df.sort_values("theta").reset_index(drop=True)
    pooled_if = random_effects_pool(df_sorted["theta"].values, df_sorted["se_if"].values)
    pooled_cb = random_effects_pool(df_sorted["theta"].values, df_sorted["se_cb"].values)

    n_rows = len(df_sorted) + 1
    fig, ax = plt.subplots(figsize=(7.0, 0.36 * n_rows + 1.2))

    y_positions = np.arange(n_rows)[::-1]
    pooled_y = y_positions[-1]
    city_y = y_positions[:-1]

    color_if = "#2c7bb6"
    color_cb = "#d7191c"
    color_pt = "#1a1a1a"

    for yi, (_, row) in zip(city_y, df_sorted.iterrows()):
        cb_lo = row["theta"] - 1.96 * row["se_cb"]
        cb_hi = row["theta"] + 1.96 * row["se_cb"]
        if_lo = row["theta"] - 1.96 * row["se_if"]
        if_hi = row["theta"] + 1.96 * row["se_if"]
        ax.hlines(yi, cb_lo, cb_hi, colors=color_cb, lw=1.0, alpha=0.85)
        ax.hlines(yi, if_lo, if_hi, colors=color_if, lw=2.6, alpha=0.95)
        ax.plot(row["theta"], yi, marker="o", ms=4.4, mfc=color_pt,
                mec="white", mew=0.5, zorder=5)

    re_lo_cb = pooled_cb["theta_re"] - 1.96 * pooled_cb["se_re"]
    re_hi_cb = pooled_cb["theta_re"] + 1.96 * pooled_cb["se_re"]
    re_lo_if = pooled_if["theta_re"] - 1.96 * pooled_if["se_re"]
    re_hi_if = pooled_if["theta_re"] + 1.96 * pooled_if["se_re"]
    ax.hlines(pooled_y, re_lo_cb, re_hi_cb, colors=color_cb, lw=1.4)
    ax.hlines(pooled_y, re_lo_if, re_hi_if, colors=color_if, lw=3.2)
    ax.plot(pooled_cb["theta_re"], pooled_y, marker="D", ms=7,
            mfc=color_pt, mec="white", mew=0.7, zorder=6)

    ax.axvline(0, color="#888888", lw=0.7, ls="--", zorder=0)

    labels = [f"{CITY_DISPLAY.get(c,c)} (n={n})"
              for c, n in zip(df_sorted["city"], df_sorted["n"])]
    pooled_label = (f"Pooled RE (k={len(df_sorted)}, "
                    fr"$\tau^2$={pooled_cb['tau2']:.3f}, "
                    fr"$I^2$={pooled_cb['I2_pct']:.0f}%)")
    ax.set_yticks(list(city_y) + [pooled_y])
    ax.set_yticklabels(labels + [pooled_label])

    ax.set_xlabel(r"DML $\hat\theta$  (log-price effect of text PC1)")
    if title:
        ax.set_title(title, loc="left", fontweight="normal", pad=8)

    legend_handles = [
        plt.Line2D([0], [0], color=color_if, lw=2.6, label=r"$\pm 1.96\,\widehat{\mathrm{SE}}_{\mathrm{IF}}$"),
        plt.Line2D([0], [0], color=color_cb, lw=1.0, label=r"$\pm 1.96\,\widehat{\mathrm{SE}}_{\mathrm{boot}}$ (B=1000)"),
        plt.Line2D([0], [0], color=color_pt, marker="o", lw=0, ms=4.4, label=r"$\hat\theta_{\mathrm{city}}$"),
        plt.Line2D([0], [0], color=color_pt, marker="D", lw=0, ms=7, label=r"$\hat\theta_{\mathrm{pooled,RE}}$"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
              frameon=True, framealpha=0.95, edgecolor="#cccccc")

    ax.set_xlim(min(re_lo_cb, df_sorted["theta"].min() - 1.96 * df_sorted["se_cb"].max()) - 0.05,
                max(re_hi_cb, df_sorted["theta"].max() + 1.96 * df_sorted["se_cb"].max()) + 0.05)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"forest plot -> {out_path}")
    print(f"  RE pooled (using se_boot): theta={pooled_cb['theta_re']:+.4f} (se={pooled_cb['se_re']:.4f})  tau2={pooled_cb['tau2']:.4f}  I2={pooled_cb['I2_pct']:.1f}%")
    print(f"  FE pooled (using se_boot): theta={pooled_cb['theta_fe']:+.4f} (se={pooled_cb['se_fe']:.4f})")
    print(f"  Q (heterogeneity): {pooled_cb['Q']:.2f} on {pooled_cb['df']} df")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="/tmp/dml_b1000_v2.log")
    ap.add_argument("--out", default=str(REPO_ROOT / "paper" / "figures" / "forest_12cities.pdf"))
    ap.add_argument("--title", default="DML text-effect by city (residential filter, B=1000)")
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"log not found: {log_path}", file=sys.stderr); return 1
    df = parse_log(log_path)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    out_path = Path(args.out)
    plot_forest(df, out_path, title=args.title)
    csv_out = out_path.with_suffix(".csv")
    df.to_csv(csv_out, index=False)
    print(f"data table -> {csv_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
