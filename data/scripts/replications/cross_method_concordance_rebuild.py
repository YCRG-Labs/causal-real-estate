"""Cross-method concordance (Shen theta vs Baur theta) on the content-keyed
rebuild, full 12-city and leave-two-out (chicago, philadelphia dropped).

Reads:
  results/replications/baur_pooled_pca_rebuild_keyed/baur_pooled_pca_table.csv
  results/replications/shen_12city_rebuild_workers1/shen_{city}.json

Writes:
  results/replications/cross_method_concordance_rebuild.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "results" / "replications"

CITY_ORDER = ["boston", "nyc", "sf", "dc", "philadelphia", "chicago",
              "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]
DROP_FOR_LEAVE_TWO_OUT = {"chicago", "philadelphia"}


def concordance_correlation(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mx, my = x.mean(), y.mean()
    sx2, sy2 = x.var(ddof=0), y.var(ddof=0)
    cov = np.mean((x - mx) * (y - my))
    return float(2 * cov / (sx2 + sy2 + (mx - my) ** 2))


def _load_baur(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).set_index("city")
    return df[["dml_theta", "dml_se"]].rename(
        columns={"dml_theta": "baur_theta", "dml_se": "baur_se"})


def _load_shen(shen_dir: Path) -> pd.DataFrame:
    rows = []
    for city in CITY_ORDER:
        p = shen_dir / f"shen_{city}.json"
        d = json.loads(p.read_text())
        dml = d["dml_uniqueness"]
        rows.append({"city": city, "shen_theta": dml["theta"], "shen_se": dml["se"]})
    return pd.DataFrame(rows).set_index("city")


def _stats_block(df: pd.DataFrame) -> dict:
    x = df["shen_theta"].to_numpy(float)
    y = df["baur_theta"].to_numpy(float)
    pe = stats.pearsonr(x, y)
    sp = stats.spearmanr(x, y)
    ccc = concordance_correlation(x, y)
    return {
        "n_cities": int(len(df)),
        "cities": df.index.tolist(),
        "pearson_r": float(pe.statistic), "pearson_p": float(pe.pvalue),
        "spearman_rho": float(sp.statistic), "spearman_p": float(sp.pvalue),
        "lin_ccc": ccc,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--baur", default=str(RESULTS / "baur_pooled_pca_rebuild_keyed"
                                           / "baur_pooled_pca_table.csv"))
    ap.add_argument("--shen_dir", default=str(RESULTS / "shen_12city_rebuild_workers1"))
    ap.add_argument("--out", default=str(RESULTS / "cross_method_concordance_rebuild.json"))
    args = ap.parse_args()

    baur = _load_baur(Path(args.baur))
    shen = _load_shen(Path(args.shen_dir))
    df = shen.join(baur, how="inner").reindex(CITY_ORDER)
    assert df.notna().all().all(), f"missing rows after join:\n{df}"

    full = _stats_block(df)
    leave_two_out = _stats_block(df.drop(index=list(DROP_FOR_LEAVE_TWO_OUT)))

    out = {
        "note": ("Concordance between per-market Shen (Doc2Vec uniqueness) "
                 "theta and Baur (content-keyed pooled-PCA) theta across "
                 "the 12 metros. leave_two_out drops chicago and "
                 "philadelphia, the two highest-leverage / largest-|theta| "
                 "markets for both methods."),
        "full_12city": full,
        "leave_two_out_no_chicago_philadelphia": leave_two_out,
        "per_city": df.reset_index().to_dict(orient="records"),
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
