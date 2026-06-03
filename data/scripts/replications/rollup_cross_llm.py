"""Cross-LLM concordance rollup: Qwen 2.5 32B vs Llama 3.3 70B counterfactual rewrites.

Reads per-city JSONs from both rewriting models, computes the per-city
Spearman / Pearson rank correlations on the headline estimands (DML
theta, natural direct effect, total effect), reports sign-agreement
counts, and writes a CSV that §6.4 of the JBES revision references.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
CITIES = ["boston", "nyc", "sf", "dc", "philadelphia", "chicago",
          "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]


def _load_city(d: Path, city: str) -> dict | None:
    p = d / f"{city}.json"
    if not p.exists():
        return None
    j = json.loads(p.read_text())
    return {
        "city": city,
        "dml_theta": j.get("dml", {}).get("theta"),
        "dml_se": j.get("dml", {}).get("se"),
        "nde": j.get("natural_direct_effect_style_stripped", {}).get("mean_delta_logprice"),
        "te": j.get("total_effect_style_swap", {}).get("mean_delta_logprice"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen_dir",
                    default=str(REPO / "results" / "counterfactual"))
    ap.add_argument("--llama_dir",
                    default=str(REPO / "results" / "counterfactual_llama70b"))
    ap.add_argument("--out",
                    default=str(REPO / "results" / "replications" /
                                 "cross_llm_concordance.csv"))
    args = ap.parse_args()

    qwen = Path(args.qwen_dir)
    llama = Path(args.llama_dir)
    rows = []
    for city in CITIES:
        q = _load_city(qwen, city)
        l = _load_city(llama, city)
        if q is None or l is None:
            print(f"  [skip] {city}: qwen={q is not None}, llama={l is not None}")
            continue
        rows.append({
            "city": city,
            "qwen_dml_theta": q["dml_theta"], "llama_dml_theta": l["dml_theta"],
            "qwen_nde": q["nde"], "llama_nde": l["nde"],
            "qwen_te": q["te"], "llama_te": l["te"],
        })
    df = pd.DataFrame(rows)
    print(f"\n=== Per-city Qwen vs Llama ({len(df)} cities) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    print("\n=== Concordance statistics ===")
    summary = []
    for col in ["dml_theta", "nde", "te"]:
        q = df[f"qwen_{col}"].values
        l = df[f"llama_{col}"].values
        sp_r, sp_p = stats.spearmanr(q, l)
        pe_r, pe_p = stats.pearsonr(q, l)
        sign_agree = int(np.sum(np.sign(q) == np.sign(l)))
        excl_zero_q = sum(abs(qi) > 0 for qi in q)
        summary.append({
            "estimand": col,
            "spearman_rho": sp_r, "spearman_p": sp_p,
            "pearson_r": pe_r, "pearson_p": pe_p,
            "sign_agree": sign_agree, "n_cities": len(df),
            "qwen_pooled_mean": float(np.mean(q)),
            "llama_pooled_mean": float(np.mean(l)),
        })
        print(f"  {col}: Spearman={sp_r:+.3f} (p={sp_p:.3f}), "
              f"Pearson={pe_r:+.3f} (p={pe_p:.3f}), "
              f"sign_agree {sign_agree}/{len(df)}, "
              f"means qwen={np.mean(q):+.4f} llama={np.mean(l):+.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    print(f"\nPer-city CSV -> {out_path}")
    print(f"Summary CSV   -> {summary_path}")


if __name__ == "__main__":
    sys.exit(main())
