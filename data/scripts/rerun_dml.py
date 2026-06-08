"""
Re-run the DML headline (PC1 partial-linear effect) for one or more cities on
the post-dedup, property-level-geocoded data, and emit a clean comparison row
per city.

Skips the slower pipeline pieces (adversarial deconfounding, CATE, randomization)
that don't bear on the immediate question of whether the dedup + geocode fix
changes the headline numbers.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _silence
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from causal_inference import (
    load_analysis_data,
    get_features_and_target,
    backdoor_adjustment,
    dml_continuous_treatment,
    doubly_robust_estimation,
)


def rerun(city, ci_method="bootstrap", n_boot=1000):
    t0 = time.time()
    res = load_analysis_data(city)
    if res is None:
        print(f"[{city}] no data, skipping")
        return None
    emb_df, parcels = res
    data = get_features_and_target(emb_df, parcels)
    if data is None:
        print(f"[{city}] no price target, skipping")
        return None
    T, conf, Y, meta = data
    n, p_t = T.shape
    p_c = conf.shape[1]
    print(f"\n[{city}] n={n}  text_dim={p_t}  confounders={p_c}  "
          f"rich={meta['has_rich_confounders']}")

    delta_r2 = backdoor_adjustment(T, conf, Y)

    dr_ate, dr_ci, dr_extras = doubly_robust_estimation(T, conf, Y)

    dml = dml_continuous_treatment(T, conf, Y, ci_method=ci_method,
                                   n_boot=n_boot, verbose=True)

    elapsed = time.time() - t0
    out = {
        "city": city,
        "n": n,
        "text_dim": p_t,
        "n_confounders": p_c,
        "has_rich_confounders": bool(meta["has_rich_confounders"]),
        "backdoor_delta_r2": float(delta_r2),
        "dr_ate": float(dr_ate),
        "dr_ci_low": float(dr_ci[0]),
        "dr_ci_high": float(dr_ci[1]),
        "dr_if_se": float(dr_extras["if_se"]),
        "dml_theta": float(dml["theta"]) if dml else None,
        "dml_se": float(dml["se"]) if dml else None,
        "dml_ci_low": float(dml["ci"][0]) if dml else None,
        "dml_ci_high": float(dml["ci"][1]) if dml else None,
        "dml_mde": float(dml["mde"]) if dml else None,
        "ci_method": ci_method,
        "elapsed_sec": round(elapsed, 1),
    }
    return out


def main():
    cities = sys.argv[1:] if len(sys.argv) > 1 else ["sf", "nyc", "boston"]
    ci_method = "bootstrap" if "--boot" in cities else "if"
    n_boot = 1000
    if "--fast" in cities:
        n_boot = 200
        cities = [c for c in cities if c != "--fast"]
    cities = [c for c in cities if c != "--boot" and not c.startswith("--")]
    if not cities:
        cities = ["sf", "nyc", "boston"]
    print(f"Cities: {cities}  ci_method={ci_method}  n_boot={n_boot}\n")

    results = []
    for c in cities:
        r = rerun(c, ci_method=ci_method, n_boot=n_boot)
        if r is not None:
            results.append(r)

    print(f"\n{'='*60}\nCOMPARISON\n{'='*60}")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    out = Path("results/rerun_dml_postdedup.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
