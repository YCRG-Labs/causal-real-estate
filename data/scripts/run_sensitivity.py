import sys
import numpy as np
import pandas as pd
from config import PROCESSED_DIR, CITIES, EMBEDDING_MODEL, EMBEDDING_ALTERNATIVES, EMBEDDING_DIM
from causal_inference import (
    load_analysis_data,
    get_features_and_target,
    backdoor_adjustment,
    doubly_robust_estimation,
)
from confounding_metrics import (
    load_embeddings,
    compute_nmi,
    compute_location_classifier,
    build_location_labels,
)


def run_model_sensitivity(city, models=None):
    if models is None:
        models = {EMBEDDING_MODEL: EMBEDDING_DIM}
        models.update(EMBEDDING_ALTERNATIVES)

    print(f"\n{'#'*60}")
    print(f"EMBEDDING MODEL SENSITIVITY: {city.upper()}")
    print(f"{'#'*60}")

    results = []

    for model_name, dim in models.items():
        is_primary = model_name == EMBEDDING_MODEL
        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({dim}d) {'[PRIMARY]' if is_primary else '[ALTERNATIVE]'}")
        print(f"{'='*60}")

        emb_model_arg = None if is_primary else model_name
        data = load_analysis_data(city, embedding_model=emb_model_arg)
        if data is None:
            print(f"  No embeddings found for {model_name}, skipping")
            continue

        emb_df, parcels = data

        emb_cols = [f"emb_{i}" for i in range(dim)]
        available = [c for c in emb_cols if c in emb_df.columns]
        if not available:
            print(f"  No embedding columns found (expected emb_0..emb_{dim-1}), skipping")
            continue

        embeddings = emb_df[available].values

        locations = build_location_labels(emb_df)
        if locations is not None:
            nmi = compute_nmi(embeddings, locations)
            acc, baseline = compute_location_classifier(embeddings, locations, n_components=min(50, dim))
            print(f"  NMI: {nmi:.4f}")
            if acc is not None:
                print(f"  Location classifier: {acc:.4f} (random: {baseline:.4f}, ratio: {acc/baseline:.1f}x)")
        else:
            nmi = acc = baseline = None

        import causal_inference as ci
        old_dim = ci.EMBEDDING_DIM
        ci.EMBEDDING_DIM = dim

        feat = get_features_and_target(emb_df, parcels)
        if feat is not None:
            T, confounders, Y, meta = feat
            delta_r2 = backdoor_adjustment(T, confounders, Y, n_pca=min(50, dim))
            dr_effect, dr_ci, _ = doubly_robust_estimation(T, confounders, Y, n_pca=min(50, dim))
        else:
            delta_r2 = dr_effect = None
            dr_ci = (None, None)

        ci.EMBEDDING_DIM = old_dim

        results.append({
            "model": model_name,
            "dim": dim,
            "nmi": nmi,
            "loc_acc": acc,
            "loc_baseline": baseline,
            "delta_r2": delta_r2,
            "dr_ate": dr_effect,
            "dr_ci_low": dr_ci[0],
            "dr_ci_high": dr_ci[1],
        })

    print(f"\n{'='*60}")
    print("SENSITIVITY SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Dim':>4} {'NMI':>6} {'Loc Acc':>8} {'ΔR²':>7} {'DR ATE':>8} {'CI':>20}")
    print("-" * 85)
    for r in results:
        nmi_s = f"{r['nmi']:.3f}" if r['nmi'] is not None else "N/A"
        acc_s = f"{r['loc_acc']:.3f}" if r['loc_acc'] is not None else "N/A"
        dr2_s = f"{r['delta_r2']:.3f}" if r['delta_r2'] is not None else "N/A"
        ate_s = f"{r['dr_ate']:.4f}" if r['dr_ate'] is not None else "N/A"
        ci_s = f"[{r['dr_ci_low']:.3f}, {r['dr_ci_high']:.3f}]" if r['dr_ci_low'] is not None else "N/A"
        print(f"{r['model']:<25} {r['dim']:>4} {nmi_s:>6} {acc_s:>8} {dr2_s:>7} {ate_s:>8} {ci_s:>20}")

    if len(results) >= 2:
        ates = [r["dr_ate"] for r in results if r["dr_ate"] is not None]
        nmis = [r["nmi"] for r in results if r["nmi"] is not None]
        if len(ates) >= 2:
            ate_range = max(ates) - min(ates)
            print(f"\n  ATE range across models: {ate_range:.4f}")
            if all(r["dr_ci_low"] is not None and r["dr_ci_low"] <= 0 <= r["dr_ci_high"]
                   for r in results if r["dr_ci_low"] is not None):
                print("  All CIs contain zero → zero-effect finding is ROBUST across embedding models")
            else:
                print("  WARNING: Some CIs exclude zero → findings may be model-dependent")

        if len(nmis) >= 2:
            nmi_range = max(nmis) - min(nmis)
            print(f"  NMI range across models: {nmi_range:.4f}")

    return results


def main():
    cities = sys.argv[1:] if len(sys.argv) > 1 else list(CITIES.keys())
    for city in cities:
        run_model_sensitivity(city)


if __name__ == "__main__":
    main()
