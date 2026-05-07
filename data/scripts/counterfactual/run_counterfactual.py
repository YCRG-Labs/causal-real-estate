"""End-to-end counterfactual LLM rewrite pipeline.

For each SF listing:
  1. Extract numeric slots
  2. Generate 4 variants (3 style-swap to alternative submarkets + 1 stripped)
  3. Validate (slot, perplexity, attribute-classifier)
  4. Re-encode each surviving rewrite via the SAME sentence-transformers model
     used in the production DML pipeline (all-mpnet-base-v2)
  5. Project rewrite embedding through the trained PCA, standardize, push
     through the DML residual model: predicted_log_price = model_y(conf) +
     theta * pc1_resid_rewrite
  6. Aggregate per-arm: NDE on style-stripped (avg Δ log-price when style is
     stripped), Total Effect on style-swap (avg Δ log-price toward implied
     submarket price level)

Usage:
  python run_counterfactual.py --city sf --n_listings 25 --out results/counterfactual/sf.json
  python run_counterfactual.py --city sf --n_listings 500 --out results/counterfactual/sf_full.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Make sibling scripts importable as flat modules to mirror negative_controls.py.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                # counterfactual/ for local imports
sys.path.insert(0, str(HERE.parent))         # scripts/ for causal_inference, config

from causal_inference import (
    get_features_and_target,
    load_analysis_data,
)
from config import EMBEDDING_DIM, EMBEDDING_MODEL, PROCESSED_DIR, RAW_DIR

from generator import GenerationResult, MockGenerator, make_generator
from prompts import SUBMARKET_HINTS, style_stripped_prompt, style_swap_prompt
from slot_extractor import extract_slots
from validator import (
    fit_zip_classifier,
    perplexity,
    reset_caches,
    validate_rewrite,
)


# ---------- data structures --------------------------------------------------

@dataclass
class RewriteRecord:
    arm: str                       # "style_swap:<submarket>" or "style_stripped"
    target_submarket: Optional[str]
    target_zip: Optional[int]
    rewritten_text: str
    used_mock: bool
    validation: dict
    pred_logprice_baseline: float
    pred_logprice_rewrite: float
    delta_logprice: float


@dataclass
class ListingRecord:
    listing_idx: int
    address: str
    zip: int
    original_text: str
    slots: dict
    rewrites: list[RewriteRecord] = field(default_factory=list)


# ---------- DML re-fit (no print spam) ---------------------------------------

@dataclass
class DMLArtifacts:
    pca: PCA
    pc1_mean: float
    pc1_std: float
    model_y_full: GradientBoostingRegressor
    conf_scaler: StandardScaler
    theta: float
    se: float
    pc1_resid_train: np.ndarray
    pc1_norm_train: np.ndarray


def fit_dml_artifacts(
    T: np.ndarray, confounders: np.ndarray, Y: np.ndarray,
    n_pca: int = 50, k_folds: int = 5,
) -> DMLArtifacts:
    """Refit the same DML pipeline as causal_inference.dml_continuous_treatment
    but RETAIN the fitted PCA, conf scaler, and a full-data model_y so we can
    score new rewrites."""
    n_pca = min(n_pca, T.shape[1], T.shape[0] - 1)
    pca = PCA(n_components=n_pca, random_state=42)
    T_pca = pca.fit_transform(T)
    pc1 = T_pca[:, 0]
    pc1_mean = float(pc1.mean())
    pc1_std = float(pc1.std()) if pc1.std() > 0 else 1.0
    pc1_norm = (pc1 - pc1_mean) / pc1_std

    conf_scaler = StandardScaler()
    conf_s = conf_scaler.fit_transform(confounders)

    n = len(Y)
    Y_resid = np.zeros(n)
    T_resid = np.zeros(n)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    for tr, te in kf.split(np.arange(n)):
        m_y = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42,
        )
        m_y.fit(conf_s[tr], Y[tr])
        Y_resid[te] = Y[te] - m_y.predict(conf_s[te])

        m_t = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42,
        )
        m_t.fit(conf_s[tr], pc1_norm[tr])
        T_resid[te] = pc1_norm[te] - m_t.predict(conf_s[te])

    denom = float(np.mean(T_resid ** 2))
    if denom < 1e-12:
        raise RuntimeError("Treatment fully explained by confounders; theta undefined")
    theta = float(np.mean(T_resid * Y_resid)) / denom
    psi = (Y_resid - theta * T_resid) * T_resid / denom
    se = float(np.sqrt(float(np.var(psi, ddof=1)) / n))

    # Full-data outcome model so we can predict log-price at any conf vector.
    model_y_full = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42,
    )
    model_y_full.fit(conf_s, Y)

    return DMLArtifacts(
        pca=pca,
        pc1_mean=pc1_mean,
        pc1_std=pc1_std,
        model_y_full=model_y_full,
        conf_scaler=conf_scaler,
        theta=theta,
        se=se,
        pc1_resid_train=T_resid,
        pc1_norm_train=pc1_norm,
    )


# ---------- sentence-transformers re-encoder ---------------------------------

_ENCODER_CACHE: dict = {}


def _get_encoder(model_name: str = EMBEDDING_MODEL):
    if "model" in _ENCODER_CACHE:
        return _ENCODER_CACHE["model"]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from sentence_transformers import SentenceTransformer
        _ENCODER_CACHE["model"] = SentenceTransformer(model_name)
    return _ENCODER_CACHE["model"]


def encode_texts(texts: list[str]) -> np.ndarray:
    enc = _get_encoder()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return enc.encode(texts, batch_size=32, show_progress_bar=False)


# ---------- predicted log-price for a rewrite --------------------------------

def predicted_logprice_for_rewrite(
    art: DMLArtifacts,
    rewrite_emb: np.ndarray,
    listing_conf_s: np.ndarray,
) -> float:
    """Push a rewrite through the trained DML model.

    Baseline = model_y(listing's confounder vector). Marginal text effect =
    theta * (pc1_norm_rewrite). Returns baseline + marginal."""
    pc1_rewrite = float(art.pca.transform(rewrite_emb.reshape(1, -1))[0, 0])
    pc1_rewrite_norm = (pc1_rewrite - art.pc1_mean) / art.pc1_std
    baseline = float(art.model_y_full.predict(listing_conf_s.reshape(1, -1))[0])
    return baseline + art.theta * pc1_rewrite_norm


def baseline_logprice(art: DMLArtifacts, listing_conf_s: np.ndarray) -> float:
    return float(art.model_y_full.predict(listing_conf_s.reshape(1, -1))[0])


# ---------- bootstrap CIs -----------------------------------------------------

def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 42,
                      alpha: float = 0.05) -> tuple[float, float, float]:
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    n = len(values)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = values[idx].mean()
    return float(values.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


# ---------- pipeline orchestration -------------------------------------------

def _load_sf_descriptions() -> pd.DataFrame:
    path = RAW_DIR / "descriptions" / "sf_descriptions.csv"
    df = pd.read_csv(path)
    df = df[df["description"].astype(str).str.len() > 50].reset_index(drop=True)
    return df


def _zip_to_target_submarket(zip_int: int) -> str:
    """Crude zip→submarket map for the SF dataset. Used to (a) seed an attribute
    classifier label, (b) implicitly map a target submarket to a target zip
    label for the validator."""
    z = int(zip_int)
    return {
        94110: "Mission District",
        94103: "SoMa",
        94114: "Castro",
        94131: "Noe Valley",
        94121: "Richmond",
        94122: "Sunset",
        94123: "Marina",
        94115: "Pacific Heights",
        94109: "Pacific Heights",
        94107: "SoMa",
        94105: "SoMa",
    }.get(z, "Mission District")


def _submarket_to_target_zip(submarket: str) -> Optional[int]:
    rev = {
        "Mission District": 94110,
        "SoMa": 94103,
        "Castro": 94114,
        "Noe Valley": 94131,
        "Richmond": 94121,
        "Sunset": 94122,
        "Marina": 94123,
        "Pacific Heights": 94115,
    }
    return rev.get(submarket)


def _pick_swap_targets(orig_submarket: str, k: int = 3) -> list[str]:
    """Pick k alternative submarkets, deterministically, for the swap arms."""
    pool = [s for s in SUBMARKET_HINTS.keys() if s != orig_submarket]
    return pool[:k]


def run_pipeline(
    city: str,
    n_listings: int,
    out_path: Path,
    force_mock: bool = False,
    skip_perplexity: bool = False,
    n_pca: int = 50,
    seed: int = 42,
) -> dict:
    """Top-level orchestration. Returns the result dict written to disk."""
    print(f"\n=== Counterfactual pipeline: {city} (n_listings={n_listings}) ===")

    # 1. load SF embeddings + parcels for the production DML refit
    loaded = load_analysis_data(city)
    if loaded is None:
        raise FileNotFoundError(f"No analysis data found for city={city}")
    emb_df, parcels = loaded
    feats = get_features_and_target(emb_df, parcels, drop_mismatched_crime=True)
    if feats is None:
        raise RuntimeError("get_features_and_target returned None")
    T, confounders, Y, meta = feats
    print(f"  DML training set: N={len(Y):,}, text_dim={T.shape[1]}, conf_dim={confounders.shape[1]}")

    # 2. fit DML artifacts (PCA, conf-scaler, model_y, theta)
    art = fit_dml_artifacts(T, confounders, Y, n_pca=n_pca)
    print(f"  Fitted DML θ = {art.theta:+.4f} (SE {art.se:.4f}, n_pca={art.pca.n_components_})")

    # 3. attribute classifier on raw description text
    sf_desc = _load_sf_descriptions()
    print(f"  Raw SF descriptions: {len(sf_desc)}")
    fit_zip_classifier(
        sf_desc["description"].astype(str).tolist(),
        sf_desc["zip"].astype(int).tolist(),
    )
    print(f"  Fitted zip-as-label TF-IDF + LogisticRegressionCV classifier")

    # 4. select listings for the experiment
    rng = np.random.default_rng(seed)
    n_pick = min(n_listings, len(sf_desc))
    pick_idx = rng.choice(len(sf_desc), size=n_pick, replace=False)
    pick_idx.sort()

    # 5. align listings to confounder rows (best-effort; missing → use mean conf)
    conf_s_full = art.conf_scaler.transform(confounders)
    mean_conf_s = conf_s_full.mean(axis=0)

    # join sf_desc to emb_df by address+zip to map a listing to its conf row
    addr_zip_to_row: dict[tuple[str, int], int] = {}
    if "address" in emb_df.columns and "zip" in emb_df.columns:
        for i, row in emb_df.reset_index(drop=True).iterrows():
            try:
                addr_zip_to_row[(str(row["address"]).strip().lower(), int(row["zip"]))] = i
            except Exception:
                pass

    generator = make_generator(force_mock=force_mock)
    print(f"  Generator: {type(generator).__name__}")

    # 6. main loop
    listings_out: list[ListingRecord] = []
    print(f"\n  Generating + validating {n_pick} listings × 4 variants ...")
    for n_done, li in enumerate(pick_idx):
        row = sf_desc.iloc[li]
        original_text = str(row["description"])
        zip_int = int(row["zip"])
        addr = str(row.get("address", ""))
        slots = extract_slots(original_text)

        orig_submarket = _zip_to_target_submarket(zip_int)
        swap_targets = _pick_swap_targets(orig_submarket, k=3)
        arms: list[tuple[str, Optional[str], str]] = []  # (arm_name, target_sub, prompt)
        for tgt in swap_targets:
            arms.append((
                f"style_swap:{tgt}", tgt,
                style_swap_prompt(tgt, original_text, slots),
            ))
        arms.append((
            "style_stripped", None,
            style_stripped_prompt(original_text, slots),
        ))

        # baseline: use the listing's matched conf row if we can; else mean.
        conf_key = (addr.strip().lower(), zip_int)
        if conf_key in addr_zip_to_row and addr_zip_to_row[conf_key] < len(conf_s_full):
            listing_conf_s = conf_s_full[addr_zip_to_row[conf_key]]
        else:
            listing_conf_s = mean_conf_s
        baseline = baseline_logprice(art, listing_conf_s)

        rec = ListingRecord(
            listing_idx=int(li),
            address=addr,
            zip=zip_int,
            original_text=original_text,
            slots=slots,
        )

        # batch the 4 rewrites' embeddings together to amortize encoder load
        gen_results: list[tuple[str, Optional[str], GenerationResult]] = []
        for arm_name, target_sub, prompt in arms:
            res = generator.generate(prompt, slot_dict=slots, original_text=original_text)
            gen_results.append((arm_name, target_sub, res))

        rewrite_texts = [r.rewritten_text for _, _, r in gen_results]
        rewrite_embs = encode_texts(rewrite_texts)

        for (arm_name, target_sub, gres), emb in zip(gen_results, rewrite_embs):
            target_zip = _submarket_to_target_zip(target_sub) if target_sub else None
            v = validate_rewrite(
                original_text=original_text,
                rewritten_text=gres.rewritten_text,
                target_zip=target_zip,
                skip_perplexity=skip_perplexity,
            )
            pred = predicted_logprice_for_rewrite(art, emb, listing_conf_s)
            rec.rewrites.append(RewriteRecord(
                arm=arm_name,
                target_submarket=target_sub,
                target_zip=target_zip,
                rewritten_text=gres.rewritten_text,
                used_mock=gres.used_mock,
                validation=asdict(v),
                pred_logprice_baseline=baseline,
                pred_logprice_rewrite=pred,
                delta_logprice=pred - baseline,
            ))
        listings_out.append(rec)
        if (n_done + 1) % 5 == 0 or n_done == n_pick - 1:
            print(f"    [{n_done + 1}/{n_pick}] processed")

    # 7. aggregate per-arm causal effects
    per_arm: dict[str, dict] = {}
    for arm_kind in ("style_stripped", "style_swap"):
        deltas: list[float] = []
        for L in listings_out:
            for rw in L.rewrites:
                if not rw.validation["overall_pass"]:
                    continue
                if arm_kind == "style_stripped" and rw.arm == "style_stripped":
                    deltas.append(rw.delta_logprice)
                if arm_kind == "style_swap" and rw.arm.startswith("style_swap:"):
                    deltas.append(rw.delta_logprice)
        arr = np.asarray(deltas, dtype=float)
        mean, lo, hi = bootstrap_mean_ci(arr) if len(arr) > 0 else (float("nan"),) * 3
        per_arm[arm_kind] = {
            "n_valid": int(len(arr)),
            "mean_delta_logprice": mean,
            "ci_low": lo,
            "ci_high": hi,
            "pct_change_implied": (float(np.exp(mean) - 1) * 100) if not np.isnan(mean) else float("nan"),
        }

    # 8. validation pass rates
    n_total = sum(len(L.rewrites) for L in listings_out)
    pass_rates = {
        "slot_preserved": sum(1 for L in listings_out for r in L.rewrites if r.validation["slot_preserved"]) / max(n_total, 1),
        "ppl_ok":          sum(1 for L in listings_out for r in L.rewrites if r.validation["ppl_ok"]) / max(n_total, 1),
        "classifier_flipped_toward_target": sum(1 for L in listings_out for r in L.rewrites if r.validation["classifier_flipped_toward_target"]) / max(n_total, 1),
        "overall_pass":    sum(1 for L in listings_out for r in L.rewrites if r.validation["overall_pass"]) / max(n_total, 1),
    }

    # 9. assemble + write JSON
    out: dict = {
        "city": city,
        "n_listings_requested": int(n_listings),
        "n_listings_processed": int(len(listings_out)),
        "n_rewrites_total": int(n_total),
        "used_mock_generator": bool(isinstance(generator, MockGenerator)),
        "skip_perplexity": bool(skip_perplexity),
        "dml": {
            "theta": art.theta,
            "se": art.se,
            "n_pca": int(art.pca.n_components_),
        },
        "validation_pass_rates": pass_rates,
        "natural_direct_effect_style_stripped": per_arm["style_stripped"],
        "total_effect_style_swap": per_arm["style_swap"],
        "listings": [
            {
                "listing_idx": L.listing_idx,
                "address": L.address,
                "zip": L.zip,
                "slots": L.slots,
                "rewrites": [asdict(r) for r in L.rewrites],
            }
            for L in listings_out
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Wrote {out_path}")
    print(f"  validation pass rates: {pass_rates}")
    print(f"  NDE (style-stripped): {per_arm['style_stripped']}")
    print(f"  TE  (style-swap):     {per_arm['style_swap']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="sf", choices=["sf"], help="only sf supported initially")
    ap.add_argument("--n_listings", type=int, default=25)
    ap.add_argument("--out", type=Path, default=Path("results/counterfactual/sf.json"))
    ap.add_argument("--force_mock", action="store_true",
                    help="ignore ANTHROPIC_API_KEY and use MockGenerator")
    ap.add_argument("--skip_perplexity", action="store_true",
                    help="skip GPT-2 perplexity check (faster smoke runs)")
    ap.add_argument("--n_pca", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    reset_caches()
    run_pipeline(
        city=args.city,
        n_listings=args.n_listings,
        out_path=args.out,
        force_mock=args.force_mock,
        skip_perplexity=args.skip_perplexity,
        n_pca=args.n_pca,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
