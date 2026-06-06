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
from sklearn.ensemble import GradientBoostingRegressor  # noqa: F401
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from booster import make_regressor
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

from generator import GenerationResult, MockGenerator, AsyncMockGenerator, make_async_generator, make_generator  # noqa: F401
from prompts import SUBMARKET_HINTS, style_stripped_blocks, style_swap_blocks
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
    model_y_full: object  # GradientBoostingRegressor | lightgbm.LGBMRegressor
    conf_scaler: StandardScaler
    theta: float
    se: float
    pc1_resid_train: np.ndarray
    pc1_norm_train: np.ndarray


def fit_dml_artifacts(
    T: np.ndarray, confounders: np.ndarray, Y: np.ndarray,
    n_pca: int = 50, k_folds: int = 5, use_ridge: bool = True,
) -> DMLArtifacts:
    """Refit the same DML pipeline as causal_inference.dml_continuous_treatment
    but RETAIN the fitted PCA, conf scaler, and a full-data model_y so we can
    score new rewrites.

    Ridge nuisances (use_ridge=True, default) match the Shen/Baur production
    pipeline and run ~50-100x faster than GBM at n<2000 with equivalent MSE
    per Chernozhukov et al. 2018 and the DoubleML py_learner benchmark.
    """
    from sklearn.linear_model import RidgeCV

    n_pca = min(n_pca, T.shape[1], T.shape[0] - 1)
    pca = PCA(n_components=n_pca, random_state=42)
    T_pca = pca.fit_transform(T)
    pc1 = T_pca[:, 0]
    pc1_mean = float(pc1.mean())
    pc1_std = float(pc1.std()) if pc1.std() > 0 else 1.0
    pc1_norm = (pc1 - pc1_mean) / pc1_std

    conf_scaler = StandardScaler()
    conf_s = conf_scaler.fit_transform(confounders)

    def _new_regressor():
        if use_ridge:
            return RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0, 1000.0))
        return make_regressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42,
        )

    n = len(Y)
    Y_resid = np.zeros(n)
    T_resid = np.zeros(n)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    for tr, te in kf.split(np.arange(n)):
        m_y = _new_regressor(); m_y.fit(conf_s[tr], Y[tr])
        Y_resid[te] = Y[te] - m_y.predict(conf_s[te])
        m_t = _new_regressor(); m_t.fit(conf_s[tr], pc1_norm[tr])
        T_resid[te] = pc1_norm[te] - m_t.predict(conf_s[te])

    denom = float(np.mean(T_resid ** 2))
    if denom < 1e-12:
        raise RuntimeError("Treatment fully explained by confounders; theta undefined")
    theta = float(np.mean(T_resid * Y_resid)) / denom
    psi = (Y_resid - theta * T_resid) * T_resid / denom
    se = float(np.sqrt(float(np.var(psi, ddof=1)) / n))

    model_y_full = _new_regressor()
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
        import torch
        m = SentenceTransformer(model_name)
        if torch.cuda.is_available():
            m = m.to("cuda")
        _ENCODER_CACHE["model"] = m
    return _ENCODER_CACHE["model"]


def encode_texts(texts: list[str], batch_size: int = 128) -> np.ndarray:
    """Encode texts in one batch. On GPU, batch_size=128 saturates the encoder
    without OOM on mpnet-768. CPU batches at 64 to avoid memory pressure."""
    enc = _get_encoder()
    import torch
    bs = batch_size if torch.cuda.is_available() else min(64, batch_size)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return enc.encode(texts, batch_size=bs, show_progress_bar=False,
                          convert_to_numpy=True)


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
    n = len(values)
    # Vectorized: draw all B*n indices at once, take row-means in one shot.
    boot_idx = rng.integers(0, n, size=(n_boot, n))
    boots = values[boot_idx].mean(axis=1)
    return (float(values.mean()),
            float(np.quantile(boots, alpha / 2)),
            float(np.quantile(boots, 1 - alpha / 2)))


# ---------- pipeline orchestration -------------------------------------------

def _load_city_descriptions(city: str, emb_df: pd.DataFrame) -> pd.DataFrame:
    """Pull (description, zip, address) from the city's embeddings parquet.

    For SF the original sf_descriptions.csv exists at RAW_DIR / 'descriptions';
    we prefer that to preserve the SF replication numbers. For the other 11
    cities we read description columns directly from emb_df (already loaded
    by run_pipeline). All paths return a DataFrame with at least 'description'
    and 'zip' columns; descriptions shorter than 50 chars are dropped.
    """
    if city == "sf":
        path = RAW_DIR / "descriptions" / "sf_descriptions.csv"
        if path.exists():
            df = pd.read_csv(path)
            df = df[df["description"].astype(str).str.len() > 50].reset_index(drop=True)
            return df
    desc_col = ("clean_description" if "clean_description" in emb_df.columns
                else "description" if "description" in emb_df.columns else None)
    if desc_col is None:
        raise FileNotFoundError(f"no description column on emb_df for city={city}")
    cols = [desc_col]
    if "zip" in emb_df.columns:
        cols.append("zip")
    if "address" in emb_df.columns:
        cols.append("address")
    df = emb_df[cols].copy()
    df = df.rename(columns={desc_col: "description"})
    if "zip" not in df.columns:
        df["zip"] = 0
    df["zip"] = pd.to_numeric(df["zip"], errors="coerce").fillna(0).astype(int)
    df = df[df["description"].astype(str).str.len() > 50].reset_index(drop=True)
    return df


# SF-specific zip→submarket map preserved from the 3-city pilot. For the 11
# new cities we have submarket vocabularies (prompts.SUBMARKET_HINTS[city])
# but no curated zip-to-submarket mapping; the new-city style-swap arm picks
# k=3 alternative submarkets at random from the city's pool, losing the
# "originally was X" precision but retaining the validator's discriminator
# check on the target submarket. Defensible scope choice given the JBES
# revision timeline; can be tightened in a v2 with hand-curated mappings.
_SF_ZIP_TO_SUBMARKET = {
    94110: "Mission District", 94103: "SoMa", 94114: "Castro",
    94131: "Noe Valley", 94121: "Richmond", 94122: "Sunset",
    94123: "Marina", 94115: "Pacific Heights", 94109: "Pacific Heights",
    94107: "SoMa", 94105: "SoMa",
}

_SF_SUBMARKET_TO_ZIP = {
    "Mission District": 94110, "SoMa": 94103, "Castro": 94114,
    "Noe Valley": 94131, "Richmond": 94121, "Sunset": 94122,
    "Marina": 94123, "Pacific Heights": 94115,
}


def _zip_to_target_submarket(city: str, zip_int: int) -> str:
    if city == "sf":
        return _SF_ZIP_TO_SUBMARKET.get(int(zip_int), "Mission District")
    city_subs = list(SUBMARKET_HINTS.get(city, {}).keys())
    if not city_subs:
        return "Unknown"
    return city_subs[int(zip_int) % len(city_subs)]


def _submarket_to_target_zip(city: str, submarket: str) -> Optional[int]:
    if city == "sf":
        return _SF_SUBMARKET_TO_ZIP.get(submarket)
    return None


def _pick_swap_targets(city: str, orig_submarket: str, k: int = 3) -> list[str]:
    """Pick k alternative submarkets, deterministically, for the swap arms."""
    pool = [s for s in SUBMARKET_HINTS.get(city, {}).keys() if s != orig_submarket]
    return pool[:k]


def _write_result(out: dict, out_path: Path) -> None:
    """Serialize the result dict to disk (also used for intra-city checkpoints)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)


def _assemble_result(
    city: str,
    n_listings: int,
    listings_out: list[ListingRecord],
    art: DMLArtifacts,
    generator: object,
    skip_perplexity: bool,
) -> dict:
    """Build the output dict from whatever listings have been processed so far.

    Called both for the final write and for the every-25-listings checkpoints,
    so a partial file is always a valid, self-consistent JSON with fewer rows.
    """
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

    # 9. cache + cost stats from generator (real API only)
    usage_summary: dict | None = None
    if not isinstance(generator, MockGenerator):
        u = generator.usage
        usage_summary = {
            "n_calls": u.n_calls,
            "input_tokens": u.input_tokens,
            "cache_creation_input_tokens": u.cache_creation_input_tokens,
            "cache_read_input_tokens": u.cache_read_input_tokens,
            "output_tokens": u.output_tokens,
            "cache_hit_rate": u.cache_hit_rate(),
            "estimated_cost_usd_sonnet35": u.estimated_cost_usd(),
        }

    # 10. assemble JSON
    return {
        "city": city,
        "n_listings_requested": int(n_listings),
        "n_listings_processed": int(len(listings_out)),
        "n_rewrites_total": int(n_total),
        "used_mock_generator": bool(isinstance(generator, MockGenerator)),
        "skip_perplexity": bool(skip_perplexity),
        "api_usage": usage_summary,
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


async def run_pipeline(
    city: str,
    n_listings: int,
    out_path: Path,
    force_mock: bool = False,
    skip_perplexity: bool = False,
    n_pca: int = 50,
    seed: int = 42,
    use_vllm: bool = False,
    vllm_model: str | None = None,
    generator: object | None = None,
) -> dict:
    """Top-level orchestration. Returns the result dict written to disk.

    Async so the 4 arms per listing can fire concurrently via asyncio.gather.
    """
    import asyncio
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
    city_desc = _load_city_descriptions(city, emb_df)
    print(f"  Raw {city} descriptions: {len(city_desc)}")
    fit_zip_classifier(
        city_desc["description"].astype(str).tolist(),
        city_desc["zip"].astype(int).tolist(),
    )
    print(f"  Fitted zip-as-label TF-IDF + LogisticRegressionCV classifier")

    # 4. select listings for the experiment
    rng = np.random.default_rng(seed)
    n_pick = min(n_listings, len(city_desc))
    pick_idx = rng.choice(len(city_desc), size=n_pick, replace=False)
    pick_idx.sort()

    # 5. align listings to confounder rows (best-effort; missing → use mean conf)
    conf_s_full = art.conf_scaler.transform(confounders)
    mean_conf_s = conf_s_full.mean(axis=0)

    addr_zip_to_row: dict[tuple[str, int], int] = {}
    if "address" in emb_df.columns and "zip" in emb_df.columns:
        for i, row in emb_df.reset_index(drop=True).iterrows():
            try:
                addr_zip_to_row[(str(row["address"]).strip().lower(), int(row["zip"]))] = i
            except Exception:
                pass

    # The generator is normally built ONCE in main() and passed in so the vLLM
    # engine (and its prefix cache) survives across all cities. Fall back to
    # building our own only when called directly (e.g. the smoke test).
    if generator is None:
        generator = make_async_generator(force_mock=force_mock,
                                         use_vllm=use_vllm, vllm_model=vllm_model)
    print(f"  Generator: {type(generator).__name__}")

    # ---- Pass 1: build the arms + per-listing metadata for EVERY listing and
    # flatten every arm across every listing into a single batch. Generation
    # then happens in one generate_blocks_batch() call so vLLM's continuous
    # batcher processes the whole city at once (instead of 4 arms × n_pick
    # separate calls) and the prefix cache stays warm. We record the flat index
    # span [start:end) that each listing owns so the results realign exactly.
    listing_meta: list[dict] = []
    all_items: list[dict] = []
    for li in pick_idx:
        row = city_desc.iloc[li]
        original_text = str(row["description"])
        zip_int = int(row["zip"])
        addr = str(row.get("address", ""))
        slots = extract_slots(original_text)

        orig_submarket = _zip_to_target_submarket(city, zip_int)
        swap_targets = _pick_swap_targets(city, orig_submarket, k=3)
        arms: list[tuple[str, Optional[str], dict]] = []
        for tgt in swap_targets:
            arms.append((
                f"style_swap:{tgt}", tgt,
                style_swap_blocks(tgt, original_text, slots, city=city),
            ))
        arms.append((
            "style_stripped", None,
            style_stripped_blocks(original_text, slots),
        ))

        # baseline: use the listing's matched conf row if we can; else mean.
        conf_key = (addr.strip().lower(), zip_int)
        if conf_key in addr_zip_to_row and addr_zip_to_row[conf_key] < len(conf_s_full):
            listing_conf_s = conf_s_full[addr_zip_to_row[conf_key]]
        else:
            listing_conf_s = mean_conf_s
        baseline = baseline_logprice(art, listing_conf_s)

        start = len(all_items)
        for _arm_name, _target_sub, blocks in arms:
            # original_text is carried so the asyncio.gather fallback (mock /
            # async Anthropic) can echo it back exactly as before; the vLLM
            # batch path ignores the extra key.
            all_items.append({
                "system": blocks["system"],
                "user": blocks["user"],
                "slot_dict": slots,
                "original_text": original_text,
            })
        listing_meta.append({
            "li": int(li),
            "original_text": original_text,
            "zip_int": zip_int,
            "addr": addr,
            "slots": slots,
            "arms": arms,
            "listing_conf_s": listing_conf_s,
            "baseline": baseline,
            "start": start,
            "end": len(all_items),
        })

    print(f"\n  Generating {n_pick} listings × ~4 variants "
          f"({len(all_items)} prompts) in one batch ...")
    # Single generation call for the whole city. Generators that expose a batch
    # surface (vLLM) continuous-batch on the GPU; generators without one (mock /
    # async Anthropic) fan the identical prompts out through one asyncio.gather.
    if hasattr(generator, "generate_blocks_batch"):
        flat_results = await generator.generate_blocks_batch(all_items)
    else:
        flat_results = await asyncio.gather(*[
            generator.generate_blocks(
                system=it["system"],
                user=it["user"],
                slot_dict=it["slot_dict"],
                original_text=it["original_text"],
            )
            for it in all_items
        ], return_exceptions=False)

    # ---- Pass 2: encode, validate, score, and build ListingRecords from the
    # flat results, realigned per listing. The accumulating result is flushed to
    # out_path every 25 listings so a crash loses at most ~25 listings, not the
    # whole city.
    listings_out: list[ListingRecord] = []
    for n_done, meta in enumerate(listing_meta):
        arms = meta["arms"]
        arm_results = flat_results[meta["start"]:meta["end"]]
        gen_results: list[tuple[str, Optional[str], GenerationResult]] = [
            (arm_name, target_sub, res)
            for (arm_name, target_sub, _), res in zip(arms, arm_results)
        ]

        rewrite_texts = [r.rewritten_text for _, _, r in gen_results]
        rewrite_embs = encode_texts(rewrite_texts)

        listing_conf_s = meta["listing_conf_s"]
        baseline = meta["baseline"]
        rec = ListingRecord(
            listing_idx=meta["li"],
            address=meta["addr"],
            zip=meta["zip_int"],
            original_text=meta["original_text"],
            slots=meta["slots"],
        )

        for (arm_name, target_sub, gres), emb in zip(gen_results, rewrite_embs):
            target_zip = _submarket_to_target_zip(city, target_sub) if target_sub else None
            v = validate_rewrite(
                original_text=meta["original_text"],
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
        if (n_done + 1) % 25 == 0 and (n_done + 1) < n_pick:
            _write_result(
                _assemble_result(city, n_listings, listings_out, art,
                                 generator, skip_perplexity),
                out_path,
            )
            print(f"    checkpoint written ({n_done + 1}/{n_pick}) -> {out_path}")

    # 7-10. aggregate per-arm effects, pass rates, usage; assemble + write JSON.
    out = _assemble_result(
        city, n_listings, listings_out, art, generator, skip_perplexity,
    )
    _write_result(out, out_path)
    print(f"\n  Wrote {out_path}")
    print(f"  validation pass rates: {out['validation_pass_rates']}")
    print(f"  NDE (style-stripped): {out['natural_direct_effect_style_stripped']}")
    print(f"  TE  (style-swap):     {out['total_effect_style_swap']}")
    if not isinstance(generator, MockGenerator):
        print("\n  API usage (with prompt caching on system block):")
        print(generator.usage.summary())
    return out


ALL_12 = ["boston", "nyc", "sf", "dc", "philadelphia", "chicago",
          "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city")
    ap.add_argument("--all_12", action="store_true",
                    help="run the pipeline once per city across the 12-metro set")
    ap.add_argument("--n_listings", type=int, default=25)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("results/counterfactual"),
                    help="output directory; per-city files written as {city}.json")
    ap.add_argument("--force_mock", action="store_true",
                    help="ignore ANTHROPIC_API_KEY and use MockGenerator")
    ap.add_argument("--use_vllm", action="store_true",
                    help="route generation through VLLMGenerator (Qwen 2.5 32B "
                         "Instruct AWQ by default); requires CUDA + vllm + outlines")
    ap.add_argument("--vllm_model", type=str, default=None,
                    help="override the default vLLM model id")
    ap.add_argument("--skip_perplexity", action="store_true",
                    help="skip GPT-2 perplexity check (faster smoke runs)")
    ap.add_argument("--force_redo", action="store_true",
                    help="re-run cities that already have an output JSON "
                         "(default: skip cities with results, resume-friendly)")
    ap.add_argument("--n_pca", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.all_12:
        cities = list(ALL_12)
    elif args.city is not None:
        cities = [args.city]
    else:
        ap.error("specify --city or --all_12")

    import asyncio
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Build the generator ONCE, before the city loop, and reuse it across every
    # city. Previously run_pipeline() built its own per city, so --all_12 loaded
    # the 32B vLLM engine 12 times; a single resident engine also keeps its
    # prefix cache warm across cities. reset_caches() (per city below) only
    # clears the validator/encoder caches and never touches this generator.
    # Skip the load entirely when every requested city already has output.
    pending = [c for c in cities
               if args.force_redo or not (args.out_dir / f"{c}.json").exists()]
    generator = None
    if pending:
        generator = make_async_generator(
            force_mock=args.force_mock,
            use_vllm=args.use_vllm,
            vllm_model=args.vllm_model,
        )
        print(f"  Generator (shared across cities): {type(generator).__name__}")

    for c in cities:
        out_path = args.out_dir / f"{c}.json"
        if out_path.exists() and not args.force_redo:
            print(f"=== {c}: existing {out_path} found, skipping (use --force_redo to re-run) ===")
            continue
        reset_caches()
        asyncio.run(run_pipeline(
            city=c,
            n_listings=args.n_listings,
            out_path=out_path,
            force_mock=args.force_mock,
            skip_perplexity=args.skip_perplexity,
            n_pca=args.n_pca,
            seed=args.seed,
            use_vllm=args.use_vllm,
            vllm_model=args.vllm_model,
            generator=generator,
        ))


if __name__ == "__main__":
    main()
