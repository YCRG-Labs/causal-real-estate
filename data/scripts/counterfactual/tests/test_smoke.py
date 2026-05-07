"""End-to-end smoke test: run the pipeline on N=2 SF listings under MockGenerator.

Confirms the JSON output has the structure the paper's analysis code expects.
Uses --skip_perplexity to keep wall time low (no GPT-2 model download in the
smoke path); slot + classifier checks are still exercised.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
SCRIPTS = PKG.parent
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(SCRIPTS))


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    """Ensure the mock path is taken regardless of host env."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_smoke_n2_mock():
    from run_counterfactual import run_pipeline
    from validator import reset_caches

    reset_caches()

    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "sf_smoke.json"
        result = run_pipeline(
            city="sf",
            n_listings=2,
            out_path=out_path,
            force_mock=True,
            skip_perplexity=True,
            n_pca=20,
            seed=7,
        )

        assert out_path.exists(), "expected output JSON to be written"
        on_disk = json.loads(out_path.read_text())

        # ---- top-level shape ----
        for key in (
            "city", "n_listings_requested", "n_listings_processed",
            "n_rewrites_total", "used_mock_generator", "dml",
            "validation_pass_rates",
            "natural_direct_effect_style_stripped",
            "total_effect_style_swap",
            "listings",
        ):
            assert key in on_disk, f"missing top-level key: {key}"

        assert on_disk["city"] == "sf"
        assert on_disk["used_mock_generator"] is True
        assert on_disk["n_listings_processed"] == 2
        # 2 listings × 4 variants
        assert on_disk["n_rewrites_total"] == 8
        assert len(on_disk["listings"]) == 2

        # ---- DML block ----
        for k in ("theta", "se", "n_pca"):
            assert k in on_disk["dml"]

        # ---- per-listing shape ----
        for L in on_disk["listings"]:
            for key in ("listing_idx", "address", "zip", "slots", "rewrites"):
                assert key in L
            assert len(L["rewrites"]) == 4
            arm_kinds = {r["arm"].split(":", 1)[0] for r in L["rewrites"]}
            assert "style_swap" in arm_kinds
            assert "style_stripped" in arm_kinds
            for rw in L["rewrites"]:
                for key in (
                    "arm", "rewritten_text", "used_mock", "validation",
                    "pred_logprice_baseline", "pred_logprice_rewrite",
                    "delta_logprice",
                ):
                    assert key in rw
                for vkey in (
                    "slot_preserved", "ppl_ok",
                    "classifier_flipped_toward_target", "overall_pass",
                ):
                    assert vkey in rw["validation"]
                # MockGenerator returns the original verbatim, so slot
                # preservation MUST hold
                assert rw["validation"]["slot_preserved"] is True
                # Mock returns the raw description; the production embeddings
                # were built from a cleaned/lowercased version of that text,
                # so the re-encoded "rewrite" gives a slightly different PC1
                # than the training embedding. Delta should still be tiny —
                # well under 1% in log-price (~0.01).
                assert abs(rw["delta_logprice"]) < 0.01, (
                    f"mock should produce ~zero delta (text unchanged), "
                    f"got {rw['delta_logprice']}"
                )

        # ---- validation pass rates ----
        rates = on_disk["validation_pass_rates"]
        for k in ("slot_preserved", "ppl_ok",
                  "classifier_flipped_toward_target", "overall_pass"):
            assert k in rates
            assert 0.0 <= rates[k] <= 1.0
        # under mock, slot preservation is always 1.0
        assert rates["slot_preserved"] == 1.0
        # classifier flip cannot happen for unchanged text → 0
        assert rates["classifier_flipped_toward_target"] == 0.0

        # ---- aggregate effect blocks ----
        for arm in ("natural_direct_effect_style_stripped", "total_effect_style_swap"):
            block = on_disk[arm]
            for k in ("n_valid", "mean_delta_logprice", "ci_low", "ci_high",
                      "pct_change_implied"):
                assert k in block
            # mock → all valid (slot-preserved, ppl skipped); mean delta
            # is close to zero but not bit-exact because the production
            # embedding was built from a cleaned variant of the text
            assert block["n_valid"] > 0
            assert abs(block["mean_delta_logprice"]) < 0.01

        print("\n=== smoke test assertions ===")
        print(f"  listings processed: {on_disk['n_listings_processed']}")
        print(f"  rewrites total:     {on_disk['n_rewrites_total']}")
        print(f"  DML θ:              {on_disk['dml']['theta']:+.4f} "
              f"(SE {on_disk['dml']['se']:.4f}, n_pca={on_disk['dml']['n_pca']})")
        print(f"  pass rates:         {rates}")
        print(f"  NDE block:          {on_disk['natural_direct_effect_style_stripped']}")
        print(f"  TE  block:          {on_disk['total_effect_style_swap']}")
        print("=== smoke test PASS ===")
