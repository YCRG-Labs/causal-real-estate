"""Smoke test for the replications package.

Runs both the Shen and Baur replications on a 200-row subset of the SF data
and confirms the JSON output structure. Designed to finish in under two
minutes on a CPU.

  pytest tests/test_smoke.py -v        # run from data/scripts/replications/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make `replications` importable regardless of how pytest is launched.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from replications.baur_2023 import run_baur  # noqa: E402
from replications.shen_2021 import run_shen  # noqa: E402

N_SUBSET = 200


@pytest.fixture(scope="module")
def shen_result():
    return run_shen(city="sf", n_subset=N_SUBSET, seed=0)


@pytest.fixture(scope="module")
def baur_result():
    return run_baur(city="sf", n_subset=N_SUBSET, seed=0, k_folds=3)


def test_shen_structure(shen_result, tmp_path_factory):
    r = shen_result
    assert r.get("paper", "").startswith("Shen"), r
    assert r["city"] == "sf"
    assert r["n"] == N_SUBSET
    assert r["n_confounders"] >= 5

    ols = r["ols_uniqueness"]
    for k in ("coef", "se", "p", "ci_low", "ci_high", "n", "r2", "pct_per_sd"):
        assert k in ols, f"OLS missing {k}"
    assert ols["n"] == N_SUBSET
    assert ols["se"] >= 0

    dml = r["dml_uniqueness"]
    if "error" not in dml:
        for k in ("theta", "se", "ci_low", "ci_high", "mde", "contains_zero"):
            assert k in dml, f"DML missing {k}"
        assert dml["se"] >= 0
        assert dml["ci_low"] <= dml["ci_high"]

    out = tmp_path_factory.mktemp("shen") / "shen.json"
    out.write_text(json.dumps(r, indent=2, default=str))
    assert out.exists() and out.stat().st_size > 0


def test_baur_structure(baur_result, tmp_path_factory):
    r = baur_result
    assert r.get("paper", "").startswith("Baur"), r
    assert r["city"] == "sf"
    assert r["n"] == N_SUBSET
    assert r["engine"] in {"lightgbm", "sklearn-gbr"}
    assert r["n_structured_features"] >= 5
    assert r["embedding_dim"] == 768

    for cv_key in ("cv_structured", "cv_structured_plus_bert"):
        cv = r[cv_key]
        for k in ("mae", "mape", "rmse", "n", "n_features"):
            assert k in cv, f"{cv_key} missing {k}"
        assert cv["mae"] > 0
        assert 0 <= cv["mape"]
        assert cv["rmse"] >= cv["mae"] / 10  # rough sanity

    gain = r["predictive_gain"]
    for k in ("delta_mae", "delta_mape", "delta_rmse"):
        assert k in gain

    dml = r["dml_bert"]
    if "error" not in dml:
        for k in ("theta", "se", "ci_low", "ci_high", "mde", "contains_zero"):
            assert k in dml, f"DML missing {k}"
        assert dml["se"] >= 0
        assert dml["ci_low"] <= dml["ci_high"]

    out = tmp_path_factory.mktemp("baur") / "baur.json"
    out.write_text(json.dumps(r, indent=2, default=str))
    assert out.exists() and out.stat().st_size > 0


def test_engine_recorded(baur_result):
    """If LightGBM substitution happened, the JSON must record it."""
    notes = baur_result["method_notes"]
    if notes["engine_used"] != "lightgbm":
        assert notes["engine_substituted"] is True
        assert notes["engine_substitution_reason"]
