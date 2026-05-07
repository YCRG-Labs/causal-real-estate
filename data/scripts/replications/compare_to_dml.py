"""Thin wrapper around dml_continuous_treatment for the replication scripts.

The published replications (Shen 2021, Baur 2023) each define their own
"treatment" of interest:

  - Shen: a scalar TF-IDF uniqueness score per listing
  - Baur: the 768-dim BERT (mpnet) embedding (PC1 is what DML actually scores)

dml_continuous_treatment expects a (n, k) matrix and PCAs it down. This
wrapper accepts either a 1-D vector or a 2-D matrix, calls the project DML
silently, and packs the result into a uniform DMLResult dataclass that both
replications can consume.
"""
from __future__ import annotations

import contextlib
import io
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from causal_inference import dml_continuous_treatment


@dataclass
class DMLResult:
    label: str
    n: int
    theta: float
    se: float
    ci_low: float
    ci_high: float
    mde: float
    contains_zero: bool


def run_dml(
    T: np.ndarray,
    confounders: np.ndarray,
    Y: np.ndarray,
    label: str,
    n_pca: int = 50,
    k_folds: int = 5,
) -> DMLResult | None:
    """Run dml_continuous_treatment and box the output as a DMLResult.

    T may be 1-D (scalar treatment) or 2-D (vector treatment).  If 1-D it is
    reshaped to (n, 1) so the underlying PCA degenerates to a standardisation.
    """
    T = np.asarray(T)
    if T.ndim == 1:
        T = T.reshape(-1, 1)

    n_pca = min(n_pca, T.shape[1], T.shape[0] - 1)

    with contextlib.redirect_stdout(io.StringIO()):
        raw = dml_continuous_treatment(T, confounders, Y, n_pca=n_pca, k_folds=k_folds)
    if raw is None:
        return None
    lo, hi = raw["ci"]
    return DMLResult(
        label=label,
        n=int(len(Y)),
        theta=float(raw["theta"]),
        se=float(raw["se"]),
        ci_low=float(lo),
        ci_high=float(hi),
        mde=float(raw["mde"]),
        contains_zero=bool(lo <= 0 <= hi),
    )


def result_to_dict(res: DMLResult | None) -> dict:
    if res is None:
        return {"error": "DML failed (treatment fully explained by confounders)"}
    return asdict(res)
