"""Three-pronged validation for each counterfactual rewrite.

  (a) slot preservation — every numeric fact extracted from rewrite must match
      the original (slot_extractor; absolute equality on non-None pairs).
  (b) perplexity sanity — GPT-2 perplexity ratio rewrite/original; flagged when
      the rewrite is more than `ppl_ratio_max` (default 3.0) times the
      original. Catches degenerate / nonsense outputs.
  (c) attribute classification — a sklearn LogisticRegressionCV trained on the
      995 real listings with zip-as-label is asked to classify the rewrite. We
      report the predicted zip and whether it shifted toward the target zip
      (style-swap) or away from the original (style-stripped).

GPT-2 and the LR classifier are lazy-loaded once and cached at module level so
the smoke test stays cheap when validating only a few rewrites.
"""
from __future__ import annotations

import contextlib
import io
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from slot_extractor import extract_slots, slots_match


@dataclass
class ValidationResult:
    slot_preserved: bool
    slot_diffs: dict[str, bool]
    ppl_original: float
    ppl_rewrite: float
    ppl_ratio: float
    ppl_ok: bool
    classifier_orig_label: Optional[int]
    classifier_rewrite_label: Optional[int]
    classifier_target_label: Optional[int]
    classifier_flipped_toward_target: bool
    overall_pass: bool
    notes: list[str] = field(default_factory=list)


# ---------- perplexity (GPT-2) ----------------------------------------------

_PPL_CACHE: dict = {}


def _load_gpt2():
    if "model" in _PPL_CACHE:
        return _PPL_CACHE["model"], _PPL_CACHE["tok"]
    # Suppress chatty HF download logs during model load.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        import torch
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.eval()
    _PPL_CACHE["model"] = model
    _PPL_CACHE["tok"] = tok
    _PPL_CACHE["torch"] = torch
    return model, tok


def perplexity(text: str, max_tokens: int = 512) -> float:
    """Per-token cross-entropy perplexity under GPT-2. Cheap (CPU is fine for
    short Redfin-length descriptions); cached model load."""
    if not text or not text.strip():
        return float("inf")
    model, tok = _load_gpt2()
    torch = _PPL_CACHE["torch"]
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    if enc["input_ids"].shape[1] < 2:
        return float("inf")
    with torch.no_grad():
        out = model(**enc, labels=enc["input_ids"])
    return float(math.exp(min(out.loss.item(), 50.0)))


# ---------- attribute classifier (zip from text) -----------------------------

_CLF_CACHE: dict = {}


def fit_zip_classifier(texts: list[str], zip_labels: list[int], random_state: int = 42):
    """One-shot fit of a TfidfVectorizer + LogisticRegressionCV pipeline that
    predicts the listing's zip. Cached at the module level so we fit once for
    the whole pipeline run."""
    if "pipe" in _CLF_CACHE:
        return _CLF_CACHE["pipe"], _CLF_CACHE["labels"]
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            stop_words="english",
            max_features=20_000,
        )),
        ("clf", LogisticRegressionCV(
            Cs=5,
            cv=3,
            max_iter=1000,
            n_jobs=1,
            random_state=random_state,
        )),
    ])
    pipe.fit(texts, zip_labels)
    _CLF_CACHE["pipe"] = pipe
    _CLF_CACHE["labels"] = sorted(set(zip_labels))
    return pipe, _CLF_CACHE["labels"]


def reset_caches() -> None:
    """Used by tests to ensure isolated state across runs."""
    _PPL_CACHE.clear()
    _CLF_CACHE.clear()


# ---------- main validation entry point --------------------------------------

def validate_rewrite(
    original_text: str,
    rewritten_text: str,
    target_zip: Optional[int] = None,
    ppl_ratio_max: float = 3.0,
    skip_perplexity: bool = False,
) -> ValidationResult:
    """Run all three checks against a single rewrite.

    target_zip: for style-swap rewrites, the zip-code label corresponding to
    the target submarket. None for style-stripped (we don't expect a flip
    toward a specific target — only away from the original)."""
    notes: list[str] = []

    # (a) slot preservation
    so = extract_slots(original_text)
    sr = extract_slots(rewritten_text)
    diffs = slots_match(so, sr)
    slot_ok = all(diffs.values())
    if not slot_ok:
        bad = [k for k, ok in diffs.items() if not ok]
        notes.append(f"slot mismatch: {bad}")

    # (b) perplexity sanity
    if skip_perplexity:
        ppl_o = ppl_r = float("nan")
        ppl_ratio = float("nan")
        ppl_ok = True
        notes.append("perplexity skipped")
    else:
        ppl_o = perplexity(original_text)
        ppl_r = perplexity(rewritten_text)
        if ppl_o > 0 and not math.isinf(ppl_o):
            ppl_ratio = ppl_r / ppl_o
        else:
            ppl_ratio = float("inf")
        ppl_ok = bool(ppl_ratio <= ppl_ratio_max) if not math.isinf(ppl_ratio) else False
        if not ppl_ok:
            notes.append(f"perplexity blew up: {ppl_ratio:.2f}x")

    # (c) attribute classifier (must be pre-fit)
    orig_label = rew_label = None
    flipped = False
    if "pipe" in _CLF_CACHE:
        pipe = _CLF_CACHE["pipe"]
        try:
            orig_label = int(pipe.predict([original_text])[0])
            rew_label = int(pipe.predict([rewritten_text])[0])
        except Exception as e:
            notes.append(f"classifier predict failed: {e}")
        if orig_label is not None and rew_label is not None:
            if target_zip is not None:
                flipped = (rew_label == int(target_zip)) or (rew_label != orig_label)
            else:
                flipped = rew_label != orig_label
    else:
        notes.append("classifier not fit; flip check skipped")

    overall = slot_ok and ppl_ok  # classifier flip is a soft signal, not gate
    return ValidationResult(
        slot_preserved=slot_ok,
        slot_diffs=diffs,
        ppl_original=ppl_o,
        ppl_rewrite=ppl_r,
        ppl_ratio=ppl_ratio,
        ppl_ok=ppl_ok,
        classifier_orig_label=orig_label,
        classifier_rewrite_label=rew_label,
        classifier_target_label=int(target_zip) if target_zip is not None else None,
        classifier_flipped_toward_target=bool(flipped),
        overall_pass=bool(overall),
        notes=notes,
    )
