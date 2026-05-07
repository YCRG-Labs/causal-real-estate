"""Anthropic SDK wrapper with a mock fallback so the pipeline runs without a key.

If `ANTHROPIC_API_KEY` is set AND the `anthropic` package imports cleanly, we
use the real client (Claude 3.5 Sonnet by default) with retry-with-backoff on
transient API errors. Otherwise we drop in a `MockGenerator` that returns the
original text wrapped in the expected JSON shape — sufficient for end-to-end
dry-runs and the smoke test.

The mock IS the default when no key is present. This module never raises on
import for missing keys or missing package.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Optional

try:
    import anthropic  # type: ignore
    _HAS_ANTHROPIC = True
except ImportError:
    anthropic = None  # type: ignore
    _HAS_ANTHROPIC = False


@dataclass
class GenerationResult:
    rewritten_text: str
    preserved_slots: dict
    raw: str
    used_mock: bool


def _parse_json_payload(raw: str) -> dict:
    """Robustly pull the first JSON object out of a model response."""
    raw = raw.strip()
    # Strip code fences if the model added them despite instructions
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Fall back: find the first balanced {...}
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse JSON payload: {e}\nRaw: {raw[:500]}")
    raise ValueError(f"No JSON object found in response: {raw[:500]}")


class MockGenerator:
    """No-op generator for dry runs and tests.

    Returns the original text unchanged, with the slot dict echoed back. This
    means downstream validation (slot-preservation) will always pass on the
    mock; perplexity-ratio will always be exactly 1.0; the attribute-classifier
    flip check will always FAIL (since the text didn't actually change). The
    pipeline reports these per-check pass rates so the failure mode is visible.
    """

    used_mock = True

    def generate(self, prompt: str, slot_dict: Optional[dict] = None,
                 original_text: Optional[str] = None, **_: object) -> GenerationResult:
        text = original_text if original_text is not None else ""
        slots = slot_dict if slot_dict is not None else {}
        raw = json.dumps({"rewritten_text": text, "preserved_slots": slots})
        return GenerationResult(
            rewritten_text=text,
            preserved_slots=slots,
            raw=raw,
            used_mock=True,
        )


class AnthropicGenerator:
    """Thin Anthropic client with retry-with-backoff."""

    used_mock = False

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-latest",
        max_tokens: int = 2048,
        temperature: float = 0.4,
        max_retries: int = 4,
        base_delay_s: float = 1.5,
    ):
        if not _HAS_ANTHROPIC:
            raise RuntimeError("anthropic package not installed")
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.base_delay_s = base_delay_s

    def generate(self, prompt: str, slot_dict: Optional[dict] = None,
                 original_text: Optional[str] = None, **_: object) -> GenerationResult:
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = "".join(
                    block.text for block in resp.content
                    if getattr(block, "type", None) == "text"
                )
                payload = _parse_json_payload(raw)
                return GenerationResult(
                    rewritten_text=str(payload.get("rewritten_text", "")),
                    preserved_slots=dict(payload.get("preserved_slots", {})),
                    raw=raw,
                    used_mock=False,
                )
            except Exception as e:  # noqa: BLE001 — retry-anything for transient API errors
                last_err = e
                delay = self.base_delay_s * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(delay)
        raise RuntimeError(f"AnthropicGenerator failed after {self.max_retries} attempts: {last_err}")


def make_generator(force_mock: bool = False) -> MockGenerator | AnthropicGenerator:
    """Return the real generator if a key is set and the SDK is available;
    otherwise the mock. Never fails on missing key."""
    if force_mock or not _HAS_ANTHROPIC or not os.environ.get("ANTHROPIC_API_KEY"):
        if not _HAS_ANTHROPIC:
            print("  [generator] anthropic package unavailable — using MockGenerator")
        elif not os.environ.get("ANTHROPIC_API_KEY"):
            print("  [generator] ANTHROPIC_API_KEY not set — using MockGenerator")
        return MockGenerator()
    return AnthropicGenerator()
