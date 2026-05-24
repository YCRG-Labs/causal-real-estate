"""Anthropic SDK wrapper with a mock fallback so the pipeline runs without a key.

If `ANTHROPIC_API_KEY` is set AND the `anthropic` package imports cleanly, we
use the real client (Claude 3.5 Sonnet by default) with retry-with-backoff on
transient API errors. Otherwise we drop in a `MockGenerator` that returns the
original text wrapped in the expected JSON shape — sufficient for end-to-end
dry-runs and the smoke test.

The mock IS the default when no key is present. This module never raises on
import for missing keys or missing package.

Two API surfaces:

  generate(prompt, ...)
    Single-string prompt path. Backward compatible. No prompt caching — the
    full prompt is sent uncached each call.

  generate_blocks(system, user, ...)
    Structured prompt path. Sends `system` as a cacheable text block with
    cache_control: ephemeral. Subsequent calls within the 5-minute cache
    window read the system prompt at ~10% input price (~90% cost reduction
    on the cached portion). Verify with usage.cache_read_input_tokens.

Usage tracking: AnthropicGenerator aggregates the per-call usage object
across every call so the orchestrator can print a final cache-hit summary.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
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


@dataclass
class UsageStats:
    """Aggregated token usage across an entire run."""
    n_calls: int = 0
    input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0

    def add(self, usage_obj: object) -> None:
        self.n_calls += 1
        self.input_tokens += int(getattr(usage_obj, "input_tokens", 0) or 0)
        self.cache_creation_input_tokens += int(
            getattr(usage_obj, "cache_creation_input_tokens", 0) or 0
        )
        self.cache_read_input_tokens += int(
            getattr(usage_obj, "cache_read_input_tokens", 0) or 0
        )
        self.output_tokens += int(getattr(usage_obj, "output_tokens", 0) or 0)

    def estimated_cost_usd(
        self,
        input_per_mtok: float = 3.00,
        output_per_mtok: float = 15.00,
    ) -> float:
        """Estimate cost using public Sonnet 3.5/4.x pricing.

        Cache write costs 1.25x input, cache read costs 0.10x input.
        """
        cost = (
            self.input_tokens * input_per_mtok
            + self.cache_creation_input_tokens * input_per_mtok * 1.25
            + self.cache_read_input_tokens * input_per_mtok * 0.10
            + self.output_tokens * output_per_mtok
        ) / 1_000_000.0
        return float(cost)

    def cache_hit_rate(self) -> float:
        cached_eligible = self.cache_creation_input_tokens + self.cache_read_input_tokens
        if cached_eligible == 0:
            return 0.0
        return self.cache_read_input_tokens / cached_eligible

    def summary(self) -> str:
        return (
            f"  calls={self.n_calls}\n"
            f"  uncached input tokens:  {self.input_tokens:>10,}\n"
            f"  cache creation tokens:  {self.cache_creation_input_tokens:>10,}  (~1.25x input price)\n"
            f"  cache read tokens:      {self.cache_read_input_tokens:>10,}  (~0.10x input price)\n"
            f"  output tokens:          {self.output_tokens:>10,}\n"
            f"  cache hit rate (cached eligible): {self.cache_hit_rate() * 100:.1f}%\n"
            f"  estimated cost @ Sonnet 3.5 pricing: ${self.estimated_cost_usd():.4f}"
        )


def _parse_json_payload(raw: str) -> dict:
    """Robustly pull the first JSON object out of a model response."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
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

    def __init__(self) -> None:
        self.usage = UsageStats()

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

    def generate_blocks(self, system: str, user: str,
                        slot_dict: Optional[dict] = None,
                        original_text: Optional[str] = None,
                        **_: object) -> GenerationResult:
        return self.generate(system + "\n\n" + user, slot_dict=slot_dict,
                             original_text=original_text)


class AnthropicGenerator:
    """Thin Anthropic client with retry-with-backoff and prompt caching."""

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
        self.usage = UsageStats()

    def _record_usage(self, response: object) -> None:
        try:
            self.usage.add(response.usage)  # type: ignore[attr-defined]
        except Exception:
            pass

    def generate(self, prompt: str, slot_dict: Optional[dict] = None,
                 original_text: Optional[str] = None, **_: object) -> GenerationResult:
        """Single-string prompt path. No caching."""
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                self._record_usage(resp)
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

    def generate_blocks(
        self,
        system: str,
        user: str,
        slot_dict: Optional[dict] = None,
        original_text: Optional[str] = None,
        **_: object,
    ) -> GenerationResult:
        """Structured prompt path with prompt caching on the system block.

        The system prompt is sent as a single text block tagged with
        cache_control: ephemeral. The first call within a 5-minute window
        writes the cache (1.25x input price); subsequent calls with an
        identical system prompt read it (0.10x input price).

        For caching to fire reliably, the system block must clear the
        per-model token minimum (1024 tokens for Sonnet 3.5). prompts.py
        sizes its system prompts above this threshold.
        """
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=[
                        {
                            "type": "text",
                            "text": system,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": user}],
                )
                self._record_usage(resp)
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
            except Exception as e:  # noqa: BLE001
                last_err = e
                delay = self.base_delay_s * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(delay)
        raise RuntimeError(
            f"AnthropicGenerator failed after {self.max_retries} attempts: {last_err}"
        )


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
