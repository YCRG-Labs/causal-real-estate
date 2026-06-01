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
            except (anthropic.APIError, ValueError) as e:
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
            except (anthropic.APIError, ValueError) as e:
                # Retry transient API errors + JSON parse failures only;
                # programmer errors (TypeError, AttributeError, ...) fail fast.
                last_err = e
                delay = self.base_delay_s * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(delay)
        raise RuntimeError(
            f"AnthropicGenerator failed after {self.max_retries} attempts: {last_err}"
        )


class VLLMGenerator:
    """Offline vLLM generator with Outlines-style guided JSON decoding.

    Loads the model into GPU memory once (vLLM offline mode), then serves
    generate() / generate_blocks() / generate_batch() calls against the
    resident model. Default model is Qwen 2.5 32B Instruct AWQ which fits on
    a single 48GB GPU at 8k context.

    Reproducibility: temperature=0, seed pinned, vLLM offline mode (no
    scheduling non-determinism), version-pinned dependency. Per
    https://docs.vllm.ai/en/v0.9.1/usage/reproducibility.html online serving
    is not reproducible; offline LLM is.

    Slot-fact preservation: when a slot_dict is supplied, we build a JSON
    schema whose 'preserved_slots' field has one Literal-typed property per
    slot key, forcing the decoder to emit the exact slot values. This is
    the JBES-defensible alternative to post-hoc regex validation +
    rejection sampling.
    """

    used_mock = False

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-32B-Instruct-AWQ",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        seed: int = 42,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.90,
        quantization: str | None = "awq",
        enable_prefix_caching: bool = True,
    ):
        try:
            from vllm import LLM, SamplingParams  # type: ignore
            from vllm.sampling_params import GuidedDecodingParams  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "vllm not installed; run: pip install 'vllm>=0.9.1' outlines"
            ) from e
        self._SamplingParams = SamplingParams
        self._GuidedDecodingParams = GuidedDecodingParams
        self.model_name = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        kwargs: dict = dict(
            model=model,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
            seed=seed,
            dtype="auto",
        )
        if quantization:
            kwargs["quantization"] = quantization
        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        self.usage = UsageStats()
        # vLLM offline has no API usage to bill; we count requests and
        # output_tokens for the orchestrator's summary line.

    @staticmethod
    def _slot_schema(slot_dict: Optional[dict]) -> dict | None:
        if not slot_dict:
            return None
        slot_props: dict = {}
        slot_required: list[str] = []
        for k, v in slot_dict.items():
            v_str = "" if v is None else str(v)
            slot_props[str(k)] = {"type": "string", "enum": [v_str]}
            slot_required.append(str(k))
        return {
            "type": "object",
            "properties": {
                "rewritten_text": {"type": "string"},
                "preserved_slots": {
                    "type": "object",
                    "properties": slot_props,
                    "required": slot_required,
                    "additionalProperties": False,
                },
            },
            "required": ["rewritten_text", "preserved_slots"],
            "additionalProperties": False,
        }

    def _sampling_params(self, slot_dict: Optional[dict]):
        schema = self._slot_schema(slot_dict)
        guided = None
        if schema is not None:
            guided = self._GuidedDecodingParams(json=schema)
        return self._SamplingParams(
            temperature=self.temperature,
            top_p=1.0,
            max_tokens=self.max_tokens,
            seed=self.seed,
            guided_decoding=guided,
        )

    def _apply_chat_template(self, system: Optional[str], user: str) -> str:
        msgs: list[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    def _wrap_output(self, raw: str, slot_dict: Optional[dict]) -> GenerationResult:
        payload = _parse_json_payload(raw)
        slots = dict(payload.get("preserved_slots", slot_dict or {}))
        self.usage.n_calls += 1
        # rough token accounting from string length, no tokenizer round-trip per call
        self.usage.output_tokens += len(raw) // 4
        return GenerationResult(
            rewritten_text=str(payload.get("rewritten_text", "")),
            preserved_slots=slots,
            raw=raw,
            used_mock=False,
        )

    def generate(self, prompt: str, slot_dict: Optional[dict] = None,
                 original_text: Optional[str] = None, **_: object) -> GenerationResult:
        params = self._sampling_params(slot_dict)
        outputs = self.llm.generate([prompt], params, use_tqdm=False)
        raw = outputs[0].outputs[0].text
        return self._wrap_output(raw, slot_dict)

    def generate_blocks(self, system: str, user: str,
                        slot_dict: Optional[dict] = None,
                        original_text: Optional[str] = None,
                        **_: object) -> GenerationResult:
        prompt = self._apply_chat_template(system, user)
        return self.generate(prompt, slot_dict=slot_dict,
                             original_text=original_text)

    def generate_batch(
        self,
        items: list[dict],
    ) -> list[GenerationResult]:
        """Batch-generate over a list of {prompt, slot_dict, system, user} dicts.

        Either ``prompt`` (single-string path) or both ``system`` and ``user``
        (block path) must be supplied per item. Each item carries its own
        slot_dict so the per-call guided schema differs across items in the
        batch. vLLM continuous-batches these on the GPU; this is the only
        code path that hits the published 200-400 tok/s throughput.
        """
        prompts: list[str] = []
        params_list: list = []
        slot_dicts: list[Optional[dict]] = []
        for it in items:
            sd = it.get("slot_dict")
            if "prompt" in it:
                prompts.append(it["prompt"])
            elif "system" in it and "user" in it:
                prompts.append(self._apply_chat_template(it["system"], it["user"]))
            else:
                raise ValueError("batch item needs 'prompt' or ('system','user')")
            params_list.append(self._sampling_params(sd))
            slot_dicts.append(sd)
        outputs = self.llm.generate(prompts, params_list, use_tqdm=False)
        return [
            self._wrap_output(o.outputs[0].text, sd)
            for o, sd in zip(outputs, slot_dicts)
        ]


def make_generator(force_mock: bool = False,
                   use_vllm: bool = False,
                   vllm_model: str | None = None,
                   ) -> MockGenerator | AnthropicGenerator | VLLMGenerator:
    """Return the real generator. Priority: vLLM > Anthropic > Mock.

    use_vllm=True forces the VLLMGenerator path (requires CUDA + vllm
    installed); falls back to Mock if vllm cannot be loaded. Otherwise the
    pre-existing Anthropic/Mock router is preserved.
    """
    if use_vllm:
        try:
            return VLLMGenerator(model=vllm_model or "Qwen/Qwen2.5-32B-Instruct-AWQ")
        except Exception as e:
            print(f"  [generator] vLLM init failed ({e}); falling back to mock")
            return MockGenerator()
    if force_mock or not _HAS_ANTHROPIC or not os.environ.get("ANTHROPIC_API_KEY"):
        if not _HAS_ANTHROPIC:
            print("  [generator] anthropic package unavailable — using MockGenerator")
        elif not os.environ.get("ANTHROPIC_API_KEY"):
            print("  [generator] ANTHROPIC_API_KEY not set — using MockGenerator")
        return MockGenerator()
    return AnthropicGenerator()


# -----------------------------------------------------------------------------
# Async variants: AsyncMockGenerator + AsyncAnthropicGenerator
# Used by run_counterfactual to fire all 4 arms per listing concurrently via
# asyncio.gather. Each arm gets its own API request; the 4 requests share the
# same cached system prompt (prompt caching unaffected by concurrency).
# Net effect at 100 listings: ~4x speedup on API-bound time (max(4) vs sum(4)).
# -----------------------------------------------------------------------------


class AsyncMockGenerator(MockGenerator):
    async def generate_blocks(self, system: str, user: str,
                              slot_dict: Optional[dict] = None,
                              original_text: Optional[str] = None,
                              **_: object) -> GenerationResult:
        return super().generate_blocks(system, user, slot_dict=slot_dict,
                                       original_text=original_text)


class AsyncAnthropicGenerator:
    """Async Anthropic client. Fires concurrent requests via asyncio.gather."""

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
        self.client = anthropic.AsyncAnthropic()
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

    async def generate_blocks(self, system: str, user: str,
                              slot_dict: Optional[dict] = None,
                              original_text: Optional[str] = None,
                              **_: object) -> GenerationResult:
        import asyncio
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = await self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=[{
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }],
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
            except (anthropic.APIError, ValueError) as e:
                last_err = e
                delay = self.base_delay_s * (2 ** attempt) + random.uniform(0, 0.5)
                await asyncio.sleep(delay)
        raise RuntimeError(
            f"AsyncAnthropicGenerator failed after {self.max_retries} attempts: {last_err}"
        )


class AsyncVLLMGenerator:
    """Async wrapper around VLLMGenerator using asyncio.to_thread.

    vLLM offline LLM is a sync API and serializes calls per-process. The
    wrapper preserves the existing async pipeline so the 4 arms per listing
    still gang up under asyncio.gather, but the underlying execution is
    serial through vLLM's internal continuous batcher. For real throughput
    at 24k generations, prefer driving generate_batch directly on the sync
    VLLMGenerator (see notes in generator.py:VLLMGenerator).
    """

    used_mock = False

    def __init__(self, model: str | None = None, **vllm_kwargs):
        self._sync = VLLMGenerator(
            model=model or "Qwen/Qwen2.5-32B-Instruct-AWQ", **vllm_kwargs
        )
        self.usage = self._sync.usage

    async def generate_blocks(self, system: str, user: str,
                              slot_dict: Optional[dict] = None,
                              original_text: Optional[str] = None,
                              **_: object) -> GenerationResult:
        import asyncio
        return await asyncio.to_thread(
            self._sync.generate_blocks,
            system, user, slot_dict=slot_dict, original_text=original_text,
        )


def make_async_generator(force_mock: bool = False,
                         use_vllm: bool = False,
                         vllm_model: str | None = None):
    """Return the async generator. Same key/install rules as make_generator.

    Priority: vLLM > Anthropic > Mock. use_vllm=True forces the
    AsyncVLLMGenerator path; falls back to Mock if vllm cannot be loaded.
    """
    if use_vllm:
        try:
            return AsyncVLLMGenerator(model=vllm_model)
        except Exception as e:
            print(f"  [generator] vLLM init failed ({e}); falling back to AsyncMockGenerator")
            return AsyncMockGenerator()
    if force_mock or not _HAS_ANTHROPIC or not os.environ.get("ANTHROPIC_API_KEY"):
        if not _HAS_ANTHROPIC:
            print("  [generator] anthropic package unavailable — using AsyncMockGenerator")
        elif not os.environ.get("ANTHROPIC_API_KEY"):
            print("  [generator] ANTHROPIC_API_KEY not set — using AsyncMockGenerator")
        return AsyncMockGenerator()
    return AsyncAnthropicGenerator()
