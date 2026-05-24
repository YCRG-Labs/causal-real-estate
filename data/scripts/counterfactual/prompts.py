"""Prompt templates for counterfactual listing rewrites.

Two templates: style_swap and style_stripped. Both follow the prompt-engineering
recipe from the dossier (Madaan et al. 2021; Dixit et al. CORE 2022; Bhattacharjee
& Liu 2024):

  - explicit "preserve X" / "change Y" constraint lists
  - slot-fact JSON the model must verify against
  - structured JSON output schema (`rewritten_text`, `preserved_slots`)
  - chain-of-thought verification step inlined in the instructions

Output schema is JSON-only so the generator can parse without regex hacks.

Two API surfaces:

  Legacy (uncached):  style_swap_prompt() / style_stripped_prompt()
                      Single-string prompts. Backward compatible.

  Cached (preferred): style_swap_blocks() / style_stripped_blocks()
                      Returns {"system": <constant>, "user": <variable>} so the
                      generator can mark the system prompt with cache_control:
                      ephemeral. Anthropic Sonnet 3.5 caches at 1024 token
                      minimum; the system templates here are sized to clear it.
"""
from __future__ import annotations

import json
from typing import Optional

# Style lexicons keyed to SF submarkets, used as control codes.
SUBMARKET_HINTS: dict[str, str] = {
    "Mission District": (
        "vibrant Latinx culture, mural-lined alleys, taquerias, Dolores Park, "
        "Valencia Street nightlife, victorian and edwardian flats, walkable, "
        "BART-adjacent, working-class roots blending with newer cafes"
    ),
    "Pacific Heights": (
        "elite mansion district, panoramic Bay views, Lyon Street stairs, "
        "Fillmore Street boutiques, manicured garden squares, prestigious, "
        "stately period architecture, blue-chip address"
    ),
    "Sunset": (
        "fog-cooled residential calm, classic two-story stucco homes, family "
        "neighborhood, Ocean Beach, Golden Gate Park access, quiet avenues, "
        "Asian-American community, value-oriented vs. east-side prices"
    ),
    "SoMa": (
        "live/work conversions, tech-corridor energy, modern condo towers, "
        "warehouse loft aesthetic, walkable to Caltrain and Salesforce Park, "
        "boutique restaurants, dynamic urban grit"
    ),
    "Noe Valley": (
        "stroller-friendly Castro-adjacent village feel, sun-pocket microclimate, "
        "24th Street shopping, edwardian single-family homes, family-oriented, "
        "premium school district, quietly affluent"
    ),
    "Castro": (
        "iconic LGBTQ+ neighborhood, rainbow-flag Castro Street, victorian flats, "
        "walkable cafe and nightlife scene, Harvey Milk legacy, vibrant social fabric"
    ),
    "Marina": (
        "waterfront Marina Green and Crissy Field, Mediterranean revival "
        "architecture, Chestnut Street boutiques, athleisure energy, young "
        "professional crowd, sweeping Bay and Golden Gate views"
    ),
    "Richmond": (
        "Pan-Asian culinary corridor along Clement Street, Golden Gate Park "
        "frontage, foggy residential calm, classic SF row houses, established "
        "immigrant communities"
    ),
}

_OUTPUT_INSTR = (
    "Return ONLY a single JSON object with this exact schema and no other "
    "prose, code fences, or commentary:\n"
    "{\n"
    '  "rewritten_text": "<full rewritten listing description, plain text>",\n'
    '  "preserved_slots": {<copy of the slot-fact JSON you were given verbatim>}\n'
    "}\n"
    "If you cannot preserve a fact verbatim, set rewritten_text to the empty "
    "string and explain nothing — leave preserved_slots empty too."
)

_VERIFY_STEP = (
    "Before writing the final JSON, internally walk through every numeric "
    "fact in the slot-fact JSON and confirm it appears with the SAME value in "
    "your rewrite. Do not include this reasoning in your output — only the "
    "final JSON object."
)

# Quality and pitfall guidance shared by both system prompts. Pulled out so it
# is part of the cached prefix and not part of the per-call user content.
_QUALITY_CRITERIA = (
    "QUALITY CRITERIA — a successful counterfactual rewrite:\n"
    "  - Numeric facts: every value in the slot-fact JSON appears unchanged.\n"
    "  - Style shift: a knowledgeable local could identify the target submarket\n"
    "    from the rewrite alone, without seeing the slot-fact JSON.\n"
    "  - Length: within ±20% of the original word count.\n"
    "  - Register: matches the polish of the original (MLS-formal vs casual).\n"
    "  - No double-claiming: do not assert prestige cues that were not in the\n"
    "    original. The rewrite should plausibly substitute for the original,\n"
    "    not upsell it."
)

_PITFALLS = (
    "COMMON FAILURE MODES TO AVOID:\n"
    "  - Generic openers (\"In the heart of [neighborhood]...\") — these are\n"
    "    tells of LLM-generated copy and dilute the submarket signal.\n"
    "  - Inventing transit lines, parks, schools, or businesses that do not\n"
    "    exist near the target submarket.\n"
    "  - Substituting a numeric fact (\"3 bedrooms\" → \"spacious bedrooms\")\n"
    "    to avoid repeating the slot — preserve the number explicitly.\n"
    "  - Adding new amenities (gym, pool, doorman) not in the original.\n"
    "  - Reusing landmark names from the original (Dolores Park, Crissy\n"
    "    Field, etc.) when the rewrite targets a different submarket."
)

_VERIFY_CHECKLIST = (
    "VERIFICATION CHECKLIST (run mentally before output):\n"
    "  1. Read each entry in the slot-fact JSON. For each entry, find the\n"
    "     identical value in your rewrite. If any is missing, REWRITE before\n"
    "     emitting.\n"
    "  2. Scan for any neighborhood name or street from the original. If\n"
    "     found, replace with a target-submarket equivalent (or remove for\n"
    "     style_stripped).\n"
    "  3. Confirm the rewrite would be plausible from a real-estate agent\n"
    "     working that neighborhood (or a flat MLS feed for style_stripped).\n"
    "     If it reads as foreign, revise the vocabulary."
)

_SWAP_EXAMPLE = (
    "WORKED EXAMPLE (Mission District → Pacific Heights swap):\n"
    "  Slot-fact JSON: {\"bedrooms\": 2, \"bathrooms\": 1, \"sqft\": 1100, "
    "\"year_built\": 1908, \"parking\": null}\n"
    "  Original: \"Charming 2-bedroom Victorian flat steps from Dolores Park "
    "and 24th Street taquerias. 1100 sqft, 1 bath, original 1908 millwork. "
    "Walk to BART. Mission's vibrant cafe scene at your door.\"\n"
    "  Good rewrite: \"Stately 2-bedroom Edwardian flat just off Lyon Street, "
    "walking distance to Fillmore Street boutiques. 1100 sqft, 1 bath, "
    "original 1908 millwork. Sweeping Bay views from the parlor floor. "
    "Refined Pacific Heights address with mature street trees.\"\n"
    "  Why it works: every slot-fact value preserved verbatim (2, 1, 1100, "
    "1908); Mission landmarks (Dolores Park, 24th Street, BART) replaced\n"
    "  with Pacific Heights equivalents (Lyon Street, Fillmore Street, Bay\n"
    "  views); register elevated to match the destination submarket\n"
    "  without inventing facts that were not in the original.\n"
    "  Bad rewrite (REJECT): \"In the heart of prestigious Pacific Heights, "
    "this stunning 2-bedroom Victorian masterpiece offers Dolores Park "
    "views, 1100+ sqft, with luxury parking included.\" Why it fails: "
    "generic opener, retains \"Dolores Park\" from origin submarket, "
    "embellishes square footage with \"+\", invents parking that the "
    "slot-fact JSON marks as null, and asserts unsupported prestige."
)

_STRIPPED_EXAMPLE = (
    "WORKED EXAMPLE (style_stripped):\n"
    "  Slot-fact JSON: {\"bedrooms\": 2, \"bathrooms\": 1, \"sqft\": 1100, "
    "\"year_built\": 1908, \"parking\": null}\n"
    "  Original: \"Charming 2-bedroom Victorian flat steps from Dolores Park "
    "and 24th Street taquerias. 1100 sqft, 1 bath, original 1908 millwork. "
    "Walk to BART. Mission's vibrant cafe scene at your door.\"\n"
    "  Good rewrite: \"2-bedroom flat. 1100 sqft. 1 bath. Built 1908. "
    "Original millwork. No parking. Single-level layout above ground floor.\"\n"
    "  Why it works: every slot-fact value preserved verbatim (2, 1, 1100, "
    "1908); all neighborhood references (Dolores Park, 24th Street, BART,\n"
    "  Mission), all aspirational lexicon (\"charming\", \"vibrant\"), and\n"
    "  all geographic cues removed; tone reduced to MLS-style enumeration\n"
    "  without inventing new facts.\n"
    "  Bad rewrite (REJECT): \"Beautifully maintained 2-bedroom residence "
    "in a desirable urban setting. 1100+ sqft of charming living space. "
    "1 bath. Built in 1908. Excellent transit access. Move-in ready.\" "
    "Why it fails: retains aspirational lexicon (\"beautifully\", "
    "\"charming\", \"desirable\"), implies geography (\"urban\", "
    "\"transit access\"), embellishes square footage with \"+\", and "
    "asserts prestige cues (\"move-in ready\") not in the original."
)


def _format_slots(slot_dict: dict[str, Optional[float]]) -> str:
    """JSON-serialize slots with None preserved as null."""
    return json.dumps({k: v for k, v in slot_dict.items()}, indent=2)


# ---------------------------------------------------------------------------
# Cacheable system / variable user split (preferred)
# ---------------------------------------------------------------------------

def style_swap_system() -> str:
    """Constant instruction template for style_swap. Cacheable.

    Does NOT include the target submarket name, hint, slot dict, or original
    text — those vary per call and live in the user message.
    """
    return (
        "You are a real-estate copywriter producing a counterfactual listing "
        "rewrite for an academic causal-inference experiment.\n\n"
        "GENERAL TASK: Rewrite a property listing as if the property were "
        "located in a different submarket of San Francisco. Preserve every "
        "numeric property fact verbatim; replace every submarket-evocative "
        "cue with an equivalent for the target submarket.\n\n"
        "PRESERVE EXACTLY (must appear with identical numeric values in the rewrite):\n"
        "  - bedroom count\n"
        "  - bathroom count\n"
        "  - square footage\n"
        "  - year built\n"
        "  - lot size (if present)\n"
        "  - parking / garage capacity (if present)\n"
        "  - number of stories / levels (if present)\n\n"
        "CHANGE (this is the counterfactual treatment):\n"
        "  - all neighborhood names, landmarks, and street references\n"
        "  - all submarket-evocative lexicon (vibe words, cultural cues, view claims)\n"
        "  - implied price tier and prestige cues\n"
        "  - the target submarket should be unambiguous to a local reader\n\n"
        "DO NOT:\n"
        "  - invent new numeric facts\n"
        "  - change any fact in the slot-fact JSON the user provides\n"
        "  - retain any landmark or street name from the original\n\n"
        f"{_QUALITY_CRITERIA}\n\n"
        f"{_PITFALLS}\n\n"
        f"{_SWAP_EXAMPLE}\n\n"
        f"{_VERIFY_CHECKLIST}\n\n"
        f"{_VERIFY_STEP}\n\n"
        f"{_OUTPUT_INSTR}"
    )


def style_swap_user(
    target_submarket: str,
    original_text: str,
    slot_dict: dict[str, Optional[float]],
) -> str:
    """Variable per-call content for style_swap. NOT cacheable."""
    hint = SUBMARKET_HINTS.get(
        target_submarket,
        f"the {target_submarket} neighborhood of San Francisco",
    )
    return (
        f"TARGET SUBMARKET: {target_submarket}, San Francisco.\n\n"
        f"TARGET SUBMARKET STYLE NOTES (use this lexicon and vibe):\n  {hint}\n\n"
        f"SLOT-FACT JSON (must be preserved verbatim):\n{_format_slots(slot_dict)}\n\n"
        f"ORIGINAL LISTING:\n\"\"\"\n{original_text}\n\"\"\""
    )


def style_swap_blocks(
    target_submarket: str,
    original_text: str,
    slot_dict: dict[str, Optional[float]],
) -> dict[str, str]:
    return {
        "system": style_swap_system(),
        "user": style_swap_user(target_submarket, original_text, slot_dict),
    }


def style_stripped_system() -> str:
    """Constant instruction template for style_stripped. Cacheable."""
    return (
        "You are a real-estate copywriter producing a NEUTRALIZED counterfactual "
        "listing rewrite for an academic causal-inference experiment.\n\n"
        "GENERAL TASK: Rewrite a property listing in a flat, neutral, fact-"
        "forward tone that strips ALL neighborhood-evocative language while "
        "preserving every numeric property fact.\n\n"
        "PRESERVE EXACTLY (must appear with identical numeric values in the rewrite):\n"
        "  - bedroom count, bathroom count, square footage\n"
        "  - year built, lot size, parking capacity, story count (if present)\n"
        "  - structural facts (room layout, construction features)\n\n"
        "STRIP / REMOVE (this is the counterfactual treatment):\n"
        "  - all neighborhood names and submarket references\n"
        "  - all landmark, street, park, and transit-line names\n"
        "  - all aspirational lifestyle lexicon (\"vibrant\", \"prestigious\", "
        "\"sun-drenched\", \"coveted\", etc.)\n"
        "  - all view claims that imply geography (\"ocean views\", \"Bay views\")\n"
        "  - all cultural / demographic cues\n"
        "  - all price-tier signals (\"luxury\", \"entry-level\", "
        "\"investment opportunity\")\n\n"
        "TONE: dry MLS-style enumeration of physical specifications. Short "
        "sentences. No adjectives that signal neighborhood prestige.\n\n"
        "DO NOT:\n"
        "  - mention San Francisco or any neighborhood\n"
        "  - invent new numeric facts\n"
        "  - change any fact in the slot-fact JSON the user provides\n\n"
        f"{_QUALITY_CRITERIA}\n\n"
        f"{_PITFALLS}\n\n"
        f"{_STRIPPED_EXAMPLE}\n\n"
        f"{_VERIFY_CHECKLIST}\n\n"
        f"{_VERIFY_STEP}\n\n"
        f"{_OUTPUT_INSTR}"
    )


def style_stripped_user(
    original_text: str,
    slot_dict: dict[str, Optional[float]],
) -> str:
    """Variable per-call content for style_stripped. NOT cacheable."""
    return (
        f"SLOT-FACT JSON (must be preserved verbatim):\n{_format_slots(slot_dict)}\n\n"
        f"ORIGINAL LISTING:\n\"\"\"\n{original_text}\n\"\"\""
    )


def style_stripped_blocks(
    original_text: str,
    slot_dict: dict[str, Optional[float]],
) -> dict[str, str]:
    return {
        "system": style_stripped_system(),
        "user": style_stripped_user(original_text, slot_dict),
    }


# ---------------------------------------------------------------------------
# Legacy single-string prompts (kept for backward compat; uncached)
# ---------------------------------------------------------------------------

def style_swap_prompt(
    target_submarket: str,
    original_text: str,
    slot_dict: dict[str, Optional[float]],
) -> str:
    """Single-string form, retained for backward compat. Prefer style_swap_blocks."""
    blocks = style_swap_blocks(target_submarket, original_text, slot_dict)
    return f"{blocks['system']}\n\n{blocks['user']}"


def style_stripped_prompt(
    original_text: str,
    slot_dict: dict[str, Optional[float]],
) -> str:
    """Single-string form, retained for backward compat. Prefer style_stripped_blocks."""
    blocks = style_stripped_blocks(original_text, slot_dict)
    return f"{blocks['system']}\n\n{blocks['user']}"
