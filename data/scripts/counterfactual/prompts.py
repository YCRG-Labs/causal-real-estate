"""Prompt templates for counterfactual listing rewrites.

Two templates: style_swap and style_stripped. Both follow the prompt-engineering
recipe from the dossier (Madaan et al. 2021; Dixit et al. CORE 2022; Bhattacharjee
& Liu 2024):

  - explicit "preserve X" / "change Y" constraint lists
  - slot-fact JSON the model must verify against
  - structured JSON output schema (`rewritten_text`, `preserved_slots`)
  - chain-of-thought verification step inlined in the instructions

Output schema is JSON-only so the generator can parse without regex hacks.
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


def _format_slots(slot_dict: dict[str, Optional[float]]) -> str:
    """JSON-serialize slots with None preserved as null."""
    return json.dumps({k: v for k, v in slot_dict.items()}, indent=2)


def style_swap_prompt(
    target_submarket: str,
    original_text: str,
    slot_dict: dict[str, Optional[float]],
) -> str:
    """Rewrite as if the listing were located in `target_submarket`, preserving
    all numeric property facts. Style/lexicon must shift to the submarket
    norm; numeric content must NOT shift."""
    hint = SUBMARKET_HINTS.get(
        target_submarket,
        f"the {target_submarket} neighborhood of San Francisco",
    )
    return (
        "You are a real-estate copywriter producing a counterfactual listing "
        "rewrite for an academic causal-inference experiment.\n\n"
        f"TASK: Rewrite the listing below as if the property were located in "
        f"{target_submarket}, San Francisco.\n\n"
        f"TARGET SUBMARKET STYLE NOTES (use this lexicon and vibe):\n  {hint}\n\n"
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
        "  - change any fact in the slot-fact JSON below\n"
        "  - retain any landmark or street name from the original\n\n"
        f"SLOT-FACT JSON (must be preserved verbatim):\n{_format_slots(slot_dict)}\n\n"
        f"ORIGINAL LISTING:\n\"\"\"\n{original_text}\n\"\"\"\n\n"
        f"{_VERIFY_STEP}\n\n"
        f"{_OUTPUT_INSTR}"
    )


def style_stripped_prompt(
    original_text: str,
    slot_dict: dict[str, Optional[float]],
) -> str:
    """Rewrite removing every neighborhood-evocative cue while preserving all
    numeric property facts. This is the Natural-Direct-Effect arm: facts held
    fixed, style set to neutral baseline."""
    return (
        "You are a real-estate copywriter producing a NEUTRALIZED counterfactual "
        "listing rewrite for an academic causal-inference experiment.\n\n"
        "TASK: Rewrite the listing below in a flat, neutral, fact-forward tone "
        "that strips ALL neighborhood-evocative language while preserving every "
        "numeric property fact.\n\n"
        "PRESERVE EXACTLY (must appear with identical numeric values in the rewrite):\n"
        "  - bedroom count, bathroom count, square footage\n"
        "  - year built, lot size, parking capacity, story count (if present)\n"
        "  - structural facts (room layout, construction features)\n\n"
        "STRIP / REMOVE (this is the counterfactual treatment):\n"
        "  - all neighborhood names and submarket references\n"
        "  - all landmark, street, park, and transit-line names\n"
        "  - all aspirational lifestyle lexicon (\"vibrant\", \"prestigious\", \"sun-drenched\", \"coveted\", etc.)\n"
        "  - all view claims that imply geography (\"ocean views\", \"Bay views\")\n"
        "  - all cultural / demographic cues\n"
        "  - all price-tier signals (\"luxury\", \"entry-level\", \"investment opportunity\")\n\n"
        "TONE: dry MLS-style enumeration of physical specifications. Short "
        "sentences. No adjectives that signal neighborhood prestige.\n\n"
        "DO NOT:\n"
        "  - mention San Francisco or any neighborhood\n"
        "  - invent new numeric facts\n"
        "  - change any fact in the slot-fact JSON below\n\n"
        f"SLOT-FACT JSON (must be preserved verbatim):\n{_format_slots(slot_dict)}\n\n"
        f"ORIGINAL LISTING:\n\"\"\"\n{original_text}\n\"\"\"\n\n"
        f"{_VERIFY_STEP}\n\n"
        f"{_OUTPUT_INSTR}"
    )
