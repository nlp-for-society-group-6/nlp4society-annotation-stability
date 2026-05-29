"""Stage 3 (parsing half): turn raw_text into a clean binary label.

Kept separate from inference on purpose: parsing logic can change without
re-running any (expensive, rate-limited) inference, because raw_text is logged
verbatim. Run this over raw_outputs.jsonl whenever the rules improve.

Even with response_format=json_object and "No explanation", small models
occasionally emit fences, prose, or odd casing. Parse defensively and record
WHY something failed rather than silently coercing, so unparseable outputs
become a visible data-quality number, not hidden noise.
"""
from __future__ import annotations

import json
import re
from typing import Optional

LABELS = {"hate", "not_hate"}

# tolerant variants the parser will normalise to the canonical labels
_NORMALISE = {
    "hate": "hate", "hateful": "hate", "hate_speech": "hate", "1": "hate",
    "not_hate": "not_hate", "nothate": "not_hate", "not hate": "not_hate",
    "non_hate": "not_hate", "non-hate": "not_hate", "0": "not_hate",
    "neutral": "not_hate", "none": "not_hate",
}


def parse_label(raw_text: str) -> tuple[Optional[str], str]:
    """Return (label_or_None, reason).

    reason is "" on success, else a short tag ("empty", "no_json",
    "bad_value", "unparseable") for data-quality accounting.
    """
    if not raw_text or not raw_text.strip():
        return None, "empty"

    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.IGNORECASE).strip()

    # 1) try to find and parse a JSON object
    m = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            val = str(obj.get("label", "")).strip().lower()
            norm = _NORMALISE.get(val)
            if norm in LABELS:
                return norm, ""
            return None, "bad_value"
        except json.JSONDecodeError:
            pass  # fall through to keyword scan

    # 2) fallback: scan for a bare label keyword
    low = text.lower()
    if "not_hate" in low or "not hate" in low or "non-hate" in low:
        return "not_hate", ""
    if "hate" in low:
        return "hate", ""

    return None, "unparseable"
