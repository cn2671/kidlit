"""Query parsing utilities for KidLit.

This module extracts coarse filters (age_range, tone, themes) from a free‑text
user request. It uses a deterministic pass first, and only falls back to an LLM
when both tone and themes are empty.

"""
from __future__ import annotations
import json
import re
from typing import Any, Dict, List
import textwrap

from scripts.core.config import get_openai_client
from scripts.core.text_utils import (
    BASE_STOPWORDS as STOPWORDS,
    detect_age_from_text,
)

# Initialize OpenAI client 
client = get_openai_client()

# -----------------------------------------------------------------------------
# Configs
# -----------------------------------------------------------------------------
TONE_WHITELIST: set[str] = {
    "whimsical",
    "calm",
    "funny",
    "silly",
    "gentle",
    "heartfelt",
    "mysterious",
    "spooky",
    "sad",
    "uplifting",
    "adventurous",
    "exciting",
    "cozy",
    "poetic",
    "suspenseful",
    "inspiring",
    "quiet",
    "witty",
    "sweet",
    "goofy",
    "dramatic",
}

# Normalize common variants to canonical theme tokens
THEME_NORMALIZATION: dict[str, str] = {
    "magical": "magic",
    "wizard": "magic",
    "wizards": "magic",
    "wizarding": "magic",
    "spell": "magic",
    "spells": "magic",
    "friendships": "friendship",
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _norm(text: str) -> str:
    """Lowercase + collapse whitespace."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _user_supplied_age(text: str) -> str:
    """Return an explicit age/range (e.g., "5" or "6-8") if present, else ''."""
    return detect_age_from_text(text)


def _tokenize(text: str) -> List[str]:
    """Tokenize to alphabetic words, drop stopwords and very short tokens."""
    words = re.findall(r"[a-z]+", _norm(text))
    return [w for w in words if w not in STOPWORDS and len(w) >= 3]


def _normalize_theme_token(tok: str) -> str:
    return THEME_NORMALIZATION.get(tok, tok)


def _dedup_keep_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _deterministic_parse(user_input: str) -> Dict[str, Any]:
    """Strict, model‑free parse.

    Returns
    -------
    dict with keys: age_range, tone, themes
    """
    text = user_input or ""
    age = _user_supplied_age(text)  # '' if not explicit
    toks = _tokenize(text)

    tone = next((t for t in toks if t in TONE_WHITELIST), "")

    # themes = all tokens except the chosen tone, normalized
    theme_candidates = [t for t in toks if t != tone]
    themes = _dedup_keep_order([_normalize_theme_token(t) for t in theme_candidates])
    # prefer at most 4 themes
    themes = themes[:4]

    return {"age_range": age or "", "tone": tone or "", "themes": themes}


def _merge_user_first(user_themes: List[str], ai_themes: List[str], cap: int = 4) -> List[str]:
    """Merge themes, preserving user tokens first; normalize, dedup, cap length."""
    user_norm = [_normalize_theme_token(_norm(x)) for x in user_themes if _norm(x)]
    ai_norm = [_normalize_theme_token(_norm(x)) for x in ai_themes if _norm(x)]
    merged = _dedup_keep_order(user_norm + ai_norm)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def parse_user_query(user_input: str) -> Dict[str, Any]:
    """Hybrid parsing with conservative LLM fallback.

    Strategy
    --------
    1) Deterministic rules (no guesses). If tone or themes found, return.
    2) Otherwise, call the LLM to propose fields; sanitize and merge.

    Contract
    --------
    - "age_range" is *never* guessed; it's only present if typed explicitly.
    - "tone" is one word from a whitelist; unrecognized tones are demoted
      to themes.
    - "themes" are literal, single‑word tokens (normalized variants allowed),
      1–4 items total.
    """
    # Pass 1: deterministic
    det = _deterministic_parse(user_input)
    if det["tone"] or det["themes"]:
        return det  

    # Pass 2: model fallback 
    prompt = textwrap.dedent(f"""\
        You extract simple search filters from a parent's request for children's book recs.

        Rules:
        - Output ONLY a JSON object with keys: "age_range", "tone", "themes".
        - "age_range": string like "3-5" ONLY if the USER EXPLICITLY typed an age or range. Otherwise return "".
        - "tone": ONE word (e.g., "whimsical", "calm"), or "" if unclear.
        - "themes": 1-4 SINGLE-WORD literal keywords from the user's text (do NOT replace short queries with synonyms).
        - Do NOT invent or guess "age_range". If it's not explicit, set it to "".

        User input:
        \"\"\"{user_input}\"\"\"
        JSON:
    """)

    ai: Dict[str, Any] = {"age_range": "", "tone": "", "themes": []}

    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.0,
            max_tokens=120,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = (resp.choices[0].message.content or "").strip()
        # tolerate ```json ... ``` or stray text by extracting the first {...}
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if m:
            raw = m.group(0)
        ai = json.loads(raw)
    except Exception:
        # API error or bad JSON => return deterministic result (empty tone/themes)
        return det

    # --- Sanitize model output ---
    age = _user_supplied_age(user_input) or ""

    # Normalize AI outputs
    ai_tone = _norm(ai.get("tone", ""))
    ai_themes = ai.get("themes", [])
    if not isinstance(ai_themes, list):
        ai_themes = []

    # If model gave a non-whitelisted "tone", convert it to a theme
    # and clear tone (may later promote to tone if appropriate).
    if ai_tone and ai_tone not in TONE_WHITELIST:
        converted = _normalize_theme_token(ai_tone)  # e.g., "magical" -> "magic"
        if converted:
            ai_themes = [converted] + ai_themes
        tone = ""
    else:
        tone = ai_tone

    # Merge: user tokens first, then model suggestions
    merged_themes = _merge_user_first(det["themes"], ai_themes, cap=4)

    return {"age_range": age, "tone": tone, "themes": merged_themes}


__all__ = ["parse_user_query"]


