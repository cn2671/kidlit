"""Text utilities for KidLit.

Functions here support tokenization, theme cleaning, and age extraction from
free‑text queries. This pass focuses on clarity and consistent behavior.
"""
from __future__ import annotations
import re
from typing import Iterable, List


# -----------------------------------------------------------------------------
# Core normalization
# -----------------------------------------------------------------------------

def norm(s: str) -> str:
    """Lowercase, collapse internal whitespace, and strip."""
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


# -----------------------------------------------------------------------------
# Stopwords / title stopwords
# -----------------------------------------------------------------------------
BASE_STOPWORDS = {
    "book","books","about","for","a","an","the","and","or","with","on","like",
    "kids","kid","child","children","of","to","please","show","find",
    "i","im","i'm","want","looking","that","this","is","are","need","some","something","who",
    "recommend","recommendation","recommendations","teaches",
    "age","ages","year","years","yr","yrs","yo","old"
}

# keep the alias for clarity 
TITLE_STOPWORDS = BASE_STOPWORDS


# -----------------------------------------------------------------------------
# Tokenization / theme cleaning
# -----------------------------------------------------------------------------

def tokenize_alpha(q: str) -> List[str]:
    """Return alphabetic tokens (>=3 chars), pre‑normalized."""
    toks = re.findall(r"[a-z]+", norm(q))
    return [t for t in toks if len(t) >= 3]


def clean_themes(lst: Iterable[str]) -> List[str]:
    """Normalize, de‑duplicate, and filter out trivial theme tokens."""
    out: List[str] = []
    seen = set()
    for x in (lst or []):
        w = norm(x)
        if not w or len(w) < 3 or w in BASE_STOPWORDS:
            continue
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


# -----------------------------------------------------------------------------
# User-facing age categories (phrases in queries) -> numeric ranges
# -----------------------------------------------------------------------------
AGE_CATEGORY_LABELS = {
    ("young adult", "young adults", "ya", "teen", "teens"): (12, 18),
    ("middle grade", "middle-grade", "mg"): (8, 12),
    (
        "early reader",
        "early readers",
        "beginner reader",
        "beginning reader",
    ): (6, 8),
    (
        "chapter book",
        "chapter books",
        "early chapter book",
        "early chapter books",
    ): (6, 9),
    ("kindergarten", "kinder"): (5, 6),
    ("preschool", "pre-school", "pre k", "pre-k", "prek"): (3, 5),
    ("toddler", "toddlers"): (1, 3),
    ("baby", "infant", "board book", "board books"): (0, 2),
}


# -----------------------------------------------------------------------------
# Age detection helpers
# -----------------------------------------------------------------------------

def detect_age_from_text(text: str) -> str:
    """Return "A-B" (range) or "N" (single) if the user typed an age/range or a
    category phrase; return "" if nothing explicit was found.

    Examples accepted:
    - "6-8", "6–8", "6—8"
    - "age 5" / "ages 5"
    - "5 year old", "5yo", "5 yr", "5-year-old", "5yearold"
    - category phrases like "middle grade" → "8-12"
    """
    t = norm(text)

    # 1) Numeric range: 6-8 
    m = re.search(r"\b(\d{1,2})\s*[–—-]\s*(\d{1,2})\b", t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a <= b:
            return f"{a}-{b}"

    # 2) "age 5" / "ages 5"
    m = re.search(r"\bages?\s*(\d{1,2})\b", t)
    if m:
        return m.group(1)

    # 3) "5 year(s) old" / "5yo" / "5 yr" / "5-year-old" / "5yearold"
    m = re.search(r"\b(\d{1,2})\s*(?:-?\s*year(?:s)?\s*old|-?\s*(?:yo|yr|yrs))\b", t)
    if m:
        return m.group(1)
    m = re.search(r"\b(\d{1,2})yearold\b", t)
    if m:
        return m.group(1)

    # 4) Category phrases => numeric span
    for labels, (lo, hi) in AGE_CATEGORY_LABELS.items():
        for label in labels:
            if re.search(rf"\b{re.escape(label)}\b", t):
                return f"{lo}-{hi}"

    return ""


def strip_age_category_tokens(themes: Iterable[str], user_text: str) -> List[str]:
    """If a category phrase was present in the query, remove its words from the
    provided theme list (e.g., remove "young" and "adult" when "young adult"
    is detected)."""
    t = norm(user_text)
    words_to_remove = set()
    for labels in AGE_CATEGORY_LABELS.keys():
        for label in labels:
            if re.search(rf"\b{re.escape(label)}\b", t):
                for w in label.split():
                    words_to_remove.add(norm(w))
    return [th for th in (themes or []) if norm(th) not in words_to_remove]


__all__ = [
    "norm",
    "BASE_STOPWORDS",
    "TITLE_STOPWORDS",
    "tokenize_alpha",
    "clean_themes",
    "AGE_CATEGORY_LABELS",
    "detect_age_from_text",
    "strip_age_category_tokens",
]