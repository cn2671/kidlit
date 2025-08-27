"""Recommendation engine for KidLit.

This module filters a catalog of children's books by age, themes, and tone.
It includes small, data-driven helpers for theme/tone normalization and an
optional embedding-backed expansion for themes (cached locally).

"""
# IMPORTS
from __future__ import annotations
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
import pandas as pd


# ==============================================================================
# Load & normalize base dataframe
# ==============================================================================
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

CSV_PATH = DATA_DIR / "books_llm_tags.csv"
DF = pd.read_csv(CSV_PATH).fillna("")

for col in [
    "title",
    "author",
    "themes",
    "tone",
    "age_range",
    "cover_url",
    "summary_gpt",
]:
    if col not in DF.columns:
        DF[col] = ""

DF["age_range"] = (
    DF["age_range"].astype(str).str.replace("–", "-", regex=False).str.replace("—", "-", regex=False)
)
DF["themes_norm"] = DF["themes"].apply(_split_tags) if "themes" in DF.columns else [[] for _ in range(len(DF))]
DF["tones_norm"]  = DF["tone"].apply(_split_tags)   if "tone"   in DF.columns else [[] for _ in range(len(DF))]


# ==============================================================================
# String / tag helpers
# ==============================================================================

def _norm_text(s: str) -> str:
    """Lowercase, NFKD normalize, collapse whitespace, strip."""
    s = unicodedata.normalize("NFKD", str(s or ""))
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _split_tags(s: str) -> List[str]:
    """Split a tag field into a normalized, de-duplicated list of strings."""
    s = str(s or "")

    # If tags look like a JSON array, parse them
    if s.strip().startswith("[") and s.strip().endswith("]"):
        try:
            arr = json.loads(s)
            parts = [str(x) for x in arr if isinstance(x, str)]
        except Exception:
            parts = [s]
    else:
        # split on commas / semicolons / pipes / slashes
        parts = re.split(r"[;,/|]+", s)

    seen, out = set(), []
    for p in parts:
        n = _norm_text(p)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


TONE_SYNONYMS: dict[str, list[str]] = {
    "fun": ["playful", "light-hearted", "whimsical", "funny"],
    "adventure": ["adventurous", "exciting"],
}

THEME_SYNONYMS: dict[str, list[str]] = {
    "magic": ["magical", "wizard", "wizards", "wizarding", "spell", "spells"],
    "friendship": ["friends", "friendships"],
}


def _choices_for_tone(tone: str) -> List[str]:
    t = _norm_text(tone)
    choices: List[str] = []
    seen: set[str] = set()

    def add(x: str) -> None:
        x = _norm_text(x)
        if x and x not in seen:
            seen.add(x)
            choices.append(x)

    # 1) the input itself
    add(t)

    # 2) forward expansion: synonyms of the input key
    for x in TONE_SYNONYMS.get(t, []):
        add(x)

    # 3) reverse expansion: if the input appears in any synonym list,
    #    include that base key and all of its synonyms
    for base, syns in TONE_SYNONYMS.items():
        norm_syns = [_norm_text(s) for s in syns]
        if t == _norm_text(base) or t in norm_syns:
            add(base)
            for s in norm_syns:
                add(s)

    return choices

# ==============================================================================
# Age parsing helper
# ==============================================================================

def parse_age_span(s: str) -> Optional[Tuple[int, int]]:
    """Return (lo, hi) numeric age bounds if we can parse them; else None."""
    if not s:
        return None
    t = str(s).lower().strip().replace("–", "-").replace("—", "-")

    # 1) "4-7"
    m = re.search(r"\b(\d{1,2})\s*-\s*(\d{1,2})\b", t)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return (lo, hi) if lo <= hi else None

    # 2) "4 to 7" / "ages 4 to 7"
    m = re.search(r"\b(?:ages?\s*)?(\d{1,2})\s*(?:to|-|—|–)\s*(\d{1,2})\b", t)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return (lo, hi) if lo <= hi else None

    # 3) "5+"
    m = re.search(r"\b(\d{1,2})\s*\+\b", t)
    if m:
        return (int(m.group(1)), 99)

    # 4) lone "5"
    m = re.search(r"\b(\d{1,2})\b", t)
    if m:
        x = int(m.group(1))
        return (x, x)

    # 5) labels like "middle grade"
    for labels, span in {
        ("baby", "infant", "board book", "board books"): (0, 2),
        ("toddler", "toddlers"): (1, 3),
        ("preschool", "pre-school", "pre k", "pre-k", "prek"): (3, 5),
        ("kindergarten", "kinder"): (5, 6),
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
        ("middle grade", "middle-grade", "mg"): (8, 12),
        ("young adult", "ya", "teen", "teens"): (12, 18),
    }.items():
        for label in labels:
            if re.search(rf"\b{re.escape(label)}\b", t):
                return span

    return None

_parse_age_span = parse_age_span


# ==============================================================================
# Recommender
# ==============================================================================

def recommend_books(
    age_range: Optional[str] = None,
    themes: Optional[Iterable[str]] = None,
    tone: Optional[str] = None,
    n: Optional[int] = None,
    source_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Return up to *n* books that match the given criteria.

    Parameters
    ----------
    age_range : str | None
        Examples: "3–5", "6-8", "5", "5+".
    themes : Iterable[str] | None
        e.g., ["curiosity", "bedtime"].
    tone : str | None
        e.g., "calm", "funny", "whimsical".
    n : int | None
        Limit the number of rows returned.
    source_df : DataFrame | None
        Optional DataFrame to filter (defaults to module-level DF).
    """
    base = DF if source_df is None else source_df
    work = base.copy()

    # -------- Age filter (numeric-aware) --------
    if age_range:
        age_filter = str(age_range).strip().replace("–", "-").replace("—", "-")

        # Single age like "5"
        if re.fullmatch(r"\d{1,2}", age_filter):
            target = int(age_filter)
            work = work[
                work["age_range"].apply(
                    lambda v: (lambda sp: (sp is not None) and (sp[0] <= target <= sp[1]))(parse_age_span(v))
                )
            ]
        else:
            # Explicit range like "4-6"
            m = re.fullmatch(r"(\d{1,2})-(\d{1,2})", age_filter)
            if m:
                want_lo, want_hi = int(m.group(1)), int(m.group(2))
                work = work[
                    work["age_range"].apply(
                        lambda v: (lambda sp: (sp is not None) and not (sp[1] < want_lo or sp[0] > want_hi))(
                            parse_age_span(v)
                        )
                    )
                ]
            else:
                # last-resort: substring for unusual inputs
                work = work[work["age_range"].str.contains(re.escape(age_filter), na=False, case=False)]
    


    # -------- Theme OR-match (no embeddings) --------
    if themes:
        # normalize inputs
        wanted = {_norm_text(t) for t in themes if str(t).strip()}
        # optional: light manual expansion if you kept THEME_SYNONYMS
        try:
            wanted |= _expand_themes(wanted)
        except NameError:
            pass  # if you removed _expand_themes just ignore

        if wanted:
            work = work[work["themes_norm"].apply(lambda lst: any(w in (lst or []) for w in wanted))]

    # -------- Tone exact/synonym match (no embeddings) --------
    if tone:
        try:
            choices = _choices_for_tone(tone)  # e.g., ["calm","gentle","quiet",...]
        except NameError:
            # fallback: exact match only
            choices = [_norm_text(tone)]

        narrowed = work[work["tones_norm"].apply(lambda lst: any(c in (lst or []) for c in choices))]
        if not narrowed.empty:
            work = narrowed  # only narrow if we actually found tone hits


    # # -------- Theme OR-match against normalized array --------
    # if themes:
    #     wanted = [_norm_text(t) for t in themes if t]
    #     unwanted = {
    #         "book",
    #         "books",
    #         "about",
    #         "for",
    #         "the",
    #         "and",
    #         "or",
    #         "with",
    #         "on",
    #         "like",
    #         "kids",
    #         "kid",
    #         "child",
    #         "children",
    #         "of",
    #         "to",
    #         "please",
    #         "show",
    #         "find",
    #         "i",
    #         "im",
    #         "i'm",
    #         "want",
    #         "looking",
    #         "that",
    #         "this",
    #         "is",
    #         "are",
    #         "need",
    #         "some",
    #         "something",
    #         "recommend",
    #         "recommendation",
    #         "recommendations",
    #         "year",
    #         "years",
    #         "yr",
    #         "yo",
    #         "old",
    #     }
    #     wanted = [w for w in wanted if len(w) >= 3 and w not in unwanted]

    #     # Expand by chosen method; fall back to raw tokens if helper missing
    #     try:
    #         ws = _expand_themes(wanted)  # manual synonyms
    #     except NameError:
    #         ws = set(wanted)

    #     if ws:
    #         work = work[work["themes_norm"].apply(lambda lst: any(w in (lst or []) for w in ws))]

    # # -------- Tone exact/synonym match --------
    # if tone:
    #     choices = _choices_for_tone(tone)
    #     narrowed = work[work["tones_norm"].apply(lambda lst: any(c in (lst or []) for c in choices))]
    #     if not narrowed.empty:
    #         work = narrowed  # only narrow if actually found tone hits

    # -------- Empty -> return empty with schema --------
    if work.empty:
        cols_all = [
            "title",
            "author",
            "cover_url",
            "summary_gpt",
            "themes",
            "tone",
            "age_range",
            "ol_title",
            "ol_author",
            "goodreads_url",
            "openlibrary_url",
            "description",
            "summary",
        ]
        return pd.DataFrame(columns=[c for c in cols_all if c in base.columns])

    # -------- Sorting --------
    sort_cols = [c for c in ["rating", "ratings_count", "popularity"] if c in work.columns]
    if sort_cols:
        work = work.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))

    # -------- Select output columns --------
    cols = [c for c in [
        "title", "author", "cover_url", "summary_gpt", "themes", "tone", "age_range",
        "goodreads_url", "openlibrary_url", "description", "summary",
        "ol_title", "ol_author"
        ] if c in work.columns]
    out = work[cols].copy().fillna("")

    # -------- Limit if n given --------
    return out.head(n) if isinstance(n, int) and n > 0 else out


__all__ = [
    "parse_age_span",
    "recommend_books",
]
