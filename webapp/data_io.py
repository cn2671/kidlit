from __future__ import annotations
import re
import pandas as pd
from typing import Any, Dict, List
from pathlib import Path
import streamlit as st
from scripts.core.text_utils import split_tags

# ==============================================================================
# Utilities
# ==============================================================================

def _norm(s: str) -> str:
    """Lowercase, condense whitespace, and strip.

    Args:
        s: Input string (may be None/falsey).
    Returns:
        Normalized string.
    """
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def _norm_title(s: str) -> str:
    """Normalize titles by stripping parenthetical volume markers and dashes.

    Also lowercases and condenses whitespace.
    """
    s = re.sub(r"\([^)]*\)", "", str(s or ""))
    s = s.replace("–", "-").replace("—", "-")
    return _norm(s)

def as_dict(row: Any) -> Dict[str, Any]:
    """Best‑effort conversion to a plain dict.

    Handles dicts, pandas Series/Rows, and general mappings.
    """
    if isinstance(row, dict):
        return row
    try:
        return row.to_dict()
    except Exception:
        return dict(row)


def _book_key(book: Dict[str, Any] | Any) -> str:
    """Build a stable key from title+author for de‑duping.

    Tolerates dict‑like objects and falls back to a
    secondary string if both title/author are missing.
    """
    try:
        t = book.get("title") or book.get("ol_title") or ""
        a = book.get("author") or book.get("ol_author") or ""
    except AttributeError:
        b = as_dict(book)
        t = b.get("title") or b.get("ol_title") or ""
        a = b.get("author") or b.get("ol_author") or ""

    k = f"{_norm(t)}|{_norm(a)}"

    # If both title/author are blank, add a tie‑breaker so keys don’t collide.
    if k == "|":
        extra_src = b if "b" in locals() else book
        extra = _norm(
            extra_src.get("openlibrary_url")
            or extra_src.get("goodreads_url")
            or extra_src.get("title")
            or ""
        )
        if extra:
            k = f"{k}|{extra}"
    return k


def dedupe_books(lst):
    """Remove duplicate items based on the stable book key, preserving order."""
    seen, out = set(), []
    for b in lst:
        k = _book_key(b if isinstance(b, dict) else dict(b))
        if k not in seen:
            seen.add(k)
            out.append(b)
    return out

@st.cache_data
def load_catalog(path: str = "books_llm_tags.csv") -> pd.DataFrame:
    """Load the book catalog and compute normalized helper columns.

    The path is aligned with the recommender module.
    """
    df = pd.read_csv(path).fillna("")
    for col in ["title", "ol_title", "author", "ol_author", "themes", "tone"]:
        if col not in df.columns:
            df[col] = ""
    df["title_norm"]   = df["title"].apply(_norm_title)
    df["ol_title_norm"]= df["ol_title"].apply(_norm_title)
    df["author_norm"]  = df["author"].apply(_norm)
    df["ol_author_norm"]=df["ol_author"].apply(_norm)

    # Normalized tag arrays (use the shared function)
    df["themes_norm_list"] = df["themes"].apply(split_tags)
    df["tones_norm_list"]  = df["tone"].apply(split_tags)

    # Keep legacy names some code expects
    df["themes_norm"] = df["themes_norm_list"]
    df["tones_norm"]  = df["tones_norm_list"]
    return df

def _safe_key_fragment(s: str) -> str:
    """Return a filesystem/HTML‑id friendly fragment."""
    import re
    return re.sub(r"[^a-z0-9_-]+", "_", _norm(s))[:60]


def build_index(df: pd.DataFrame) -> dict[str, dict]:
    """Build a key→row mapping for quick rehydration joins."""
    idx = {}
    for _, r in df.iterrows():
        d = r.to_dict()
        k = _book_key(d)
        if k:
            idx[k] = d
    return idx


def rehydrate_book(b: Any, index: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Fill missing book fields from the catalog index when available."""
    d = as_dict(b)
    k = _book_key(d)
    src = index.get(k)
    if src:
        for col in [
            "author",
            "ol_author",
            "summary_gpt",
            "themes",
            "tone",
            "age_range",
            "cover_url",
            "title",
            "ol_title",
            "goodreads_url",
            "openlibrary_url",
            "description",
        ]:
            if not d.get(col):
                d[col] = src.get(col, d.get(col))
    return d
