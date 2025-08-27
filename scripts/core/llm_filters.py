from __future__ import annotations
import re, json
from typing import List
import streamlit as st

from scripts.core.config import get_openai_client

# --- tiny tag normalizer ---
def _split_tags_ui(s: str) -> List[str]:
    s = str(s or "")
    if s.strip().startswith("[") and s.strip().endswith("]"):
        try:
            arr = json.loads(s)
            return [str(x).strip().lower() for x in arr if isinstance(x, str)]
        except Exception:
            pass
    return [t.strip().lower() for t in re.split(r"[;,/|]+", s) if t.strip()]

# cache scores across reruns: key is (tone_query, tuple(sorted(tones_row)))
@st.cache_data(show_spinner=False)
def _cached_score(tone_query: str, tones_row_key: str) -> int:
    """Internal cache wrapper; the actual scoring happens in _score_once."""
    return _score_once(tone_query, tones_row_key.split("|"))

def _score_once(tone_query: str, tones_row: list[str]) -> int:
    """
    Ask an LLM to rate compatibility:
      4=perfect, 3=good, 2=okay, 1=poor, 0=opposite.
    Return an int in [0..4]. On any error, return 0.
    """
    client = get_openai_client()
    if not client:
        # No key or client -> neutral/bad score, but keeps app running
        return 0

    prompt = (
        "Rate how compatible a book's tones are with the requested tone.\n"
        "Return a single integer 0–4 only. 4=perfect, 3=good, 2=okay, 1=poor, 0=opposite/clashing.\n\n"
        f"Requested tone: {tone_query}\n"
        f"Book tones: {', '.join(tones_row) if tones_row else '(none)'}\n"
        "Answer with just the number."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0, max_tokens=5,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\b([0-4])\b", raw)
        return int(m.group(1)) if m else 0
    except Exception:
        return 0

def score_rows_by_tone(df, tone_query: str, tone_col: str = "tone"):
    """
    Returns a copy of df with:
      - 'tones_norm_list' : normalized list of tone tags
      - '__tone_score'    : 0..4 compatibility score from the LLM
    Only the rows present are scored (so pass a small candidate subset).
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    out["tones_norm_list"] = out.get(tone_col, "").apply(_split_tags_ui)

    # dedupe identical tone-sets to avoid repeated LLM calls
    keys = []
    for lst in out["tones_norm_list"]:
        key = "|".join(sorted(lst or []))
        keys.append(key)
    out["__tones_key"] = keys

    # score each unique key once (cached across reruns)
    unique_keys = sorted(set(keys))
    scores_map = {k: _cached_score(tone_query, k) for k in unique_keys}

    out["__tone_score"] = out["__tones_key"].map(scores_map).fillna(0).astype(int)
    return out.drop(columns=["__tones_key"])
