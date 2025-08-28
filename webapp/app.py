"""KidLit — App for personalized children's book recommendations.
"""

# IMPORTS
from __future__ import annotations
import pandas as pd
import streamlit as st
import math
import os
import re
import sys, pathlib
from collections import Counter

# --- Project imports / path setup -------------------------------------------------
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]  
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.insert(0, parent_dir)

from scripts.core.parse_query import parse_user_query
from scripts.core.recommender import parse_age_span, recommend_books
from scripts.core.text_utils import (
    TITLE_STOPWORDS,
    clean_themes as _clean_themes,
    detect_age_from_text,
    strip_age_category_tokens,
    tokenize_alpha,
)

from webapp.ui.css import inject_global_css
from webapp.ui.grid import render_book_grid
from webapp.data_io import load_catalog, build_index, rehydrate_book, dedupe_books
from webapp.data_io import _norm, _norm_title, _book_key, as_dict


# ==============================================================================
# Utilities
# ==============================================================================

def _as_list(x):
    """Coerce a value to a clean list of strings."""
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return [str(t).strip() for t in x if str(t).strip()]
    if isinstance(x, str):
        parts = re.split(r"[;,/|]+", x)
        return [p.strip() for p in parts if p.strip()]
    try:
        return [str(t).strip() for t in list(x) if str(t).strip()]
    except Exception:
        s = str(x).strip()
        return [s] if s else []


def _reset_pagination() -> None:
    st.session_state.page_num_recs = 1


def _goto_recs():
    st.session_state._nav_target = "🔎 Recommendations"
    st.rerun()

def _set_query_and_page(q):
    st.session_state.user_query = q
    st.session_state.do_search = True 
    st.session_state.menu_radio = "🔎 Recommendations"    

# --- Theme/Tone heuristics -------------------------------------------------------

def should_treat_as_theme(word: str) -> bool:
    w = _norm(word)
    if not w or CATALOG_DF.empty:
        return False
    if "themes_norm_list" not in CATALOG_DF.columns or "tones_norm_list" not in CATALOG_DF.columns:
        return False
    theme_count = int((CATALOG_DF["themes_norm_list"].apply(lambda lst: w in lst)).sum())
    tone_count = int((CATALOG_DF["tones_norm_list"].apply(lambda lst: w in lst)).sum())
    return theme_count >= max(2, tone_count * 2)


def should_treat_as_tone(word: str) -> bool:
    w = _norm(word)
    if not w or CATALOG_DF.empty:
        return False
    if "themes_norm_list" not in CATALOG_DF.columns or "tones_norm_list" not in CATALOG_DF.columns:
        return False
    theme_count = int((CATALOG_DF["themes_norm_list"].apply(lambda lst: w in lst)).sum())
    tone_count = int((CATALOG_DF["tones_norm_list"].apply(lambda lst: w in lst)).sum())
    return tone_count >= max(1, theme_count * 2)


def sanitize_parsed(user_text, parsed):
    """Post‑process parsed query fields (themes/tone/age) for consistency."""
    out = dict(parsed or {})
    themes = _as_list(out.get("themes", []))
    tone = (out.get("tone") or "").strip()

    # Keep canonical tones; otherwise demote/upgrade between tone/theme.
    if tone and not is_canonical_tone(tone):
        if should_treat_as_theme(tone):
            if _norm(tone) not in [_norm(x) for x in themes]:
                themes.append(tone)
            out["tone"] = ""
            tone = ""

    if not tone:
        for w in list(themes):
            if is_canonical_tone(w) or should_treat_as_tone(w):
                out["tone"] = w
                themes = [t for t in themes if _norm(t) != _norm(w)]
                break

    # AGE: detect from user text (numeric or category)
    age = detect_age_from_text(user_text)
    out["age_range"] = age

    # If an age category phrase was used, strip its words from themes
    themes = strip_age_category_tokens(themes, user_text)

    # Remove explicit age words and bare numbers from themes
    AGE_WORDS = {"age","ages","year", "years", "yr", "yrs", "yo", "old"}
    themes = [
        t
        for t in themes
        if _norm(t) not in AGE_WORDS and not re.fullmatch(r"\d+", _norm(t))
    ]

    out["themes"] = _clean_themes(themes)
    return out


# Canonical tone set
TONE_CANON = {
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


def is_canonical_tone(word: str) -> bool:
    return _norm(word) in TONE_CANON


def _age_sort_key(token: str) -> tuple[int, int]:
    """
    Normalize an age token (e.g., '3–5', '6-8', '5', 'young adult', 'middle grade')
    into a (lo, hi) tuple so we can sort consistently.
    Unknowns sort to the end.
    """
    t = (token or "").strip().lower().replace("–", "-").replace("—", "-")

    # Map common labels to numeric spans
    LABEL_SPANS = {
        "baby": (0, 2), "infant": (0, 2),
        "toddler": (1, 3), "toddlers": (1, 3),
        "preschool": (3, 5), "pre-school": (3, 5), "pre k": (3, 5), "pre-k": (3, 5), "prek": (3, 5),
        "kindergarten": (5, 6), "kinder": (5, 6),
        "early reader": (6, 8), "early readers": (6, 8), "beginner reader": (6, 8), "beginning reader": (6, 8),
        "chapter book": (6, 9), "chapter books": (6, 9),
        "middle grade": (8, 12), "middle-grade": (8, 12), "mg": (8, 12),
        "young adult": (12, 18), "ya": (12, 18), "teen": (12, 18), "teens": (12, 18),
    }
    if t in LABEL_SPANS:
        return LABEL_SPANS[t]

    # Numeric range like '3-5'
    m = re.fullmatch(r"(\d{1,2})-(\d{1,2})", t)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return (lo, hi) if lo <= hi else (hi, lo)

    # Single age like '5'
    m = re.fullmatch(r"\d{1,2}", t)
    if m:
        x = int(m.group(0))
        return (x, x)

    # Push unknowns to the end
    return (999, 999)

def _tones_list(s: str) -> list[str]:
    s = str(s or "")
    if s.strip().startswith("[") and s.strip().endswith("]"):
        try:
            arr = json.loads(s)
            return [str(x).strip().lower() for x in arr if isinstance(x, str)]
        except Exception:
            pass
    # comma/semicolon/pipe/slash fallback
    return [t.strip().lower() for t in re.split(r"[;,/|]+", s) if t.strip()]

# ==============================================================================
# Page Config & Session State
# ==============================================================================

st.set_page_config(page_title="KidLit", page_icon="📚", layout="wide")
inject_global_css()

for key in ["user_query", "generated", "liked_books", "skipped_books", "read_books"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key.endswith("_books") else (False if key == "generated" else "")


if "menu_radio" not in st.session_state:
    st.session_state.menu_radio = "🏠 Home"
if "recs_df" not in st.session_state:
    st.session_state.recs_df = None
if "last_query_str" not in st.session_state:
    st.session_state.last_query_str = ""
if "expanded_cards" not in st.session_state:
    st.session_state.expanded_cards = set()
if "page_size_recs" not in st.session_state:
    st.session_state.page_size_recs = 9
if "do_search" not in st.session_state:
    st.session_state.do_search = False
if "user_query" not in st.session_state:
    st.session_state.user_query = ""
if "menu_radio" not in st.session_state:
    st.session_state.menu_radio = "🏠 Home"
if st.session_state.get("do_search") and st.session_state.get("menu_radio") != "🔎 Recommendations":
    st.session_state.menu_radio = "🔎 Recommendations"

# Apply nav intent before the widget is instantiated
nav_target = st.session_state.pop("_nav_target", None)
if nav_target:
    st.session_state.menu_radio = nav_target


# Normalize legacy skipped entries (strings → dicts)
st.session_state.skipped_books = [
    {"title": b} if isinstance(b, str) else as_dict(b) for b in st.session_state.skipped_books
]

# Load catalog & rehydrate lists for full card data
CATALOG_DF = load_catalog("data/books_llm_tags.csv")
CATALOG_IDX = build_index(CATALOG_DF)

st.session_state.liked_books = [rehydrate_book(b, CATALOG_IDX) for b in st.session_state.liked_books]
st.session_state.read_books = [rehydrate_book(b, CATALOG_IDX) for b in st.session_state.read_books]
st.session_state.skipped_books = [rehydrate_book(b, CATALOG_IDX) for b in st.session_state.skipped_books]

# Deduplicate after rehydration
st.session_state.liked_books = dedupe_books(st.session_state.liked_books)
st.session_state.read_books = dedupe_books(st.session_state.read_books)
st.session_state.skipped_books = dedupe_books(st.session_state.skipped_books)



# ==============================================================================
# UI
# ==============================================================================

# Sidebar branding
st.sidebar.markdown(
    '<div class="k-side-hero"><h3>Menu</h3><p>Browse your favorites and fresh picks</p></div>',
    unsafe_allow_html=True,
)


MENU = [
    ("🏠 Home",            "Home"),
    ("🔎 Recommendations", "Recommendations"),
    ("❤️ Favorites", "Favorites"),
    ("🚫 Skipped", "Skipped"),
    ("📖 Read", "Read"),
]
labels = [lbl for lbl, _ in MENU]
values = {lbl: val for lbl, val in MENU}
page_label = st.sidebar.radio(
    "Menu",
    labels,
    index=labels.index(st.session_state.get("menu_radio", "🏠 Home")),
    key="menu_radio",
    label_visibility="collapsed",
)
page = values[page_label]

# Title
st.markdown(
    '<div class="k-hero">'
    '<h1 class="kidlit-logo">KidLit</h1>'
    '<p class="kidlit-sub">Kid’s Literature: Personalized children’s books by age, themes, and tone.</p>'
    '</div>',
    unsafe_allow_html=True,
)

if page == "Home":
    st.title("👋 Welcome to KidLit")
    st.markdown("""
**Find the perfect children’s book** by age, theme, and tone.
Try a few examples:
- “book about **friendship** for a **5 year old**”
- “**whimsical** bedtime story for **3–5**”
- “**adventurous** chapter book about **magic**”
    """)

    # Big CTA to jump to Recommendations
    st.button("Start finding books →", type="primary", on_click=_goto_recs)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("5-year-old • friendship",
                  on_click=_set_query_and_page,
                  args=("book about friendship for a 5 year old",))
    with col2:
        st.button("3–5 • whimsical • bedtime",
                  on_click=_set_query_and_page,
                  args=("whimsical bedtime story for ages 3–5",))
    with col3:
        st.button("magic • adventurous • 7",
                  on_click=_set_query_and_page,
                  args=("adventurous book about magic for a 7 year old",))

    
    st.markdown("---")
    st.subheader("Browse by Theme, Age, and Tone")
    st.caption("These are just examples — you can search with any themes, tones, or ages.")

    # --- Build frequency tables from catalog  ---
    theme_counts = Counter()
    for lst in CATALOG_DF.get("themes_norm_list", []):
        for t in (lst or []):
            theme_counts[_norm(t)] += 1

    tone_counts = Counter()
    for lst in CATALOG_DF.get("tones_norm_list", []):
        for t in (lst or []):
            tone_counts[_norm(t)] += 1
    
    # --- Age counts (normalize 3–5 / 3-5 / 5 -> "3-5" or "5") ---
    def _norm_age_str(s: str) -> str:
        s = str(s or "").strip().replace("–", "-").replace("—", "-")
        m = re.match(r"^\s*(\d{1,2})(?:\s*-\s*(\d{1,2}))?\s*$", s)
        if not m:
            return ""
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) else lo
        if lo > hi:
            lo, hi = hi, lo
        return f"{lo}-{hi}" if lo != hi else f"{lo}"

    age_counts = Counter()
    for s in CATALOG_DF.get("age_range", []):
        a = _norm_age_str(s)
        if a:
            age_counts[a] += 1

    # ------- 1) Popular chips (Top N) -------
    TOP_THEMES = [t for t, _ in theme_counts.most_common(12)]
    TOP_TONES  = [t for t, _ in tone_counts.most_common(12)]
    TOP_AGES   = [a for a, _ in age_counts.most_common(9)]


    def chip(label: str, query: str, key: str):
        st.button(label, key=key, on_click=_set_query_and_page, args=(query,), use_container_width=True)

    st.markdown("#### Popular Themes")
    for i in range(0, len(TOP_THEMES), 6):
        cols = st.columns(6)
        for j, t in enumerate(TOP_THEMES[i:i+6]):
            with cols[j]:
                chip(t.title(), f"book about {t}", key=f"pop_theme_{i}_{j}")

    st.markdown("#### Popular Tones")
    for i in range(0, len(TOP_TONES), 6):
        cols = st.columns(6)
        for j, t in enumerate(TOP_TONES[i:i+6]):
            with cols[j]:
                chip(t.title(), f"{t} children's book", key=f"pop_tone_{i}_{j}")
    

    st.markdown("#### Age")

    ages = [
        ("0–2 (baby)",        "books for ages 0-2"),
        ("1–3 (toddler)",     "books for ages 1-3"),
        ("3–5 (preschool)",   "books for ages 3-5"),
        ("5–6 (kindergarten)","books for ages 5-6"),
        ("6–9 (early reader)","books for ages 6-9"),
        ("8–12 (middle grade)","books for ages 8-12"),
        ("12–18 (young adult)","books for young adults"),
        ]

    for i in range(0, len(ages), 3):
        cols = st.columns(3)
        for j, (label, q) in enumerate(ages[i:i+3]):
            with cols[j]:
                chip(label, q, key=f"chip_age_curated_{i}_{j}")



    # -------  Searchable multiselect -------
    st.markdown("#### Build your own search")
    colA, colB, colC = st.columns([3, 2, 2])

    with colA:
        sel_themes = st.multiselect("Themes (type to search)", sorted(theme_counts.keys()), max_selections=3)
    with colB:
        sel_tone = st.selectbox("Tone (optional)", [""] + sorted(tone_counts.keys()))
    with colC:
        sel_age = st.selectbox(
            "Age (optional)",
            [""] + sorted(age_counts.keys(), key=_age_sort_key),
            format_func=lambda a: ("—" if not a else (f"Ages {a.replace('-', '–')}" if "-" in a else f"Age {a}")),
        )

    if st.button("Search with selected", key="builder_go"):
        parts = []
        if sel_themes:
            parts.append("book about " + ", ".join(sel_themes))
        if sel_tone:
            parts.append(sel_tone)
        if sel_age:
            parts.append(f"books for ages {sel_age}" if "-" in sel_age else f"book for a {sel_age} year old")
        q = " ".join(parts) if parts else "children's book"
        _set_query_and_page(q)



if page == "Recommendations":
    st.title("📚 Recommendations")

    # SEARCH FORM 
    with st.form("search_form", clear_on_submit=False):
        st.markdown('<div class="k-searchbox">', unsafe_allow_html=True)
        q_col, btn_col = st.columns([5, 1], vertical_alignment="center")
        with q_col:
            st.text_input(
                "Describe the book you want…",
                placeholder="e.g., 5-year-old • friendship • adventurous",
                label_visibility="collapsed",
                key="user_query",
            )
        with btn_col:
            submitted = st.form_submit_button("🔍 Search", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    # Home page buttons as submit as well
    submitted = submitted or st.session_state.pop("do_search", False)

    if submitted:
        st.session_state.do_search = False
        st.session_state.generated = True
        st.session_state.expanded_cards = set()  # collapse all on new search
        st.session_state.page_num_recs = 1  # reset page on new search

        # Compute/stash recs ONCE when submitted
        parsed_raw = parse_user_query(st.session_state.user_query)
        parsed = sanitize_parsed(st.session_state.user_query, parsed_raw)

        themes_list = _clean_themes(parsed.get("themes", []))
        tone_word = parsed.get("tone", "")

        st.session_state.filters = {
            "age_range": parsed.get("age_range", ""),
            "tone": parsed.get("tone", ""),
            "themes": themes_list,
        }

        q_raw = (st.session_state.user_query or "").strip()
        q_norm = _norm_title(q_raw)

        title_hits = pd.DataFrame()
        if q_norm:
            df = CATALOG_DF.copy()

            # --- Title search (exact + token AND fuzzy) ---
            mask_exact = (df["title_norm"] == q_norm) | (df["ol_title_norm"] == q_norm)

            tokens = [t for t in tokenize_alpha(q_norm) if t not in TITLE_STOPWORDS]

            if tokens:
                mask_fuzzy = True
                for t in tokens:
                    mask_fuzzy = mask_fuzzy & (
                        df["title_norm"].str.contains(t) | df["ol_title_norm"].str.contains(t)
                    )
            else:
                mask_fuzzy = False

            title_hits = df[mask_exact | mask_fuzzy]

            # --- AUTHOR FALLBACK  ---
            if title_hits.empty and tokens:  
                mask_author = True
                for t in tokens:
                    mask_author = mask_author & (
                        df["author_norm"].str.contains(t) | df["ol_author_norm"].str.contains(t)
                    )
                author_hits = df[mask_author]
                if not author_hits.empty:
                    title_hits = author_hits

        # Always respect explicit age if provided; only broaden if nothing matches
        age_range = parsed.get("age_range", "")
        tone_word = parsed.get("tone", "")
        themes_list = _clean_themes(parsed.get("themes", []))

        if age_range:
            recs = recommend_books(age_range, themes_list, tone_word, n=None)
            # Optional fallback: if truly nothing matched, try without age
            if isinstance(recs, pd.DataFrame) and recs.empty:
                recs = recommend_books(None, themes_list, tone_word, n=None)
        else:
            # No age supplied => broad search
            recs = recommend_books(None, themes_list, tone_word, n=None)

        # If we have title hits/ age filter, combine them with recommender results
        if not age_range and not title_hits.empty:
            # Combine title hits with recommender results
            combined = pd.concat([title_hits, recs], ignore_index=True)
            # Remove duplicates based on title/author
            combined["kidlit_key"] = combined.apply(lambda r: _book_key(as_dict(r)), axis=1)
            recs = combined.drop_duplicates(subset=["kidlit_key"]).drop(columns=["kidlit_key"])
        if not isinstance(recs, pd.DataFrame):
            recs = pd.DataFrame(recs)

        # Ensure columns exist
        for col in [
            "summary_gpt",
            "themes",
            "tone",
            "age_range",
            "cover_url",
            "author",
            "title",
            "ol_title",
            "ol_author",
            "goodreads_url",
            "openlibrary_url",
            "description",
            "summary",
        ]:
            if col not in recs.columns:
                recs[col] = ""

        # Kill NaNs and placeholder/empty rows
        recs = recs.fillna("")

        def _empty(s) -> bool:
            return str(s or "").strip().lower() in ("", "nan", "none", "null")

        # Drop rows where BOTH title and author are empty
        recs = recs[~(recs["title"].apply(_empty) & recs["author"].apply(_empty))]

        # Drop any legacy “no matches” placeholders 
        def _is_placeholder_title(s: str) -> bool:
            t = _norm_title(s)
            return t in {"no matches", "no match", "no results", "no result", "⚠️ no matches", "!no matches"}

        recs = recs[~recs["title"].apply(_is_placeholder_title)]

        recs["kidlit_key"] = recs.apply(lambda r: _book_key(as_dict(r)), axis=1)
        recs = recs.drop_duplicates(subset=["kidlit_key"]).drop(columns=["kidlit_key"])

       # --- LLM tone compatibility (Filter + Fallback) ---
        tone_raw = (parsed.get("tone") or "").strip().lower()

        if tone_raw:
            recs["tones_norm_list"] = recs["tone"].apply(_tones_list)
            strict = recs[recs["tones_norm_list"].apply(lambda lst: tone_raw in (lst or []))]

            # If already have enough exact matches, use them and skip GPT
            MIN_STRICT = 6
            if len(strict) >= MIN_STRICT:
                recs = strict
            else:
                # Try GPT soft-match; if scoring fails, fall back to strict (even if small)
                try:
                    from scripts.core.llm_filters import score_rows_by_tone

                    MAX_TO_SCORE = 150
                    HARD_TH      = 3   # 4=perfect, 3=good
                    SOFT_TH      = 2   # 2=okay
                    MIN_KEEP     = 6

                    scored = score_rows_by_tone(recs.head(MAX_TO_SCORE).copy(), tone_raw, tone_col="tone")

                    hard = scored[scored["__tone_score"] >= HARD_TH]
                    soft = scored[scored["__tone_score"] >= SOFT_TH]

                    # always include any strict matches found
                    def _dedup_keep(df):
                        df = df.copy()
                        df["kidlit_key"] = df.apply(lambda r: _book_key(as_dict(r)), axis=1)
                        return df.drop_duplicates(subset=["kidlit_key"]).drop(columns=["kidlit_key"])

                    if len(hard) >= MIN_KEEP:
                        recs = _dedup_keep(pd.concat([strict, hard], ignore_index=True))
                        st.caption("Not many exact tone matches; including close tone matches.")
                    elif len(soft) >= min(3, MIN_KEEP):
                        recs = _dedup_keep(pd.concat([strict, soft], ignore_index=True))
                        st.caption("Limited exact matches; showing the nearest tone matches.")
                    else:
                        # Scoring produced nothing useful—show strict (even if small)
                        recs = strict if not strict.empty else recs
                        if strict.empty:
                            st.caption("No tone matches found; showing age/theme matches only.")
                except Exception:
                    # GPT not available / error: just use strict results
                    recs = strict if not strict.empty else recs

        if (
            age_range
            and re.fullmatch(r"\d{1,2}", str(age_range).strip())
            and not recs.empty
            and "age_range" in recs.columns
        ):
            target = int(age_range)
            recs = recs[
                recs["age_range"].apply(
                    lambda v: (lambda sp: (sp is not None) and (sp[0] <= target <= sp[1]))(parse_age_span(v))
                )
            ]
        st.session_state.recs_df = recs  # store for reruns

    # Render using stored results (no recompute on toggle)
    if st.session_state.get("generated") and st.session_state.get("recs_df") is not None:
        # Use sanitized filters saved on submit; if absent, recompute + sanitize
        if "filters" in st.session_state and st.session_state.filters:
            parsed = st.session_state.filters
        else:
            parsed = sanitize_parsed(
                st.session_state.user_query, parse_user_query(st.session_state.user_query)
            )

        themes_list = _clean_themes(parsed.get("themes", []))
        themes_display = ", ".join(themes_list) if themes_list else "—"
        tone_display = parsed.get("tone") or "—"
        age_display = parsed.get("age_range") or "—"

        st.markdown(f"**Filters:** Age {age_display}, Tone {tone_display}, Themes {themes_display}")

        recs = st.session_state.recs_df.copy()
        skip_keys = {_book_key(as_dict(b)) for b in st.session_state.skipped_books}
        if skip_keys and not recs.empty:
            recs = recs[~recs.apply(lambda r: _book_key(as_dict(r)) in skip_keys, axis=1)]

        if recs.empty:
            st.warning("No matches. Try different filters.")
        else:
            total = len(recs)
            # Page size now comes from the SIDEBAR selectbox
            page_size = int(st.session_state.get("page_size_recs", 9))

            # Compute page numbers, clamp, and store in state
            max_page = max(1, math.ceil(total / page_size))
            page_num = int(st.session_state.get("page_num_recs", 1))
            page_num = max(1, min(page_num, max_page))
            st.session_state.page_num_recs = page_num

            # Slice current page
            start = (page_num - 1) * page_size
            end = start + page_size
            recs_page = recs.iloc[start:end]

            # Top-of-results info
            # Top toolbar: total | showing | per-page selector
            size_options = [9, 12, 18, 24, 36, 48]
            current_size = int(st.session_state.get("page_size_recs", 9))
            idx = size_options.index(current_size) if current_size in size_options else 0

            left, mid, right = st.columns([2, 3, 2], vertical_alignment="center")
            with left:
                st.markdown(f"**Total matches:** {total}")
            with mid:
                st.caption(f"Showing {start+1}–{min(end, total)} of {total}")
            with right:
                st.selectbox(
                    "Results per page",          # <- visible label now
                    size_options,
                    index=idx,
                    key="page_size_recs",
                    on_change=_reset_pagination,
                    help="How many books to show on this page",
                )



            # Grid
            render_book_grid(recs_page, prefix="rec", show_actions=True, cols=3)

            # ---- Bottom pager (Previous / Page X of Y / Next) ----
            st.markdown("<br>", unsafe_allow_html=True)
            st.divider()

            prev_col, info_col, next_col = st.columns([1, 2, 1], vertical_alignment="center")
            with prev_col:
                if st.button("⬅️ Previous", use_container_width=True, disabled=(page_num <= 1), key="prev_btn"):
                    st.session_state.page_num_recs = max(1, page_num - 1)
                    st.rerun()

            with info_col:
                st.markdown(
                    f"<div style='text-align:center; font-weight:600;'>Page {page_num} / {max_page}</div>",
                    unsafe_allow_html=True,
                )

            with next_col:
                if st.button("Next ➡️", use_container_width=True, disabled=(page_num >= max_page), key="next_btn"):
                    st.session_state.page_num_recs = min(max_page, page_num + 1)
                    st.rerun()

elif page == "Favorites":
    st.title("❤️ Favorites")
    if not st.session_state.liked_books:
        st.info("No favorites yet.")
    else:
        fav_df = pd.DataFrame(st.session_state.liked_books)
        render_book_grid(fav_df, prefix="fav", show_actions=False, cols=3, page_mode="Favorites")

elif page == "Skipped":
    st.title("🚫 Skipped")
    if not st.session_state.skipped_books:
        st.info("No skipped books.")
    else:
        skip_df = pd.DataFrame(st.session_state.skipped_books)
        render_book_grid(skip_df, prefix="skip", show_actions=False, cols=3, page_mode="Skipped")

elif page == "Read":
    st.title("📖 Read")
    if not st.session_state.read_books:
        st.info("No books marked as read.")
    else:
        read_df = pd.DataFrame(st.session_state.read_books)
        render_book_grid(read_df, prefix="read", show_actions=False, cols=3, page_mode="Read")

