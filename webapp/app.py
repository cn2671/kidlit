import streamlit as st
import pandas as pd
import urllib.parse
import math
import sys
import os
import re

# --- Project imports / path setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from scripts.parse_query import parse_user_query
from scripts.text_utils import (
    BASE_STOPWORDS as STOPWORDS,   
    TITLE_STOPWORDS,
    tokenize_alpha,
    clean_themes as _clean_themes, 
    detect_age_from_text,
    strip_age_category_tokens,
)
from scripts.recommender import recommend_books
from scripts.recommender import parse_age_span

# ------------- Utilities -------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())
  
def _norm_title(s: str) -> str:
    # remove any series/volume bits in parentheses and normalize spaces/case
    s = re.sub(r"\([^)]*\)", "", str(s or ""))
    s = s.replace("–", "-").replace("—", "-")
    return _norm(s)


def _book_key(book: dict) -> str:
    # tolerate non-dicts too
    try:
        t = book.get("title") or book.get("ol_title") or ""
        a = book.get("author") or book.get("ol_author") or ""
    except AttributeError:
        # e.g., a pandas Series/Row
        b = as_dict(book)
        t = b.get("title") or b.get("ol_title") or ""
        a = b.get("author") or b.get("ol_author") or ""

    k = f"{_norm(t)}|{_norm(a)}"
    # if both title/author are blank, add a tie-breaker so keys don’t all collide
    if k == "|":
        extra = _norm(
            (b if 'b' in locals() else book).get("openlibrary_url")
            or (b if 'b' in locals() else book).get("goodreads_url")
            or (b if 'b' in locals() else book).get("title")
            or ""
        )
        if extra:
            k = f"{k}|{extra}"
    return k


def _safe_key_fragment(s: str) -> str:
    return re.sub(r'[^a-z0-9_-]+', '_', _norm(s))[:60]

def as_dict(row):
    if isinstance(row, dict):
        return row
    try:
        return row.to_dict()
    except Exception:
        return dict(row)

def dedupe_books(lst: list[dict]) -> list[dict]:
    seen, out = set(), []
    for b in lst:
        k = _book_key(b if isinstance(b, dict) else dict(b))
        if k not in seen:
            seen.add(k)
            out.append(b)
    return out

@st.cache_data
def load_catalog(path="books_llm_tags.csv"):   # <- make path match recommender
    try:
        df = pd.read_csv(path).fillna("")
        for col in ["title", "ol_title", "author", "ol_author", "themes", "tone"]:
            if col not in df.columns:
                df[col] = ""
        df["title_norm"]     = df["title"].apply(_norm_title)
        df["ol_title_norm"]  = df["ol_title"].apply(_norm_title)
        df["author_norm"]    = df["author"].apply(_norm)
        df["ol_author_norm"] = df["ol_author"].apply(_norm)

        # NEW: normalized tag arrays
        df["themes_norm_list"] = df["themes"].apply(_split_tags)
        df["tones_norm_list"]  = df["tone"].apply(_split_tags)
        return df
    except Exception:
        return pd.DataFrame()



def _split_tags(s: str) -> list[str]:
    s = str(s or "")
    # If tags are stored like ["magic","friendship"], parse JSON
    if s.strip().startswith("[") and s.strip().endswith("]"):
        try:
            import json
            arr = json.loads(s)
            return [_norm(x) for x in arr if isinstance(x, str) and _norm(x)]
        except Exception:
            pass
    # Otherwise split on commas / semicolons / slashes / pipes
    parts = re.split(r"[;,/|]+", s)
    parts = [_norm(p) for p in parts if _norm(p)]
    # de-dup preserving order
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p); out.append(p)
    return out


def build_index(df: pd.DataFrame) -> dict:
    idx = {}
    for _, r in df.iterrows():
        d = r.to_dict()
        k = _book_key(d)
        if k:
            idx[k] = d
    return idx

def rehydrate_book(b, index: dict):
    d = as_dict(b)
    k = _book_key(d)
    src = index.get(k)
    if src:
        for col in [
            "author","ol_author","summary_gpt","themes","tone","age_range",
            "cover_url","title","ol_title","goodreads_url","openlibrary_url","description"
        ]:
            if not d.get(col):
                d[col] = src.get(col, d.get(col))
    return d

def _as_list(x):
    """Coerce x into a clean list of strings."""
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return [str(t).strip() for t in x if str(t).strip()]
    if isinstance(x, str):
        import re as _re
        parts = _re.split(r'[;,/|]+', x)
        return [p.strip() for p in parts if p.strip()]
    try:
        return [str(t).strip() for t in list(x) if str(t).strip()]
    except Exception:
        s = str(x).strip()
        return [s] if s else []

def _reset_pagination():
    st.session_state.page_num_recs = 1

def _user_supplied_age(text: str) -> str:
    """
    Return normalized 'A-B' (range) or single 'N' IF the user typed it; else ''.
    Handles: '6-8', '6–8', 'age 5', '5 year old', '5yo', '5yr old', '5year old', '5yrs', '5yearold'.
    """
    t = (text or "").lower()

    # ranges like 6-8 / 6–8 / 6—8
    m = re.search(r"\b(\d{1,2})\s*[–—-]\s*(\d{1,2})\b", t)
    if m:
        a, b = m.groups()
        return f"{a}-{b}"

    # explicit "age 5"
    m = re.search(r"\bage\s*(\d{1,2})\b", t)
    if m:
        return m.group(1)

    # "5 year old", "5yo", "5 yr old", "5year old", "5yrs"
    m = re.search(r"\b(\d{1,2})\s*(?:yo|yr|yrs?|year|years)?\s*(?:old)?\b", t)
    if m:
        return m.group(1)

    # glued form: "5yearold"
    m = re.search(r"\b(\d{1,2})yearold\b", t)
    if m:
        return m.group(1)

    return ""



def should_treat_as_theme(word: str) -> bool:
    w = _norm(word)
    if not w or CATALOG_DF.empty:
        return False
    if "themes_norm_list" not in CATALOG_DF.columns or "tones_norm_list" not in CATALOG_DF.columns:
        return False
    theme_count = int((CATALOG_DF["themes_norm_list"].apply(lambda lst: w in lst)).sum())
    tone_count  = int((CATALOG_DF["tones_norm_list"].apply(lambda lst: w in lst)).sum())
    return theme_count >= max(2, tone_count * 2)
  

def should_treat_as_tone(word: str) -> bool:
    w = _norm(word)
    if not w or CATALOG_DF.empty: 
        return False
    if "themes_norm_list" not in CATALOG_DF.columns or "tones_norm_list" not in CATALOG_DF.columns:
        return False
    theme_count = int((CATALOG_DF["themes_norm_list"].apply(lambda lst: w in lst)).sum())
    tone_count  = int((CATALOG_DF["tones_norm_list"].apply(lambda lst: w in lst)).sum())
    return tone_count >= max(1, theme_count * 2)


def sanitize_parsed(user_text: str, parsed: dict) -> dict:
    out = dict(parsed or {})
    themes = _as_list(out.get("themes", []))
    tone   = (out.get("tone") or "").strip()

    # keep canonical tones
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
    AGE_WORDS = {"year","years","yr","yrs","yo","old"}
    themes = [
        t for t in themes
        if _norm(t) not in AGE_WORDS and not re.fullmatch(r"\d+", _norm(t))
    ]

    out["themes"] = _clean_themes(themes)
    return out




# ==== in app.py, near other globals/utilities ====
TONE_CANON = {
    "whimsical","calm","funny","silly","gentle","heartfelt","mysterious",
    "spooky","sad","uplifting","adventurous","exciting","cozy","poetic",
    "suspenseful","inspiring","quiet","witty","sweet","goofy","dramatic",
}

def is_canonical_tone(word: str) -> bool:
    return (_norm(word) in TONE_CANON)





# ------------- Page Config & State -------------

st.set_page_config(page_title="KidLit", page_icon="📚", layout="wide")

for key in ["user_query", "generated", "liked_books", "skipped_books", "read_books"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key.endswith("_books") else (False if key == "generated" else "")

if "recs_df" not in st.session_state:
    st.session_state.recs_df = None
if "last_query_str" not in st.session_state:
    st.session_state.last_query_str = ""

if "expanded_cards" not in st.session_state:
    st.session_state.expanded_cards = set()

# normalize legacy skipped entries (strings -> dicts)
st.session_state.skipped_books = [
    {"title": b} if isinstance(b, str) else as_dict(b)
    for b in st.session_state.skipped_books
]

# load catalog & rehydrate lists for full card data
CATALOG_DF  = load_catalog("books_llm_tags.csv")
CATALOG_IDX = build_index(CATALOG_DF)

st.session_state.liked_books   = [rehydrate_book(b, CATALOG_IDX) for b in st.session_state.liked_books]
st.session_state.read_books    = [rehydrate_book(b, CATALOG_IDX) for b in st.session_state.read_books]
st.session_state.skipped_books = [rehydrate_book(b, CATALOG_IDX) for b in st.session_state.skipped_books]

# Deduplicate after rehydration
st.session_state.liked_books   = dedupe_books(st.session_state.liked_books)
st.session_state.read_books    = dedupe_books(st.session_state.read_books)
st.session_state.skipped_books = dedupe_books(st.session_state.skipped_books)


# ------------- Global CSS -------------

st.markdown(
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap">',
    unsafe_allow_html=True,
)
st.markdown("""
<style>
:root{
  --k-hdr-h: 300px;              /* fixed header height for aligned rows */
  --k-bg: #fff;                  /* card background */
  --k-border: #cbd5e1;           /* base border (slate-300) */
  --k-accent: #6366f1;           /* indigo accent */
  --k-ring: rgba(99,102,241,.18);/* hover ring */
  --k-accent-bg: rgba(99,102,241,.06); /* full-card hover tint */
}

.block-container{
  max-width: 1300px; padding-top: 0.75rem;
  font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, 'Apple Color Emoji', 'Segoe UI Emoji';
}

/* =========================================================
   CARD HOVER scoped to the Streamlit container with sentinel
   ========================================================= */
[data-testid="stVerticalBlock"]:has(> .element-container .k-card-sentinel),
[data-testid="column"]:has(> .element-container .k-card-sentinel) {
  position: relative;
  width: 100%;
  background: var(--k-bg);
  border: 2px solid var(--k-border);
  border-radius: 14px;
  box-shadow: none;
  transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease, background .18s ease;
  overflow: hidden;
  cursor: pointer;
}
[data-testid="stVerticalBlock"]:has(> .element-container .k-card-sentinel):hover,
[data-testid="column"]:has(> .element-container .k-card-sentinel):hover {
  transform: translateY(-2px) scale(1.01);
  border-color: var(--k-accent);
  box-shadow: 0 0 0 4px var(--k-ring), 0 10px 24px rgba(0,0,0,.12);
}
/* Overlay tint that doesn't block clicks */
[data-testid="stVerticalBlock"]:has(> .element-container .k-card-sentinel)::after,
[data-testid="column"]:has(> .element-container .k-card-sentinel)::after {
  content: "";
  position: absolute; inset: 0;
  background: var(--k-accent-bg);
  opacity: 0;
  transition: opacity .18s ease;
  pointer-events: none;
}
[data-testid="stVerticalBlock"]:has(> .element-container .k-card-sentinel):hover::after,
[data-testid="column"]:has(> .element-container .k-card-sentinel):hover::after {
  opacity: 1;
}

/* ==========================
   Inner card layout & content
   ========================== */
.k-card-header{
  padding: 14px 16px;
  height: var(--k-hdr-h);
  display: flex; flex-direction: column;
}
.k-header-scroll{ flex: 1 1 auto; overflow: auto; padding-right: 4px; }
.k-header-row{ display:flex; gap:12px; align-items:flex-start; }
.k-cover{ border-radius: 10px; object-fit: cover; }
.k-header-title{ margin:0 0 4px; font-size:1.05rem; }
.k-header-meta{ margin:0 6px 6px 0; color:#4b5563; display:inline-block; }
.k-pill{
  display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px;
  background:#eef2ff; color:#3730a3; margin-right:6px; margin-bottom:6px;
}
.k-meta{ color:#6b7280; font-size: 13px; margin-top: 6px; }

/* Center any "toggle" button when the sentinel is present */
[data-testid="stVerticalBlock"]:has(> .element-container .k-card-sentinel)
  > .element-container:has(.k-toggle-sentinel) .stButton,
[data-testid="column"]:has(> .element-container .k-card-sentinel)
  > .element-container:has(.k-toggle-sentinel) .stButton {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  margin: 8px 0 0 0;
  padding: 0;
}

/* Action row (Like/Skip/Read) */
.k-header-actions{ margin: 10px 16px 0 16px; }

/* Body & divider */
.k-card-body{ padding: 16px 24px; }
.k-divider{ height:1px; background:#e5e7eb; margin: 0 16px; }

/* Summary spacing + side padding */
.k-summary-full{
  display: block;
  overflow: visible;
  margin: 16px 0 20px 0;
  padding: 0 12px;
  line-height: 1.5;
}

/* Inputs & generic buttons */
.stTextInput > div > div > input{
  border-radius: 10px; border: 1px solid #e5e7eb; padding: 10px 12px;
}
.stButton>button{
  border-radius: 10px !important; padding: 6px 12px !important; font-weight: 600 !important;
}
.stButton>button:hover{ filter: brightness(0.98); }

.stButton > button {
  margin-left: auto;
  margin-right: auto;
  display: block;
}

/* Hero */
.k-hero{
  padding:18px 20px; border-radius:16px; background:linear-gradient(90deg,#eef2ff,#e0f2fe);
  border:1px solid #e5e7eb; margin-bottom:12px;
}

/* ===== Playful KidLit pill for "Sneak Peek" ===== */
.k-summary-btn .stButton > button {
  background: linear-gradient(90deg,#fef9c3,#fde68a) !important;
  color: #7c2d12 !important;
  border-radius: 16px !important;
  padding: 10px 18px !important;
  border: 1px solid #facc15 !important;
  font-weight: 700 !important;
  box-shadow: none !important;
  transition: transform .15s ease, box-shadow .15s ease;
}
.k-summary-btn .stButton > button:hover {
  transform: scale(1.05) rotate(-1deg);
  box-shadow: 0 3px 8px rgba(250,204,21,0.3);
}

/* Sneak Peek: caret ghost style */
.k-summary-ghost .stButton > button{
  background: transparent !important;
  color: #0f766e !important;               /* teal-800 */
  border: 1px solid #99f6e4 !important;    /* teal-200 */
  border-radius: 12px !important;
  padding: 6px 12px !important;
  font-weight: 700 !important;
  box-shadow: none !important;
}
.k-summary-ghost .stButton > button:hover{
  background: #ccfbf1 !important;          /* teal-100 */
}


/* Force light mode always */
html { color-scheme: light !important; }

/* --- Reduce motion --- */
@media (prefers-reduced-motion: reduce){
  [data-testid="stVerticalBlock"]:has(> .element-container .k-card-sentinel):hover,
  [data-testid="column"]:has(> .element-container .k-card-sentinel):hover { transform: none; }
  [data-testid="stVerticalBlock"]:has(> .element-container .k-card-sentinel),
  [data-testid="column"]:has(> .element-container .k-card-sentinel) {
    transition: border-color .18s ease, background .18s ease, box-shadow .18s ease;
  }
}

/* ===== Prettier Sidebar Menu (segmented pills) ===== */
section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"]{
  display: flex;
  gap: 8px;
  padding: 6px;
  align-items: center;
  justify-content: space-between;
}
section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label{
  flex: 1 1 0;
  display: flex; align-items: center; justify-content: center;
  gap: 8px; padding: 10px 12px;
  border-radius: 999px;
  cursor: pointer; font-weight: 600; color: #374151;
  transition: background .15s ease, color .15s ease, box-shadow .15s ease;
  border: 1px solid transparent;
}
section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover{
  background: #f1f5f9;
}
section[data-testid="stSidebar"] .stRadio input[type="radio"]{
  position: absolute; opacity: 0; pointer-events: none;
}
section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:has(input:checked){
  background: var(--k-accent-bg, #eef2ff);
  color: #3730a3; border-color: #c7d2fe;
  box-shadow: 0 0 0 2px var(--k-ring, rgba(99,102,241,.18)) inset;
}

/* Sidebar hero */
section[data-testid="stSidebar"] .k-side-hero{
  margin: 8px 0 14px; padding: 14px 16px;
  border-radius: 14px; background: linear-gradient(90deg,#eef2ff,#e0f2fe);
  border: 1px solid #e5e7eb;
}
section[data-testid="stSidebar"] .k-side-hero h3{ margin: 0 0 4px 0; font-size: 1.05rem; }
section[data-testid="stSidebar"] .k-side-hero p{ margin: 0; color: #475569; font-size: .9rem; }

/* ===== KidLit Logo Title ===== */
.kidlit-logo {
  font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  font-size: 2.2rem;
  font-weight: 700;
  letter-spacing: -0.5px;
  background: linear-gradient(90deg, #6366f1, #06b6d4);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  display: inline-block;
  margin: 0 auto;
}
.kidlit-sub {
  margin: 6px 0 0; color: #475569; font-size: 1rem; font-weight: 500;
}

/* ===== Sidebar Menu Styling ===== */
div[data-testid="stSidebar"] .stRadio > label {
    justify-content: flex-start !important;
    text-align: left !important;
}



/* Style each menu item */
div[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label {
    padding: 8px 14px;
    border-radius: 8px;
    font-weight: 500;
    color: #334155; /* slate-700 */
    cursor: pointer;
}

/* Active menu highlight */
div[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) {
    background-color: #e0f2fe; /* light cyan */
    color: #0e7490; /* cyan-800 */
    font-weight: 600;
}


/* Center the hero block */
.k-hero {
  text-align: center;
  padding: 18px 20px;
  border-radius: 16px;
  background: linear-gradient(90deg,#eef2ff,#e0f2fe);
  border: 1px solid #e5e7eb;
  margin-bottom: 12px;
}



/* ===== Unified search bar style ===== */
.k-searchbox .stTextInput > div > div > input {
  border-radius: 10px 0 0 10px !important;
  border: 1px solid #e5e7eb !important;
  padding: 10px 12px !important;
  height: 44px;
}
.k-searchbox .stButton > button {
  border-radius: 0 10px 10px 0 !important;
  border: 1px solid #e5e7eb !important;
  border-left: none !important;
  height: 44px;
  font-weight: 600 !important;
  background: var(--k-accent, #6366f1) !important;
  color: white !important;
  cursor: pointer;
}
.k-searchbox .stButton > button:hover { background: #4f46e5 !important; }

/* Make clickable titles look like normal text (but interactive on hover) */
.book-link { text-decoration: none !important; color: inherit !important; font-weight: inherit; }
.k-header-title a, .k-header-title a:visited, .k-header-title a:active {
  color: inherit !important; text-decoration: none !important; font-weight: inherit !important;
  padding: 2px 4px; border-radius: 6px; transition: all 0.2s ease-in-out;
}
.k-header-title a:hover {
  text-decoration: none !important; font-weight: 700 !important; color: inherit !important;
  background-color: rgba(99, 102, 241, 0.15); cursor: pointer;
  box-shadow: 0 0 6px rgba(99, 102, 241, 0.25);
}


/* Hide BaseWeb's radio-dot block in the sidebar menu */
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label[data-baseweb="radio"] > div:first-child,
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label[data-baseweb="radio"] > div:first-child * {
  display: none !important;              /* removes the dot wrapper and its inner circle */
}

/* Keep the native input present but invisible so :has(input:checked) still works */
section[data-testid="stSidebar"] .stRadio input[type="radio"] {
  position: absolute !important;
  opacity: 0 !important;
  pointer-events: none !important;
}

/* Optional: tighten spacing now that the dot is gone */
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label[data-baseweb="radio"] {
  padding-left: 12px !important;
}



</style>
""", unsafe_allow_html=True)

# ------------- Rendering -------------

def _pills_html(themes):
    raw = themes or ""
    themes_list = [t.strip() for t in (raw if isinstance(raw, list) else str(raw).split(",")) if t.strip()]
    return "".join(f'<span class="k-pill">{p}</span>' for p in themes_list)

def _nz(v: object, default: str = "") -> str:
    # Coerce NaN/None/"nan"/"null" to a clean string
    s = "" if v is None else str(v)
    s_l = s.strip().lower()
    if s_l in ("nan", "none", "null"): 
        return default
    return s.strip() or default


def render_book_card(row, key_prefix: str, show_actions: bool = True, page_mode: str | None = None):
    data = as_dict(row)
    cover = _nz(data.get("cover_url"))
    title = _nz(data.get("title")) or _nz(data.get("ol_title")) or "Untitled"
    author = _nz(data.get("author")) or _nz(data.get("ol_author")) or "Unknown"

    def esc(s): return str(s).replace('"','&quot;')

    # Build Amazon link (use explicit amazon_url if present; otherwise search title + author)
    explicit_amazon = (data.get("amazon_url") or "").strip()
    if explicit_amazon:
        amazon_url = explicit_amazon
    else:
        q = urllib.parse.quote_plus(f"{title} {author}")
        amazon_url = f"https://www.amazon.com/s?k={q}"

    # Clickable title (styled to look like normal header)
    title_html = f'''
      <a href="{esc(amazon_url)}" target="_blank" rel="noopener noreferrer" class="book-link">
        {esc(title)}
      </a>
    '''

    full_summary = _nz(data.get("summary_gpt")) or _nz(data.get("summary")) or _nz(data.get("description"))
    tone = (_nz(data.get("tone")).title()) or "—"
    age  = _nz(data.get("age_range")) or "—"
    pills_html = _pills_html(data.get("themes"))

    cid = _book_key(data) or key_prefix
    opened = (cid in st.session_state.expanded_cards)
    safe = _safe_key_fragment(cid)

    with st.container():
        # Card sentinel for scoped hover styles
        st.markdown('<div class="k-card-sentinel" aria-hidden="true"></div>', unsafe_allow_html=True)

        # HEADER
        header_html = f"""
<div id="hdr-{safe}" class="k-card-header" role="group" aria-label="{esc(title)} by {esc(author)}">
  <div class="k-header-scroll">
    <div class="k-header-row">
      <div>
        {'<img src="%s" alt="Cover" class="k-cover" width="90" height="120" />' % cover
          if cover else '<div class="k-cover" style="width:90px;height:120px;background:#f3f4f6;"></div>'}
      </div>
      <div style="flex:1;">
        <h3 class="k-header-title" title="{esc(title)}">{title_html}</h3>
        <div><span class="k-header-meta"><em>by {author}</em></span></div>
        <div style="margin:4px 0 6px 0;">{pills_html}</div>
        <div class="k-meta" title="Tone: {esc(tone)} • Ages {esc(age)}">🎭 {tone} &nbsp; • &nbsp; 📅 Ages {age}</div>
      </div>
    </div>
  </div>
</div>
"""
        st.markdown(header_html, unsafe_allow_html=True)

        # ACTIONS
        k = _book_key(data)
        if show_actions:
            st.markdown('<div class="k-header-actions">', unsafe_allow_html=True)
            a1, a2, a3 = st.columns(3)
            with a1:
                if st.button("👍 Like", key=f"{safe}_like"):
                    if k not in {_book_key(b) for b in st.session_state.liked_books}:
                        st.session_state.liked_books.append(data)
                        # if it was in read, remove there (mutually exclusive)
                        st.session_state.read_books = [b for b in st.session_state.read_books if _book_key(b) != k]
                        st.toast(f"Added to Favorites: {title}", icon="❤️")
            with a2:
                if st.button("👎 Skip", key=f"{safe}_skip"):
                    if k not in {_book_key(as_dict(b)) for b in st.session_state.skipped_books}:
                        st.session_state.skipped_books.append(data)
                        st.toast(f"Skipped: {title}", icon="🚫")
            with a3:
                if st.button("📖 ✔︎ Read", key=f"{safe}_read"):
                    if k not in {_book_key(b) for b in st.session_state.read_books}:
                        st.session_state.read_books.append(data)
                        # if it was in liked, remove there (mutually exclusive)
                        st.session_state.liked_books = [b for b in st.session_state.liked_books if _book_key(b) != k]
                        st.toast(f"Marked as Read: {title}", icon="📘")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # compact per-page action row
            st.markdown('<div class="k-header-actions">', unsafe_allow_html=True)
            if page_mode == "Favorites":
                if st.button("🗑️ Remove", key=f"{safe}_remove_fav"):
                    st.session_state.liked_books = [b for b in st.session_state.liked_books if _book_key(as_dict(b)) != k]
                    st.rerun()
            elif page_mode == "Skipped":
                if st.button("↩️ Unskip", key=f"{safe}_unskip"):
                    st.session_state.skipped_books = [b for b in st.session_state.skipped_books if _book_key(as_dict(b)) != k]
                    st.rerun()
            elif page_mode == "Read":
                if st.button("🗑️ Remove", key=f"{safe}_remove_read"):
                    st.session_state.read_books = [b for b in st.session_state.read_books if _book_key(as_dict(b)) != k]
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # TOGGLE (Playful pill) — after actions, before body
        # TOGGLE (Caret ghost) — after actions, before body
        st.markdown('<div class="k-toggle-sentinel" aria-hidden="true"></div>', unsafe_allow_html=True)
        st.markdown('<div class="k-summary-ghost">', unsafe_allow_html=True)
        label = "▲ Close Peek" if opened else "▼ Sneak Peek"
        if st.button(label, key=f"toggle_{safe}"):
            if opened:
                st.session_state.expanded_cards.remove(cid)
            else:
                st.session_state.expanded_cards.add(cid)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


        # BODY
        if opened:
            st.markdown('<div class="k-divider"></div><div class="k-card-body">', unsafe_allow_html=True)
            if full_summary:
                st.markdown(f'<div class="k-summary-full">{full_summary}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="k-summary-full"><em>No summary available.</em></div>', unsafe_allow_html=True)
            links = []
            if data.get("goodreads_url"):   links.append(f'[Goodreads]({data.get("goodreads_url")})')
            if data.get("openlibrary_url"): links.append(f'[Open Library]({data.get("openlibrary_url")})')
            if links: st.markdown(" • ".join(links))
            st.markdown('</div>', unsafe_allow_html=True)

def render_book_grid(df, prefix="rec", show_actions=True, cols=3, page_mode: str | None = None):
    if df is None or df.empty:
        return
    n = len(df)
    for start in range(0, n, cols):
        row_cols = st.columns(cols, vertical_alignment="top")
        for j in range(cols):
            idx = start + j
            if idx >= n:
                break
            with row_cols[j]:
                render_book_card(df.iloc[idx], f"{prefix}_{idx}", show_actions=show_actions, page_mode=page_mode)

# ------------- UI -------------

# Sidebar branding
st.sidebar.markdown(
    '<div class="k-side-hero"><h3>Menu</h3>'
    '<p>Browse your favorites and fresh picks</p></div>',
    unsafe_allow_html=True
)

# Display controls in the sidebar (left side)
# st.sidebar.markdown("### Display")
st.sidebar.selectbox(
    "Results per page",
    [9, 12, 18, 24, 36, 48],
    index=0,
    key="page_size_recs",
    on_change=_reset_pagination  # reset to page 1 when size changes
)

# Prettier segmented menu
MENU = [
    ("🔎 Recommendations", "Recommendations"),
    ("❤️ Favorites",        "Favorites"),
    ("🚫 Skipped",          "Skipped"),
    ("📖 Read",             "Read"),
]
labels = [lbl for lbl, _ in MENU]
values = {lbl: val for lbl, val in MENU}
page_label = st.sidebar.radio("Menu", labels, index=0, label_visibility="collapsed")
page = values[page_label]

# Hero/title
st.markdown(
    '<div class="k-hero">'
    '<h1 class="kidlit-logo">KidLit</h1>'
    '<p class="kidlit-sub">Kid Literature: Personalized children’s books by age, themes, and tone.</p>'
    '</div>',
    unsafe_allow_html=True
)

if page == "Recommendations":
    st.title("📚 Recommendations")

    # SEARCH FORM (Enter submits)
    with st.form("search_form", clear_on_submit=False):
        st.markdown('<div class="k-searchbox">', unsafe_allow_html=True)
        q_col, btn_col = st.columns([5, 1], vertical_alignment="center")
        with q_col:
            st.text_input(
                "Describe the book you want…",
                value=st.session_state.get("user_query", ""),
                placeholder="e.g., 5-year-old • friendship • adventurous",
                label_visibility="collapsed",
                key="user_query"
            )
        with btn_col:
            submitted = st.form_submit_button("🔍 Search", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        st.session_state.generated = True
        st.session_state.expanded_cards = set()  # collapse all on new search
        st.session_state.page_num_recs = 1  # reset page on new search

            # compute/stash recs ONCE when submitted
        parsed_raw = parse_user_query(st.session_state.user_query)
        parsed = sanitize_parsed(st.session_state.user_query, parsed_raw)
        
        themes_list = _clean_themes(parsed.get('themes', []))
        tone_word   = parsed.get('tone', '')

        st.session_state.filters = {
        "age_range": parsed.get("age_range", ""),
        "tone": parsed.get("tone", ""),
        "themes": themes_list,
}

        


        q_raw  = (st.session_state.user_query or "").strip()
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
            if title_hits.empty and tokens:  # q_norm already truthy, reuse the same tokens
                mask_author = True
                for t in tokens:
                    mask_author = mask_author & (
                        df["author_norm"].str.contains(t) | df["ol_author_norm"].str.contains(t)
                    )
                author_hits = df[mask_author]
                if not author_hits.empty:
                    title_hits = author_hits
        
        # Always respect explicit age if provided; only broaden if nothing matches
        age_range = parsed.get('age_range', '')
        tone_word = parsed.get('tone', '')
        themes_list = _clean_themes(parsed.get('themes', []))  # or keep themes_list as you already have

        if age_range:
            recs = recommend_books(age_range, themes_list, tone_word, n=None)
            # Optional fallback: if truly nothing matched, try without age
            if isinstance(recs, pd.DataFrame) and recs.empty:
                recs = recommend_books(None, themes_list, tone_word, n=None)
        else:
            # No age supplied → broad search
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

        # ensure columns exist
        for col in ["summary_gpt","themes","tone","age_range","cover_url","author","title",
                    "ol_title","ol_author","goodreads_url","openlibrary_url","description","summary"]:
            if col not in recs.columns:
                recs[col] = ""
        # kill NaNs and placeholder/empty rows
        recs = recs.fillna("")

        def _empty(s): 
            return (str(s or "").strip().lower() in ("", "nan", "none", "null"))
        # drop rows where BOTH title and author are empty   
        recs = recs[~(recs["title"].apply(_empty) & recs["author"].apply(_empty))]

        # drop any legacy “no matches” placeholders that may sneak in
        def _is_placeholder_title(s):
            t = _norm_title(s)
            return t in {"no matches","no match","no results","no result","⚠️ no matches","!no matches"}
        recs = recs[~recs["title"].apply(_is_placeholder_title)]

        recs["kidlit_key"] = recs.apply(lambda r: _book_key(as_dict(r)), axis=1)
        recs = recs.drop_duplicates(subset=["kidlit_key"]).drop(columns=["kidlit_key"])

        if age_range and re.fullmatch(r"\d{1,2}", str(age_range).strip()) and not recs.empty and "age_range" in recs.columns:
            target = int(age_range)
            recs = recs[recs["age_range"].apply(
                lambda v: (lambda sp: (sp is not None) and (sp[0] <= target <= sp[1]))(parse_age_span(v))
            )]
        st.session_state.recs_df = recs  # store for reruns

    # render using stored results (no recompute on toggle)
    if st.session_state.get("generated") and st.session_state.get("recs_df") is not None:
        # Use sanitized filters saved on submit; if absent, recompute + sanitize
        if "filters" in st.session_state and st.session_state.filters:
            parsed = st.session_state.filters
        else:
            parsed = sanitize_parsed(
                st.session_state.user_query,
                parse_user_query(st.session_state.user_query)
            )

        themes_list    = _clean_themes(parsed.get('themes', []))
        themes_display = ", ".join(themes_list) if themes_list else "—"
        tone_display   = parsed.get('tone') or "—"
        age_display    = parsed.get("age_range") or "—"

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
            st.markdown(f"**Total matches:** {total}")
            st.caption(f"Showing {start+1}–{min(end, total)} of {total}")

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
                    unsafe_allow_html=True
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
