from __future__ import annotations
import urllib.parse
import streamlit as st
from typing import Any, Iterable
from webapp.data_io import as_dict, _book_key, _norm, _norm_title



# ==============================================================================
# Rendering helpers for cards
# ==============================================================================

def _safe_key_fragment(s: str) -> str:
    """Return a filesystem/HTML‑id friendly fragment."""
    import re
    return re.sub(r"[^a-z0-9_-]+", "_", _norm(s))[:60]


def _pills_html(themes: Iterable[str] | str) -> str:
    raw = themes or ""
    themes_list = [t.strip() for t in (raw if isinstance(raw, list) else str(raw).split(",")) if t.strip()]
    return "".join(f'<span class="k-pill">{p}</span>' for p in themes_list)


def _nz(v: object, default: str = "") -> str:
    """Coerce NaN/None/"nan"/"null" to a clean string."""
    s = "" if v is None else str(v)
    s_l = s.strip().lower()
    if s_l in ("nan", "none", "null"):
        return default
    return s.strip() or default


def render_book_card(row: Any, key_prefix: str, show_actions: bool = True, page_mode: str | None = None) -> None:
    data = as_dict(row)
    cover = _nz(data.get("cover_url"))
    title = _nz(data.get("title")) or _nz(data.get("ol_title")) or "Untitled"
    author = _nz(data.get("author")) or _nz(data.get("ol_author")) or "Unknown"

    def esc(s: str) -> str:
        return str(s).replace('"', '&quot;')

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
    age = _nz(data.get("age_range")) or "—"
    pills_html = _pills_html(data.get("themes"))

    cid = _book_key(data) or key_prefix
    opened = cid in st.session_state.expanded_cards
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
        {('<img src="%s" alt="Cover" class="k-cover" width="90" height="120" />' % cover)
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
            a1, a2, a3 = st.columns(3, vertical_alignment="center")
            with a1:
                if st.button("👍 Like", key=f"{safe}_like"):
                    if k not in {_book_key(b) for b in st.session_state.liked_books}:
                        st.session_state.liked_books.append(data)
                        # If it was in read, remove there (mutually exclusive)
                        st.session_state.read_books = [
                            b for b in st.session_state.read_books if _book_key(b) != k
                        ]
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
                        # If it was in liked, remove there (mutually exclusive)
                        st.session_state.liked_books = [
                            b for b in st.session_state.liked_books if _book_key(b) != k
                        ]
                        st.toast(f"Marked as Read: {title}", icon="📘")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Compact per‑page action row
            st.markdown('<div class="k-header-actions">', unsafe_allow_html=True)
            if page_mode == "Favorites":
                if st.button("🗑️ Remove", key=f"{safe}_remove_fav"):
                    st.session_state.liked_books = [
                        b for b in st.session_state.liked_books if _book_key(as_dict(b)) != k
                    ]
                    st.rerun()
            elif page_mode == "Skipped":
                if st.button("↩️ Unskip", key=f"{safe}_unskip"):
                    st.session_state.skipped_books = [
                        b for b in st.session_state.skipped_books if _book_key(as_dict(b)) != k
                    ]
                    st.rerun()
            elif page_mode == "Read":
                if st.button("🗑️ Remove", key=f"{safe}_remove_read"):
                    st.session_state.read_books = [
                        b for b in st.session_state.read_books if _book_key(as_dict(b)) != k
                    ]
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # TOGGLE 
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
            if data.get("goodreads_url"):
                links.append(f'[Goodreads]({data.get("goodreads_url")})')
            if data.get("openlibrary_url"):
                links.append(f'[Open Library]({data.get("openlibrary_url")})')
            if links:
                st.markdown(" • ".join(links))
            st.markdown('</div>', unsafe_allow_html=True)
