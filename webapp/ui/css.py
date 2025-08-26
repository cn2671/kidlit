# ==============================================================================
# Global CSS for Streamlit
# ==============================================================================

import streamlit as st

def inject_global_css():
    st.markdown(
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<link rel="stylesheet" '
        'href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap">',
        unsafe_allow_html=True,
    )

    # ADD Font Awesome ONCE, as a string
    st.markdown(
        '<link rel="stylesheet" '
        'href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">',
        unsafe_allow_html=True,
    )

    # ADD the .k-fa rule INSIDE a <style> block (as a string)
    st.markdown(
        """
        <style>
          .k-fa { font-size: 20px; line-height: 1; }
        </style>
        """,
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
   CARD HOVER 
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
.k-card-body{ padding: 16px 24px 28px; overflow: visible; }
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

/* ===== Summary: "Sneak Peek" ===== */
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

/* ===== Sidebar Menu (segmented pills) ===== */
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

/* Hide radio-dot block in the sidebar menu */
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

/* Tighten spacing now that the dot is gone */
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label[data-baseweb="radio"] {
  padding-left: 12px !important;
}

/* --- External link buttons under Sneak Peek --- */
/* Centered bar under the summary */
/* Centered bar under the summary */
.k-linkbar{
  display:flex; justify-content:center; align-items:center;
  gap:12px;
  margin:16px 0 16px;        
  flex-wrap:wrap;
}

/* Icon-only circular buttons */
.k-iconbtn{
  display:inline-flex; align-items:center; justify-content:center;
  width:40px; height:40px; border-radius:999px;
  border:1px solid #e5e7eb; background:#ffffff;
  box-shadow:0 1px 2px rgba(0,0,0,.06);
  text-decoration:none;
}
.k-iconbtn:hover{ filter:brightness(0.98); transform: translateY(-1px); }
.k-iconbtn:focus-visible{ outline:2px solid #6366f1; outline-offset:2px; border-color:#6366f1; }

/* Icon buttons */
.k-iconbtn--goodreads,
.k-iconbtn--amazon{
  background:#ffffff !important;   /* or: background: transparent */
  border-color:#e5e7eb !important; /* slate-300 */
  box-shadow:0 1px 2px rgba(0,0,0,.06);
}
.k-iconbtn:hover{ filter: none; box-shadow:0 2px 6px rgba(0,0,0,.10); }


/* Icon size */
.k-ico{ width:20px; height:20px; display:block; }

/* Ensure buttons aren’t clipped or blocked */
.k-card-body, .k-linkbar { position: relative; z-index: 5; overflow: visible; }
[data-testid="stVerticalBlock"]:has(> .element-container .k-card-sentinel),
[data-testid="column"]:has(> .element-container .k-card-sentinel){
  overflow: visible; position: relative;
}
[data-testid="stVerticalBlock"]:has(> .element-container .k-card-sentinel)::after,
[data-testid="column"]:has(> .element-container .k-card-sentinel)::after{
  pointer-events: none;
}





</style>
""", unsafe_allow_html=True)
