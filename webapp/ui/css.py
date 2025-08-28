# ==============================================================================
# Global CSS for Streamlit
# ==============================================================================

import streamlit as st

def inject_global_css():
    st.markdown(
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap">',
        unsafe_allow_html=True,
    )
    # Font Awesome 
    st.markdown(
        '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">',
        unsafe_allow_html=True,
    )
    # Small icon helper
    st.markdown(
        "<style>.k-fa{font-size:20px;line-height:1;}</style>",
        unsafe_allow_html=True,
    )

    # ---- Global styles ----
    st.markdown(
        """
<style>
:root{
  --k-hdr-h: 300px;              /* soft min-height for header area */
  --k-bg: #fff;
  --k-border: #cbd5e1;
  --k-accent: #6366f1;
  --k-ring: rgba(99,102,241,.18);
  --k-accent-bg: rgba(99,102,241,.06);
}

.block-container{
  max-width: 1300px;
  padding-top: .75rem;
  font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, 'Apple Color Emoji', 'Segoe UI Emoji';
}

/* =========================
   Card hover container
   ========================= */
[data-testid="stVerticalBlock"]:has(> .element-container .k-card-sentinel),
[data-testid="column"]:has(> .element-container .k-card-sentinel){
  position:relative; width:100%;
  background:var(--k-bg);
  border:2px solid var(--k-border);
  border-radius:14px;
  overflow:hidden;
  transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease, background .18s ease;
}
[data-testid="stVerticalBlock"]:has(> .element-container .k-card-sentinel):hover,
[data-testid="column"]:has(> .element-container .k-card-sentinel):hover{
  transform: translateY(-2px) scale(1.01);
  border-color: var(--k-accent);
  box-shadow: 0 0 0 4px var(--k-ring), 0 10px 24px rgba(0,0,0,.12);
}
[data-testid="stVerticalBlock"]:has(> .element-container .k-card-sentinel)::after,
[data-testid="column"]:has(> .element-container .k-card-sentinel)::after{
  content:""; position:absolute; inset:0;
  background:var(--k-accent-bg); opacity:0; pointer-events:none;
  transition: opacity .18s ease;
}
[data-testid="stVerticalBlock"]:has(> .element-container .k-card-sentinel):hover::after,
[data-testid="column"]:has(> .element-container .k-card-sentinel):hover::after{
  opacity:1;
}

/* =========================
   Inner header & content
   ========================= */
.k-card-header{
  padding:14px 16px;
  min-height: var(--k-hdr-h);   /* allow growth; keep a soft floor */
  height:auto;
  display:flex; flex-direction:column;
}
.k-header-scroll{
  flex:1 1 auto;
  overflow:auto;                /* enable scroll for long titles/rows */
  -webkit-overflow-scrolling:touch;
  padding-right:6px;            /* gutter so text isn't under scrollbar */
}
.k-header-row{ display:flex; gap:12px; align-items:flex-start; }
.k-cover{ border-radius:10px; object-fit:cover; }
.k-header-title{
  margin:0 0 4px;
  font-size:1.05rem;
  line-height:1.2;
  display:block;                /* no -webkit-box */
  overflow:visible;             /* show all lines */
  white-space:normal;           /* allow wrap */
  word-break:normal;
  overflow-wrap:normal;
  hyphens:none;
  text-wrap: balance; 
}
.k-header-meta{ margin:0 6px 6px 0; color:#4b5563; display:inline-block; }
.k-pill{
  display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px;
  background:#eef2ff; color:#3730a3; margin-right:6px; margin-bottom:6px;
}
.k-meta{ color:#6b7280; font-size:13px; margin-top:6px; }

/* Action row */
.k-header-actions{ margin:10px 16px 0 16px; }
.k-header-actions .stButton > button{
  white-space:nowrap !important;
  writing-mode: horizontal-tb !important;
  min-width:120px; text-align:center; padding:8px 14px;
}
.k-actions-row .stButton > button{
  white-space:nowrap; min-width:140px; height:46px; padding:8px 16px; font-weight:600;
}
.k-actions-row [data-testid="column"]{ padding-left:6px; padding-right:6px; }

/* Body & divider */
.k-card-body{ padding:10px 24px 28px; overflow:visible; }
.k-divider{ height:1px; background:#e5e7eb; margin:0 16px; }

/* Inputs & generic buttons */
.stTextInput > div > div > input{
  border-radius:10px; border:1px solid #e5e7eb; padding:10px 12px;
}
.stButton>button{ border-radius:10px !important; padding:6px 12px !important; font-weight:600 !important; }
.stButton>button:hover{ filter:brightness(0.98); }
.stButton > button{ margin-left:auto; margin-right:auto; display:block; }

/* --- Action buttons: stay one line, adapt on small screens --- */
.k-actions-row .stButton > button{
  /* never break the word into vertical letters */
  white-space: nowrap !important;
  word-break: normal !important;
  overflow-wrap: normal !important;

  /* allow the label to scale a bit with viewport */
  font-size: clamp(13px, 1.6vw, 16px);

  /* give some room but not so large it forces wrapping */
  min-width: 120px;
  max-width: 100%;

  /* if it STILL can’t fit, show an ellipsis instead of wrapping */
  text-overflow: ellipsis;
  overflow: hidden;

  /* comfy defaults you already had */
  height: 46px;
  padding: 8px 16px;
  font-weight: 600;
}

/* iPad landscape / medium widths */
@media (max-width: 1024px){
  .k-actions-row .stButton > button{
    font-size: clamp(12px, 1.8vw, 15px);
    min-width: 108px;
    padding: 8px 12px;
  }
}

/* iPad portrait / narrow columns */
@media (max-width: 820px){
  .k-actions-row .stButton > button{
    font-size: clamp(12px, 2.0vw, 14px);
    min-width: 96px;
    padding: 7px 10px;
  }
  /* tighten column gutters so buttons get a little more width */
  .k-actions-row [data-testid="column"]{ padding-left:4px; padding-right:4px; }
}

/* Narrow phones/tablets: switch to emoji-only */
@media (max-width: 560px){
  .k-actions-row .stButton > button{
    /* make the button a compact square-ish icon */
    width: 44px;
    min-width: 44px;
    height: 44px;
    padding: 0 0;              /* center the emoji nicely */
    text-align: center;

    /* show the first glyph (emoji), clip the rest (text) */
    overflow: hidden;

    /* bump emoji size a bit for tap target clarity */
    font-size: 20px;
    line-height: 44px;         /* vertically center the emoji */
  }

  /* tighten gutters so three icons fit comfortably */
  .k-actions-row [data-testid="column"]{
    padding-left: 4px; padding-right: 4px;
  }
}

/* Hero */
.k-hero{
  padding:18px 20px; border-radius:16px; background:linear-gradient(90deg,#eef2ff,#e0f2fe);
  border:1px solid #e5e7eb; margin-bottom:12px;
}

/* Sneak Peek styles */
.k-summary-btn .stButton > button{
  background:linear-gradient(90deg,#fef9c3,#fde68a) !important;
  color:#7c2d12 !important; border-radius:16px !important;
  padding:10px 18px !important; border:1px solid #facc15 !important; font-weight:700 !important;
  box-shadow:none !important; transition: transform .15s ease, box-shadow .15s ease;
}
.k-summary-btn .stButton > button:hover{ transform:scale(1.05) rotate(-1deg); box-shadow:0 3px 8px rgba(250,204,21,.3); }
.k-summary-ghost .stButton > button{
  background:transparent !important; color:#0f766e !important;
  border:1px solid #99f6e4 !important; border-radius:12px !important;
  padding:6px 12px !important; font-weight:700 !important; box-shadow:none !important;
}
.k-summary-ghost .stButton > button:hover{ background:#ccfbf1 !important; }

/* Summary spacing + side padding */
.k-summary-full{
  display: block;
  overflow: visible;  
  margin: 2px 0 1px 0;
  padding: 0 10px;
  line-height: 1.5;
}

/* Force light mode */
html{ color-scheme:light !important; }

/* Reduced motion */
@media (prefers-reduced-motion: reduce){
  [data-testid="stVerticalBlock"]:has(> .element-container .k-card-sentinel):hover,
  [data-testid="column"]:has(> .element-container .k-card-sentinel):hover{ transform:none; }
}


/* ===== Sidebar (segmented) ===== */
section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"]{
  display:flex; gap:8px; padding:6px; align-items:center; justify-content:space-between;
}
section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label{
  flex:1 1 0; display:flex; align-items:center; justify-content:center;
  gap:8px; padding:10px 12px; border-radius:999px; cursor:pointer;
  font-weight:600; color:#374151; transition: background .15s ease, color .15s ease, box-shadow .15s ease;
  border:1px solid transparent;
}
section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover{ background:#f1f5f9; }
section[data-testid="stSidebar"] .stRadio input[type="radio"]{ position:absolute; opacity:0; pointer-events:none; }
section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:has(input:checked){
  background:var(--k-accent-bg,#eef2ff); color:#3730a3; border-color:#c7d2fe; box-shadow:0 0 0 2px var(--k-ring) inset;
}
/* Sidebar hero */
section[data-testid="stSidebar"] .k-side-hero{
  margin: 8px 0 14px; padding: 14px 16px;
  border-radius: 14px; background: linear-gradient(90deg,#eef2ff,#e0f2fe);
  border: 1px solid #e5e7eb;
}
section[data-testid="stSidebar"] .k-side-hero h3{ margin: 0 0 4px 0; font-size: 1.05rem; }
section[data-testid="stSidebar"] .k-side-hero p{ margin: 0; color: #475569; font-size: .9rem; }


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

/* Hide radio-dot block in the sidebar menu */
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label[data-baseweb="radio"] > div:first-child,
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label[data-baseweb="radio"] > div:first-child * {
  display: none !important;              /* removes the dot wrapper and its inner circle */
}

/* KidLit logo */
.kidlit-logo{
  font-family:'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  font-size:2.2rem; font-weight:700; letter-spacing:-.5px;
  background:linear-gradient(90deg,#6366f1,#06b6d4);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  display:inline-block; margin:0 auto;
}
.kidlit-sub{ margin:6px 0 0; color:#475569; font-size:1rem; font-weight:500; }

/* Search bar */
.k-searchbox .stTextInput > div > div > input{
  border-radius:10px 0 0 10px !important; border:1px solid #e5e7eb !important; padding:10px 12px !important; height:44px;
}
.k-searchbox .stButton > button{
  border-radius:0 10px 10px 0 !important; border:1px solid #e5e7eb !important; border-left:none !important;
  height:44px; font-weight:600 !important; background:var(--k-accent,#6366f1) !important; color:#fff !important;
}
.k-searchbox .stButton > button:hover{ background:#4f46e5 !important; }

/* Clickable titles look like text */
.book-link{ text-decoration:none !important; color:inherit !important; font-weight:inherit; }
.k-header-title a, .k-header-title a:visited, .k-header-title a:active{
  color:inherit !important; text-decoration:none !important; font-weight:inherit !important;
  padding:2px 4px; border-radius:6px; transition:all .2s ease-in-out;
}
.k-header-title a:hover{
  text-decoration:none !important; font-weight:700 !important; color:inherit !important;
  background-color:rgba(99,102,241,.15); box-shadow:0 0 6px rgba(99,102,241,.25);
}

/* External link buttons under Sneak Peek */
.k-linkbar{ display:flex; justify-content:center; align-items:center; gap:12px; margin:16px 0; flex-wrap:wrap; }
.k-iconbtn{ display:inline-flex; align-items:center; justify-content:center; width:40px; height:40px; border-radius:999px;
  border:1px solid #e5e7eb; background:#fff; box-shadow:0 1px 2px rgba(0,0,0,.06); text-decoration:none; }
.k-iconbtn:hover{ box-shadow:0 2px 6px rgba(0,0,0,.10); transform:translateY(-1px); }
.k-iconbtn:focus-visible{ outline:2px solid #6366f1; outline-offset:2px; border-color:#6366f1; }
.k-ico{ width:20px; height:20px; display:block; }

/* Book-card buttons: keep words intact and on one line */
.k-header-actions .stButton > button,
.k-actions-row .stButton > button,
.k-summary-ghost .stButton > button,
.k-summary-btn .stButton > button {
  white-space: nowrap !important;   /* never wrap within the label */
  word-break: keep-all !important;  /* don't break words */
  overflow-wrap: normal !important; /* no char-level breaks */
  hyphens: none !important;         /* no auto-hyphenation */
  min-width: 140px;                 /* room for “📖 Read” etc. */
  height: 46px;                     /* keep consistent height */
  text-overflow: ellipsis;          /* (optional) trim if ever too narrow */
  overflow: hidden;                 /* pairs with ellipsis */
}

/* ---------- Responsive tweaks (no line clamp reintroduced) ---------- */
@media (max-width: 1024px){
  [data-testid="stSidebar"]{ min-width:220px !important; width:220px !important; }
  .k-cover{ width:78px; height:108px; }
  .k-header-row{ gap:10px; }
  .k-header-title{ font-size:.98rem; }      /* still no clamp */
}
@media (max-width: 820px){
  [data-testid="stSidebar"]{ min-width:200px !important; width:200px !important; }
  .k-cover{ width:70px; height:100px; }
  .k-header-row{ gap:10px; }
  .k-header-title{ font-size:.94rem; }      /* still no clamp */
}
</style>
        """,
        unsafe_allow_html=True,
    )
