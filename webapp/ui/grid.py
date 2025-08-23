import streamlit as st
import pandas as pd
from .cards import render_book_card

def render_book_grid(df: pd.DataFrame, prefix="rec", show_actions=True, cols=3, page_mode=None):
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

