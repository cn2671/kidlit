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
import json
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

# --- Project imports / path setup -------------------------------------------------
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]  
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.core.parse_query import parse_user_query
from scripts.core.recommender import parse_age_span, recommend_books
from scripts.core.text_utils import (
    TITLE_STOPWORDS,
    clean_themes as _clean_themes,
    detect_age_from_text,
    strip_age_category_tokens,
    tokenize_alpha,
)

from scripts.core.lexile_utils import (
    extract_lexile_from_query,
    filter_by_lexile_range,
    format_lexile_display,
    get_lexile_search_suggestions,
    calculate_lexile_progression,
    analyze_lexile_distribution
)

from webapp.ui.css import inject_global_css
from webapp.ui.grid import render_book_grid
from webapp.data_io import load_catalog, build_index, rehydrate_book, dedupe_books
from webapp.data_io import _norm, _norm_title, _book_key, as_dict


# ==============================================================================
# Utilities
# ==============================================================================

def enhanced_parse_user_query(query_text: str) -> dict:
    """Enhanced version that includes Lexile detection"""
    # Start with your existing parser
    parsed = parse_user_query(query_text)
    
    # Add Lexile detection
    lexile_range = extract_lexile_from_query(query_text)
    if lexile_range:
        parsed['lexile_range'] = lexile_range
    
    return parsed

def add_lexile_home_section():
    """Add Lexile search section to Home page"""
    st.markdown("#### Lexile Level Search")
    st.caption("Search by specific reading levels (estimated from age ranges)")
    
    # Get Lexile search suggestions
    lexile_suggestions = get_lexile_search_suggestions()
    
    # Display as buttons in rows of 3
    for i in range(0, len(lexile_suggestions), 3):
        cols = st.columns(3)
        for j, suggestion in enumerate(lexile_suggestions[i:i+3]):
            with cols[j]:
                st.button(
                    suggestion["label"], 
                    key=f"lexile_home_{i}_{j}",
                    on_click=_set_query_and_page, 
                    args=(suggestion["query"],),
                    use_container_width=True
                )
    
    # Custom Lexile range selector
    st.markdown("**Custom Lexile Range:**")
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        lexile_min = st.number_input("Min Lexile", min_value=0, max_value=1500, value=200, step=50)
    with col2:
        lexile_max = st.number_input("Max Lexile", min_value=0, max_value=1500, value=600, step=50)
    with col3:
        if st.button("Search Range", key="custom_lexile_home"):
            query = f"lexile {lexile_min}-{lexile_max}"
            _set_query_and_page(query)

def display_lexile_info_in_card(book_data, card_key=""):
    """Display Lexile information in book card"""
    lexile_score = book_data.get('lexile_score')
    lexile_confidence = book_data.get('lexile_confidence', 0)
    
    if lexile_score and pd.notna(lexile_score):
        lexile_display = format_lexile_display(lexile_score, lexile_confidence)
        st.markdown(f"📊 **Lexile:** {lexile_display}")

def add_lexile_analytics_section(df):
    """Add Lexile analytics section to Analytics page"""
    st.subheader("📊 Lexile Level Analysis")
    
    # Get Lexile analysis
    lexile_analysis = analyze_lexile_distribution(df)
    
    if lexile_analysis["total"] == 0:
        st.info("No Lexile data available")
        return
    
    # Lexile summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    stats = lexile_analysis["stats"]
    
    with col1:
        st.metric("Books with Lexile", f"{lexile_analysis['total']:,}")
    
    with col2:
        st.metric("Coverage", f"{stats['coverage_percent']:.1f}%")
    
    with col3:
        st.metric("Average Lexile", f"{stats['mean_lexile']:.0f}L")
    
    with col4:
        st.metric("Range", f"{stats['min_lexile']}L - {stats['max_lexile']}L")
    
    # Lexile distribution chart
    distribution = lexile_analysis["distribution"]
    
    if distribution:
        fig = px.bar(
            x=list(distribution.keys()),
            y=list(distribution.values()),
            title="Lexile Level Distribution",
            labels={'x': 'Lexile Range', 'y': 'Number of Books'}
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

def create_analytics_dashboard(df):
    """Create analytics dashboard for your book database"""
    st.header("📊 Database Analytics")
    
    if df.empty:
        st.error("No data available for analytics")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Books", len(df))
    
    with col2:
        # Calculate books with summaries
        books_with_summaries = len(df[df.get('summary_gpt', '').fillna('').str.len() > 10])
        st.metric("With Summaries", f"{books_with_summaries:,}")
    
    with col3:
        # Calculate average age range
        age_ranges = df.get('age_range_llm', pd.Series()).fillna('')
        valid_ages = [r for r in age_ranges if r and str(r) != 'nan']
        st.metric("Age Ranges", f"{len(valid_ages):,}")
    
    with col4:
        # Calculate unique themes
        all_themes = set()
        for themes_str in df.get('themes', pd.Series()).fillna(''):
            if themes_str and str(themes_str) != 'nan':
                themes = [t.strip() for t in str(themes_str).split(',') if t.strip()]
                all_themes.update(themes)
        st.metric("Unique Themes", len(all_themes))
    
    st.markdown("---")
    
    # Charts section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📚 Age Range Distribution") 
        
        # Process age ranges for visualization
        age_counts = Counter()
        for age_str in df.get('age_range_llm', pd.Series()).fillna(''):
            if age_str and str(age_str) != 'nan':
                age_clean = str(age_str).replace('–', '-').replace('—', '-')
                age_counts[age_clean] += 1
        
        if age_counts:
            sorted_ages = sorted(age_counts.items(), key=lambda x: _age_sort_key(x[0]))
            ages, counts = zip(*sorted_ages) if sorted_ages else ([], [])
            
            fig = px.pie(
                values=counts,
                names=ages,
                title="Books by Age Range"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No age range data available")
    
    with col2:
        st.subheader("🎨 Popular Themes")
        
        theme_counts = Counter()
        for themes_str in df.get('themes', pd.Series()).fillna(''):
            if themes_str and str(themes_str) != 'nan':
                themes = [t.strip().lower() for t in str(themes_str).split(',') if t.strip()]
                for theme in themes:
                    theme_counts[theme.title()] += 1
        
        if theme_counts:
            top_themes = dict(theme_counts.most_common(10))
            
            fig = px.bar(
                x=list(top_themes.values()),
                y=list(top_themes.keys()),
                orientation='h',
                title="Top 10 Themes",
                labels={'x': 'Number of Books', 'y': 'Theme'}
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No theme data available")

def show_about_page():
    """Show the about page with system information"""
    st.header("ℹ️ About KidLit Curator")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **KidLit Curator** helps parents find perfect books for their children based on age, themes, and emotional tone.
        
        ### 🎯 How It Works
        
        **🎂 Age Appropriateness**
        - Carefully curated reading levels from multiple sources
        - Age ranges based on content complexity and developmental readiness
        - Reading progression support to help children advance
        
        **🏷️ Theme-Based Discovery** 
        - Organized into logical categories: relationships, growth, adventure, emotions, world
        - AI-analyzed themes from book descriptions and reviews
        - Multi-theme search to find books that match specific interests
        
        **😊 Emotional Tone Matching**
        - Light, gentle, engaging, or thoughtful emotional tones
        - Helps match books to child's mood and preferences
        - Consistent tone categorization across the entire database
        
        **📊 Lexile Level Integration**
        - Estimated Lexile scores based on age ranges
        - Precise reading level filtering
        - Educational alignment with school reading programs
        
        **🔍 Smart Search**
        - Natural language queries like "adventure books for 7-year-old"
        - Similar book recommendations based on enjoyed titles
        - Reading level progression suggestions
        """)
    
    with col2:
        # Database stats
        df = CATALOG_DF
        if not df.empty:
            st.markdown("### 📊 Database Stats")
            
            total_books = len(df)
            st.metric("📚 Total Books", f"{total_books:,}")
            
            # Lexile coverage
            books_with_lexile = len(df[df.get('lexile_score', pd.Series()).notna()])
            lexile_coverage = (books_with_lexile / total_books * 100) if total_books > 0 else 0
            st.metric("📊 Lexile Coverage", f"{lexile_coverage:.1f}%")
            
            # Theme coverage
            themes_count = len(df[df.get('themes', '').fillna('').str.len() > 0])
            theme_coverage = (themes_count / total_books * 100) if total_books > 0 else 0
            st.metric("🏷️ Theme Coverage", f"{theme_coverage:.1f}%")
            
            # Summary coverage
            summary_count = len(df[df.get('summary_gpt', '').fillna('').str.len() > 10])
            summary_coverage = (summary_count / total_books * 100) if total_books > 0 else 0
            st.metric("📝 Summary Coverage", f"{summary_coverage:.1f}%")

@st.cache_data
def create_similarity_matrix(df):
    """Create similarity matrix for book recommendations using themes, tone, and age"""
    if df.empty:
        return np.array([]), None
    
    # Prepare features for similarity calculation
    features = []
    
    for idx, row in df.iterrows():
        # Combine themes, tone, and age range into feature vector
        themes = str(row.get('themes', '')).lower()
        tone = str(row.get('tone', '')).lower()
        
        # Try to get age range from multiple possible columns
        age_range = str(row.get('age_range_llm', '') or row.get('age_range', '')).lower()
        
        # Create combined feature string
        combined_features = f"{themes} {tone} {age_range}"
        features.append(combined_features)
    
    # Create TF-IDF vectors
    try:
        vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=1000,
            ngram_range=(1, 2)  # Include both single words and bigrams
        )
        tfidf_matrix = vectorizer.fit_transform(features)
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        return similarity_matrix, vectorizer
        
    except Exception as e:
        st.error(f"Error creating similarity matrix: {e}")
        return np.array([]), None

def get_similar_books(selected_book_data, df, similarity_matrix, n_recommendations=12):
    """Get books similar to the selected book"""
    if similarity_matrix.size == 0:
        return pd.DataFrame()
    
    try:
        # Find the index of the selected book in the dataframe
        selected_title = selected_book_data.get('title', '')
        selected_author = selected_book_data.get('author', '')
        
        # Try to find exact match
        matches = df[
            (df['title'].str.strip() == selected_title.strip()) & 
            (df['author'].str.strip() == selected_author.strip())
        ]
        
        if matches.empty:
            # Fallback: find by title only
            matches = df[df['title'].str.strip() == selected_title.strip()]
        
        if matches.empty:
            return pd.DataFrame()
        
        book_idx = matches.index[0]
        
        # Get similarity scores for this book
        if book_idx >= len(similarity_matrix):
            return pd.DataFrame()
        
        sim_scores = list(enumerate(similarity_matrix[book_idx]))
        
        # Sort by similarity score (excluding the book itself)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top similar books (excluding the original book)
        similar_indices = []
        for idx, score in sim_scores[1:]:  # Skip the first one (itself)
            if len(similar_indices) >= n_recommendations:
                break
            similar_indices.append((idx, score))
        
        # Create dataframe of similar books
        similar_books_data = []
        for idx, similarity_score in similar_indices:
            book_data = df.iloc[idx].to_dict()
            book_data['similarity_score'] = similarity_score
            similar_books_data.append(book_data)
        
        return pd.DataFrame(similar_books_data)
        
    except Exception as e:
        st.error(f"Error finding similar books: {e}")
        return pd.DataFrame()

def get_reading_progression_books(current_book_data, df, progression_type="same"):
    """Get books for reading progression based on current book"""
    
    # Try different age range columns
    current_age_range = (current_book_data.get('age_range_llm', '') or 
                        current_book_data.get('age_range', ''))
    current_themes = str(current_book_data.get('themes', '')).lower()
    current_tone = str(current_book_data.get('tone', '')).lower()
    current_title = current_book_data.get('title', '')
    current_author = current_book_data.get('author', '')
    
    # Parse current age range
    current_ages = parse_age_span(current_age_range)
    if not current_ages:
        return pd.DataFrame()
    
    current_min, current_max = current_ages
    
    # Define progression logic
    if progression_type == "same":
        # Same level: similar age range, prefer similar themes
        target_min, target_max = current_min, current_max
        tolerance = 1  # Allow 1 year variance
        
    elif progression_type == "up":
        # Level up: slightly older age range
        target_min = current_min + 1
        target_max = current_max + 2
        tolerance = 2
        
    elif progression_type == "down":
        # Step back: slightly younger age range
        target_min = max(3, current_min - 2)
        target_max = max(5, current_max - 1)
        tolerance = 2
    
    else:
        return pd.DataFrame()
    
    # Filter books by age range (check both age columns)
    def age_matches(row):
        age_range_str = row.get('age_range_llm', '') or row.get('age_range', '')
        ages = parse_age_span(age_range_str)
        if not ages:
            return False
        book_min, book_max = ages
        
        # Check if book's age range overlaps with target range
        return not (book_max < target_min - tolerance or book_min > target_max + tolerance)
    
    filtered_df = df[df.apply(age_matches, axis=1)].copy()
    
    # Exclude the current book
    filtered_df = filtered_df[
        ~((filtered_df['title'].str.strip() == current_title.strip()) & 
          (filtered_df['author'].str.strip() == current_author.strip()))
    ]
    
    if filtered_df.empty:
        return pd.DataFrame()
    
    # Score books based on theme and tone similarity
    def calculate_similarity_score(row):
        book_themes = str(row.get('themes', '')).lower()
        book_tone = str(row.get('tone', '')).lower()
        
        score = 0
        
        # Theme similarity (basic keyword overlap)
        if current_themes and book_themes:
            current_theme_words = set(current_themes.split())
            book_theme_words = set(book_themes.split())
            theme_overlap = len(current_theme_words & book_theme_words)
            theme_total = len(current_theme_words | book_theme_words)
            if theme_total > 0:
                score += (theme_overlap / theme_total) * 0.6
        
        # Tone similarity
        if current_tone and book_tone and current_tone == book_tone:
            score += 0.4
        
        return score
    
    filtered_df['similarity_score'] = filtered_df.apply(calculate_similarity_score, axis=1)
    
    # Sort by similarity score
    result_df = filtered_df.sort_values('similarity_score', ascending=False)
    
    return result_df.head(12)

def enhanced_book_card(book_data, key_prefix="book", show_similarity=False, show_confidence=False):
    """Enhanced book card with confidence scores, Lexile display, and expandable details"""
    
    title = str(book_data.get('title', 'Unknown Title')).strip()
    author = str(book_data.get('author', 'Unknown Author')).strip()
    
    # Create container for the book card
    container = st.container()
    
    with container:
        # Book cover and basic info
        col1, col2 = st.columns([1, 2])
        
        with col1:
            cover_url = book_data.get('cover_url', '')
            if cover_url and str(cover_url) != 'nan':
                st.image(cover_url, width=100)
            else:
                st.markdown("📖")
        
        with col2:
            st.markdown(f"**{title}**")
            st.markdown(f"*by {author}*")
            
            # Age range (try multiple columns)
            age_range = book_data.get('age_range_llm', '') or book_data.get('age_range', '')
            if age_range and str(age_range) != 'nan':
                st.markdown(f"🎂 Ages {age_range}")
            
            # Lexile score (NEW)
            display_lexile_info_in_card(book_data, key_prefix)
            
            # Themes (limit to first 3)
            themes = str(book_data.get('themes', '')).strip()
            if themes and themes != 'nan':
                theme_list = [t.strip() for t in themes.split(',')[:3] if t.strip()]
                if theme_list:
                    themes_display = ', '.join([t.title() for t in theme_list])
                    st.markdown(f"🏷️ {themes_display}")
            
            # Tone
            tone = str(book_data.get('tone', '')).strip()
            if tone and tone != 'nan':
                st.markdown(f"😊 {tone.title()}")
            
            # Show similarity score if available
            if show_similarity and 'similarity_score' in book_data:
                similarity_pct = int(book_data['similarity_score'] * 100)
                st.markdown(f"🎯 {similarity_pct}% match")
            
            # Show confidence score if requested
            if show_confidence:
                confidence = book_data.get('confidence', 0)
                if confidence > 0:
                    confidence_pct = int(float(confidence) * 100)
                    st.markdown(f"📊 {confidence_pct}% confidence")
        
        # Expandable details section
        expand_key = f"{key_prefix}_expand_{hash(f'{title}_{author}')}"
        
        if st.button("📖 View Details", key=f"{expand_key}_btn"):
            st.session_state[expand_key] = not st.session_state.get(expand_key, False)
        
        # Show expanded details if toggled
        if st.session_state.get(expand_key, False):
            st.markdown("---")
            
            # Summary
            summary = book_data.get('summary_gpt', '') or book_data.get('description', '')
            if summary and str(summary) != 'nan':
                st.markdown("**Summary:**")
                st.markdown(summary[:300] + "..." if len(str(summary)) > 300 else str(summary))
            
            # Links
            col1, col2 = st.columns(2)
            
            with col1:
                goodreads_url = book_data.get('goodreads_url', '')
                if goodreads_url and str(goodreads_url) != 'nan':
                    st.markdown(f"[📚 Goodreads]({goodreads_url})")
            
            with col2:
                openlibrary_url = book_data.get('openlibrary_url', '')
                if openlibrary_url and str(openlibrary_url) != 'nan':
                    st.markdown(f"[📖 Open Library]({openlibrary_url})")
            
            # Action buttons (if not already in favorites/read/skipped)
            col1, col2, col3 = st.columns(3)
            
            book_key = _book_key(book_data)
            
            with col1:
                if book_key not in [_book_key(b) for b in st.session_state.liked_books]:
                    if st.button("❤️ Like", key=f"{expand_key}_like"):
                        st.session_state.liked_books.append(book_data)
                        st.success("Added to favorites!")
                        st.rerun()
            
            with col2:
                if book_key not in [_book_key(b) for b in st.session_state.read_books]:
                    if st.button("📖 Mark Read", key=f"{expand_key}_read"):
                        st.session_state.read_books.append(book_data)
                        st.success("Marked as read!")
                        st.rerun()
            
            with col3:
                if book_key not in [_book_key(b) for b in st.session_state.skipped_books]:
                    if st.button("🚫 Skip", key=f"{expand_key}_skip"):
                        st.session_state.skipped_books.append(book_data)
                        st.success("Book skipped!")
                        st.rerun()

def render_enhanced_book_grid(df, key_prefix="grid", show_similarity=False, show_confidence=False, cols=3):
    """Render books in grid format with enhanced cards"""
    
    if df.empty:
        st.info("No books to display")
        return
    
    # Pagination
    books_per_page = 9
    total_books = len(df)
    total_pages = (total_books - 1) // books_per_page + 1
    
    if total_pages > 1:
        page_num = st.number_input(
            f"Page (1-{total_pages})", 
            min_value=1, 
            max_value=total_pages, 
            value=1,
            key=f"{key_prefix}_page"
        ) - 1
        
        start_idx = page_num * books_per_page
        end_idx = start_idx + books_per_page
        page_books = df.iloc[start_idx:end_idx]
    else:
        page_books = df
    
    # Display books in grid
    for i in range(0, len(page_books), cols):
        cols_list = st.columns(cols)
        
        for j in range(cols):
            if i + j < len(page_books):
                book_data = page_books.iloc[i + j].to_dict()
                
                with cols_list[j]:
                    enhanced_book_card(
                        book_data, 
                        key_prefix=f"{key_prefix}_{i}_{j}",
                        show_similarity=show_similarity,
                        show_confidence=show_confidence
                    )

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

st.set_page_config(page_title="KidLit Curator", page_icon="📚", layout="wide")
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
CATALOG_DF = load_catalog("data/books_final.csv")
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
    ("🎯 Similar Books", "Similar"),           
    ("📈 Reading Progress", "Progress"),       
    ("❤️ Favorites", "Favorites"),
    ("🚫 Skipped", "Skipped"),
    ("📖 Read", "Read"),
    ("📊 Analytics", "Analytics"),
    ("ℹ️ About", "About"),
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
    """<div class="k-hero">
    <h1 class="kidlit-logo">KidLit Curator</h1>
    <p class="kidlit-sub">Kid's Literature: Personalized children's books by age, themes, and tone.</p>
    </div>""",
    unsafe_allow_html=True,
)

if page == "Home":
    st.title("👋 Welcome to KidLit")
    st.markdown("""
**Find the perfect children's book** by age, theme, tone, and Lexile level.
Try a few examples:
- "book about **friendship** for a **5 year old**"
- "**whimsical** bedtime story for **3–5**"
- "**lexile 400-600** books about **adventure**"
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
        st.button("lexile 400-600 • adventure",
                  on_click=_set_query_and_page,
                  args=("lexile 400-600 adventure books",))

    
    st.markdown("---")
    st.subheader("Browse by Theme, Age, Tone, and Lexile")
    st.caption("Search with any combination of themes, tones, ages, or Lexile levels.")

    # --- Build frequency tables from catalog  ---
    theme_counts = Counter()
    for lst in CATALOG_DF.get("themes_norm_list", []):
        for t in (lst or []):
            theme_counts[_norm(t)] += 1

    tone_counts = Counter()
    for lst in CATALOG_DF.get("tones_norm_list", []):
        for t in (lst or []):
            tone_counts[_norm(t)] += 1
    
    # --- Age counts (try both age columns) ---
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
    # Check both possible age columns
    age_columns = ['age_range_llm', 'age_range']
    for col in age_columns:
        if col in CATALOG_DF.columns:
            for s in CATALOG_DF[col].fillna(''):
                a = _norm_age_str(s)
                if a:
                    age_counts[a] += 1

    # ------- Popular chips -------
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
    
    st.markdown("#### Age Ranges")
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

    # Add Lexile search section
    add_lexile_home_section()

    # -------  Build your own search -------
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
                placeholder="e.g., 5-year-old • friendship • lexile 300-500",
                label_visibility="collapsed",
                key="user_query",
            )
        with btn_col:
            submitted = st.form_submit_button("🔍 Search", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    submitted = submitted or st.session_state.pop("do_search", False)

    if submitted:
        st.session_state.do_search = False
        st.session_state.generated = True
        st.session_state.expanded_cards = set()  
        st.session_state.page_num_recs = 1  

        # Use enhanced parsing with Lexile detection
        parsed_raw = enhanced_parse_user_query(st.session_state.user_query)
        parsed = sanitize_parsed(st.session_state.user_query, parsed_raw)

        themes_list = _clean_themes(parsed.get("themes", []))
        tone_word = parsed.get("tone", "")
        age_range = parsed.get("age_range", "")
        lexile_range = parsed.get("lexile_range")  # NEW: Lexile filtering

        st.session_state.filters = {
            "age_range": age_range,
            "tone": tone_word,
            "themes": themes_list,
            "lexile_range": lexile_range,  # Store Lexile filter
        }

        q_raw = (st.session_state.user_query or "").strip()
        q_norm = _norm_title(q_raw)

        title_hits = pd.DataFrame()
        if q_norm:
            df = CATALOG_DF.copy()
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

            if title_hits.empty and tokens:  
                mask_author = True
                for t in tokens:
                    mask_author = mask_author & (
                        df["author_norm"].str.contains(t) | df["ol_author_norm"].str.contains(t)
                    )
                author_hits = df[mask_author]
                if not author_hits.empty:
                    title_hits = author_hits

        # Get base recommendations
        if age_range:
            recs = recommend_books(age_range, themes_list, tone_word, n=None)
            if isinstance(recs, pd.DataFrame) and recs.empty:
                recs = recommend_books(None, themes_list, tone_word, n=None)
        else:
            recs = recommend_books(None, themes_list, tone_word, n=None)

        # Combine with title hits if needed
        if not age_range and not title_hits.empty:
            combined = pd.concat([title_hits, recs], ignore_index=True)
            combined["kidlit_key"] = combined.apply(lambda r: _book_key(as_dict(r)), axis=1)
            recs = combined.drop_duplicates(subset=["kidlit_key"]).drop(columns=["kidlit_key"])
        if not isinstance(recs, pd.DataFrame):
            recs = pd.DataFrame(recs)

        # Ensure columns exist
        for col in ["summary_gpt", "themes", "tone", "age_range", "cover_url", "author", "title", 
                   "ol_title", "ol_author", "goodreads_url", "openlibrary_url", "description", "summary"]:
            if col not in recs.columns:
                recs[col] = ""

        recs = recs.fillna("")

        def _empty(s) -> bool:
            return str(s or "").strip().lower() in ("", "nan", "none", "null")

        recs = recs[~(recs["title"].apply(_empty) & recs["author"].apply(_empty))]

        def _is_placeholder_title(s: str) -> bool:
            t = _norm_title(s)
            return t in {"no matches", "no match", "no results", "no result", "⚠️ no matches", "!no matches"}

        recs = recs[~recs["title"].apply(_is_placeholder_title)]
        recs["kidlit_key"] = recs.apply(lambda r: _book_key(as_dict(r)), axis=1)
        recs = recs.drop_duplicates(subset=["kidlit_key"]).drop(columns=["kidlit_key"])

        # Apply Lexile filtering if specified
        if lexile_range and not recs.empty:
            original_count = len(recs)
            recs = filter_by_lexile_range(recs, lexile_range)
            
            if len(recs) < original_count:
                st.info(f"🎯 Filtered to Lexile range {lexile_range[0]}L - {lexile_range[1]}L ({len(recs)} books found)")
            
            if recs.empty:
                st.warning(f"No books found in Lexile range {lexile_range[0]}L - {lexile_range[1]}L. Try a broader range.")

        # Tone filtering (your existing logic)
        tone_raw = (parsed.get("tone") or "").strip().lower()
        if tone_raw:
            recs["tones_norm_list"] = recs["tone"].apply(_tones_list)
            strict = recs[recs["tones_norm_list"].apply(lambda lst: tone_raw in (lst or []))]

            MIN_STRICT = 6
            if len(strict) >= MIN_STRICT:
                recs = strict
            else:
                try:
                    from scripts.core.llm_filters import score_rows_by_tone
                    MAX_TO_SCORE = 150
                    HARD_TH = 3
                    SOFT_TH = 2
                    MIN_KEEP = 6

                    scored = score_rows_by_tone(recs.head(MAX_TO_SCORE).copy(), tone_raw, tone_col="tone")
                    hard = scored[scored["__tone_score"] >= HARD_TH]
                    soft = scored[scored["__tone_score"] >= SOFT_TH]

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
                        recs = strict if not strict.empty else recs
                        if strict.empty:
                            st.caption("No tone matches found; showing age/theme matches only.")
                except Exception:
                    recs = strict if not strict.empty else recs

        # Age filtering (your existing logic)
        if (age_range and re.fullmatch(r"\d{1,2}", str(age_range).strip()) and not recs.empty):
            target = int(age_range)
            # Check both possible age columns
            age_col = 'age_range_llm' if 'age_range_llm' in recs.columns else 'age_range'
            recs = recs[
                recs[age_col].apply(
                    lambda v: (lambda sp: (sp is not None) and (sp[0] <= target <= sp[1]))(parse_age_span(v))
                )
            ]
        
        st.session_state.recs_df = recs

    # Render results
    if st.session_state.get("generated") and st.session_state.get("recs_df") is not None:
        if "filters" in st.session_state and st.session_state.filters:
            parsed = st.session_state.filters
        else:
            parsed = sanitize_parsed(
                st.session_state.user_query, enhanced_parse_user_query(st.session_state.user_query)
            )

        themes_list = _clean_themes(parsed.get("themes", []))
        themes_display = ", ".join(themes_list) if themes_list else "—"
        tone_display = parsed.get("tone") or "—"
        age_display = parsed.get("age_range") or "—"
        lexile_display = f"{parsed['lexile_range'][0]}L-{parsed['lexile_range'][1]}L" if parsed.get('lexile_range') else "—"

        st.markdown(f"**Filters:** Age {age_display}, Tone {tone_display}, Themes {themes_display}, Lexile {lexile_display}")

        recs = st.session_state.recs_df.copy()
        skip_keys = {_book_key(as_dict(b)) for b in st.session_state.skipped_books}
        if skip_keys and not recs.empty:
            recs = recs[~recs.apply(lambda r: _book_key(as_dict(r)) in skip_keys, axis=1)]

        if recs.empty:
            st.warning("No matches. Try different filters.")
        else:
            total = len(recs)
            page_size = int(st.session_state.get("page_size_recs", 9))
            max_page = max(1, math.ceil(total / page_size))
            page_num = int(st.session_state.get("page_num_recs", 1))
            page_num = max(1, min(page_num, max_page))
            st.session_state.page_num_recs = page_num

            start = (page_num - 1) * page_size
            end = start + page_size
            recs_page = recs.iloc[start:end]

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
                    "Results per page",
                    size_options,
                    index=idx,
                    key="page_size_recs",
                    on_change=_reset_pagination,
                    help="How many books to show on this page",
                )

            render_book_grid(recs_page, prefix="rec", show_actions=True, cols=3)

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

elif page == "Similar":
    st.title("🎯 Find Similar Books")
    st.markdown("Select a book your child enjoyed to find similar recommendations!")
    
    similarity_matrix, vectorizer = create_similarity_matrix(CATALOG_DF)
    
    if similarity_matrix.size == 0:
        st.error("Unable to create similarity matrix. Please check your data.")
        st.stop()
    
    if not CATALOG_DF.empty:
        book_options = []
        for idx, row in CATALOG_DF.iterrows():
            title = str(row.get('title', 'Unknown')).strip()
            author = str(row.get('author', 'Unknown')).strip()
            book_options.append(f"{title} by {author}")
        
        selected_book_str = st.selectbox(
            "Select a book your child enjoyed:",
            [""] + sorted(book_options),
            key="similar_book_select"
        )
        
        if selected_book_str:
            if " by " in selected_book_str:
                title, author = selected_book_str.rsplit(" by ", 1)
                
                selected_book_df = CATALOG_DF[
                    (CATALOG_DF['title'].str.strip() == title.strip()) & 
                    (CATALOG_DF['author'].str.strip() == author.strip())
                ]
                
                if not selected_book_df.empty:
                    selected_book_data = selected_book_df.iloc[0].to_dict()
                    
                    st.subheader(f"📖 Selected Book: {title}")
                    enhanced_book_card(selected_book_data, "selected", show_confidence=True)
                    st.markdown("---")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        filter_by_age = st.checkbox("Filter by similar age range", value=True)
                    with col2:
                        num_recommendations = st.slider("Number of recommendations", 6, 24, 12)
                    
                    with st.spinner("Finding similar books..."):
                        similar_books_df = get_similar_books(
                            selected_book_data, CATALOG_DF, similarity_matrix, num_recommendations
                        )
                    
                    if filter_by_age and not similar_books_df.empty:
                        selected_age_range = selected_book_data.get('age_range_llm', '') or selected_book_data.get('age_range', '')
                        if selected_age_range:
                            selected_ages = parse_age_span(selected_age_range)
                            if selected_ages:
                                def age_compatible(row):
                                    book_age_range = row.get('age_range_llm', '') or row.get('age_range', '')
                                    book_ages = parse_age_span(book_age_range)
                                    if not book_ages:
                                        return True  # Include if age unknown
                                    
                                    # Check for overlap
                                    return not (book_ages[1] < selected_ages[0] - 2 or 
                                              book_ages[0] > selected_ages[1] + 2)
                                
                                similar_books_df = similar_books_df[
                                    similar_books_df.apply(age_compatible, axis=1)
                                ]
                    
                    if not similar_books_df.empty:
                        st.subheader(f"📚 Similar Books ({len(similar_books_df)} found)")
                        render_enhanced_book_grid(
                            similar_books_df, 
                            "similar", 
                            show_similarity=True
                        )
                    else:
                        st.warning("No similar books found. Try adjusting your filters or selecting a different book.")
                else:
                    st.error("Selected book not found in database.")

elif page == "Progress":
    st.title("📈 Reading Progression")
    st.markdown("Help your child advance their reading skills with level-appropriate book suggestions!")
    
    if not CATALOG_DF.empty:
        book_options = []
        for idx, row in CATALOG_DF.iterrows():
            title = str(row.get('title', 'Unknown')).strip()
            author = str(row.get('author', 'Unknown')).strip()
            age_range = str(row.get('age_range_llm', '') or row.get('age_range', '')).strip()
            display_text = f"{title} by {author}"
            if age_range and age_range != 'nan':
                display_text += f" (Ages {age_range})"
            book_options.append(display_text)
        
        current_book_str = st.selectbox(
            "What book did your child recently finish or enjoy?",
            [""] + sorted(book_options),
            key="progress_book_select"
        )
        
        if current_book_str:
            if " by " in current_book_str:
                parts = current_book_str.split(" by ")
                title = parts[0].strip()
                author_and_age = parts[1]
                
                if " (Ages " in author_and_age:
                    author = author_and_age.split(" (Ages ")[0].strip()
                else:
                    author = author_and_age.strip()
                
                current_book_df = CATALOG_DF[
                    (CATALOG_DF['title'].str.strip() == title) & 
                    (CATALOG_DF['author'].str.strip() == author)
                ]
                
                if not current_book_df.empty:
                    current_book_data = current_book_df.iloc[0].to_dict()
                    
                    st.subheader(f"📖 Current Book: {title}")
                    enhanced_book_card(current_book_data, "current", show_confidence=True)
                    st.markdown("---")
                    
                    col1, col2, col3 = st.columns(3)
                    progression_results = {}
                    
                    with col1:
                        st.subheader("📚 Same Level")
                        st.markdown("*More practice at current level*")
                        
                        if st.button("Find Same Level Books", key="same_level"):
                            with st.spinner("Finding same level books..."):
                                progression_results["same"] = get_reading_progression_books(
                                    current_book_data, CATALOG_DF, "same"
                                )
                    
                    with col2:
                        st.subheader("📈 Level Up")
                        st.markdown("*Challenge with harder books*")
                        
                        if st.button("Find Next Level Books", key="level_up"):
                            with st.spinner("Finding next level books..."):
                                progression_results["up"] = get_reading_progression_books(
                                    current_book_data, CATALOG_DF, "up"
                                )
                    
                    with col3:
                        st.subheader("📉 Step Back")
                        st.markdown("*Easier books for confidence*")
                        
                        if st.button("Find Easier Books", key="step_back"):
                            with st.spinner("Finding easier books..."):
                                progression_results["down"] = get_reading_progression_books(
                                    current_book_data, CATALOG_DF, "down"
                                )
                    
                    for progression_type, results_df in progression_results.items():
                        if not results_df.empty:
                            st.markdown("---")
                            
                            if progression_type == "same":
                                st.subheader("📚 Same Level Recommendations")
                            elif progression_type == "up":
                                st.subheader("📈 Level Up Recommendations")
                            elif progression_type == "down":
                                st.subheader("📉 Easier Books Recommendations")
                            
                            render_enhanced_book_grid(
                                results_df, 
                                f"progress_{progression_type}",
                                show_confidence=True
                            )
                        elif progression_type in progression_results:
                            st.warning(f"No books found for {progression_type} progression. Try a different book or progression type.")

elif page == "Analytics":
    create_analytics_dashboard(CATALOG_DF)
    
    # Add Lexile analytics section
    add_lexile_analytics_section(CATALOG_DF)

elif page == "About":
    show_about_page()