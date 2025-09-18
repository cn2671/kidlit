#!/usr/bin/env python3
"""
Enhanced Flask backend integrating your existing recommendation engine 
with trained ML models for hybrid book recommendations
"""

import os
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json
import re
from datetime import datetime
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import your existing modules
try:
    from scripts.core.parse_query import parse_user_query
    from scripts.core.recommender import recommend_books, parse_age_span
    from scripts.core.text_utils import split_tags
    from webapp.data_io import load_catalog, build_index
    from hybrid_query_parser import hybrid_parse_query

except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    # Fallback implementations will be provided

# Import enhanced Lexile predictor with enriched scores
try:
    # First try the new enhanced predictor
    from scripts.core.enriched_predictor import EnrichedLexilePredictor
    print("✅ Enhanced Lexile predictor with enriched scores imported successfully")
    LEXILE_PREDICTOR_AVAILABLE = True
    predictor_class = EnrichedLexilePredictor
except ImportError:
    try:
        # Fallback to production predictor
        from scripts.production.production_lexile_predictor import ProductionLexilePredictor
        print("✅ Production Lexile predictor imported from scripts")
        LEXILE_PREDICTOR_AVAILABLE = True
        predictor_class = ProductionLexilePredictor
    except ImportError:
        try:
            # Last fallback to local import
            from production_lexile_predictor import ProductionLexilePredictor
            print("✅ Production Lexile predictor imported successfully")
            LEXILE_PREDICTOR_AVAILABLE = True
            predictor_class = ProductionLexilePredictor
        except ImportError as e:
            print(f"⚠️  Could not import any Lexile predictor: {e}")
            LEXILE_PREDICTOR_AVAILABLE = False
            predictor_class = None

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Theme and Tone Group Mappings
THEME_GROUPS = {
    'Relationships & Social Skills': [
        'friendship', 'family', 'love', 'community', 'acceptance', 'belonging', 
        'kindness', 'teamwork', 'cooperation', 'sharing', 'communication', 'empathy', 
        'companionship', 'sibling relationships', 'sibling bond', 'relationships', 'bonding'
    ],
    'Personal Growth & Character Development': [
        'courage', 'bravery', 'resilience', 'identity', 'self-discovery', 'individuality', 
        'growth', 'responsibility', 'independence', 'determination', 'perseverance', 
        'self-acceptance', 'self-confidence', 'leadership', 'coming of age', 'growing up', 
        'self-worth', 'confidence', 'personal growth', 'inner strength'
    ],
    'Adventure & Exploration': [
        'adventure', 'mystery', 'curiosity', 'exploration', 'discovery', 'journey', 
        'survival', 'rescue', 'investigation', 'searching', 'quest', 'travel', 'detective work'
    ],
    'Fantasy & Imagination': [
        'magic', 'imagination', 'creativity', 'fantasy', 'supernatural', 'fairy tales', 
        'mythology', 'time travel', 'prophecy', 'witchcraft', 'mystical', 'fairy tale', 'destiny'
    ],
    'Learning & School Life': [
        'problem-solving', 'learning', 'education', 'school', 'school life', 'science', 
        'history', 'culture', 'language', 'alphabet', 'counting', 'numbers', 'math', 'knowledge'
    ],
    'Emotions & Life Experiences': [
        'humor', 'emotions', 'fear', 'loss', 'grief', 'joy', 'hope', 'dreams', 'change', 
        'challenges', 'illness', 'war', 'bullying', 'anxiety', 'happiness', 'sadness', 'anger'
    ],
    'Animals, Nature & Daily Life': [
        'nature', 'animals', 'pets', 'home', 'play', 'routine', 'bedtime', 'daily life', 
        'food', 'seasons', 'weather', 'environment', 'farm life', 'caring for others'
    ]
}

TONE_GROUPS = {
    'Light & Fun': [
        'whimsical', 'humorous', 'playful', 'lighthearted', 'light-hearted', 'funny', 
        'silly', 'fun', 'entertaining', 'lively', 'upbeat'
    ],
    'Warm & Inspiring': [
        'heartwarming', 'uplifting', 'hopeful', 'inspirational', 'empowering', 'encouraging', 
        'wholesome', 'sweet', 'heartfelt', 'warm', 'comforting'
    ],
    'Exciting & Dynamic': [
        'adventurous', 'mysterious', 'exciting', 'thrilling', 'suspenseful', 'intense', 
        'action-packed', 'epic', 'intriguing', 'dramatic'
    ],
    'Gentle & Calm': [
        'gentle', 'calm', 'peaceful', 'soothing', 'tender', 'cozy', 'reassuring', 'soft', 'quiet'
    ],
    'Thoughtful & Emotional': [
        'emotional', 'nostalgic', 'introspective', 'reflective', 'thoughtful', 'contemplative', 
        'poignant', 'touching', 'meaningful', 'bittersweet'
    ],
    'Educational & Realistic': [
        'educational', 'informative', 'realistic', 'historical', 'serious', 'cautionary', 
        'observant', 'practical'
    ]
}

def expand_theme_groups(themes_input):
    """Expand grouped themes back to individual themes for searching"""
    if not themes_input:
        return []
    
    expanded_themes = []
    for theme in themes_input:
        theme = theme.strip()
        # Check if it's a group name
        if theme in THEME_GROUPS:
            expanded_themes.extend(THEME_GROUPS[theme])
        else:
            # It's an individual theme
            expanded_themes.append(theme)
    
    return list(set(expanded_themes))  # Remove duplicates

def expand_tone_groups(tones_input):
    """Expand grouped tones back to individual tones for searching"""
    if not tones_input:
        return []
    
    expanded_tones = []
    for tone in tones_input:
        tone = tone.strip()
        # Check if it's a group name
        if tone in TONE_GROUPS:
            expanded_tones.extend(TONE_GROUPS[tone])
        else:
            # It's an individual tone
            expanded_tones.append(tone)
    
    return list(set(expanded_tones))  # Remove duplicates

@app.route('/api/theme-tone-groups', methods=['GET'])
def get_theme_tone_groups():
    """Get available theme and tone groups for frontend"""
    return jsonify({
        'success': True,
        'theme_groups': list(THEME_GROUPS.keys()),
        'tone_groups': list(TONE_GROUPS.keys())
    })

@app.route('/api/save-book', methods=['POST'])
def save_user_book():
    """Save book to user's list"""
    try:
        data = request.get_json()
        book_data = data.get('book')
        list_type = data.get('list_type')
        
        if not book_data or not list_type:
            return jsonify({'success': False, 'error': 'Missing data'}), 400
        
        if list_type not in ['favorites', 'skipped', 'read']:
            return jsonify({'success': False, 'error': 'Invalid list type'}), 400
        
        # Remove from other lists
        for other_list in ['favorites', 'skipped', 'read']:
            if other_list != list_type:
                user_books[other_list] = [b for b in user_books[other_list] if b['title'] != book_data['title']]
        
        # Add to specified list (avoid duplicates)
        existing = [b for b in user_books[list_type] if b['title'] == book_data['title']]
        if not existing:
            user_books[list_type].append(book_data)
        
        return jsonify({'success': True, 'message': f'Book added to {list_type}'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/get-user-books/<list_type>', methods=['GET'])
def get_user_book_list(list_type):
    """Get user's books from a specific list"""
    try:
        if list_type not in ['favorites', 'skipped', 'read']:
            return jsonify({'success': False, 'error': 'Invalid list type'}), 400
        
        books = user_books.get(list_type, [])
        return jsonify({'success': True, 'books': books, 'total': len(books)})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/remove-book', methods=['POST'])
def remove_user_book():
    """Remove book from user's list"""
    try:
        data = request.get_json()
        title = data.get('title')
        list_type = data.get('list_type')
        
        if not title or not list_type:
            return jsonify({'success': False, 'error': 'Missing data'}), 400
        
        user_books[list_type] = [b for b in user_books[list_type] if b['title'] != title]
        return jsonify({'success': True, 'message': f'Book removed from {list_type}'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
        
class HybridRecommendationEngine:
    """
    Combines your existing rule-based recommender with trained ML models
    for enhanced book recommendations and reading level predictions
    """
    
    def __init__(self, model_timestamp="20250831_182131"):
        self.model_timestamp = model_timestamp
        self.ml_models = {}
        self.metadata = {}
        self.feature_columns = []
        self.label_encoders = {}
        self.author_stats = {}
        self.catalog_df = None
        self.catalog_index = {}
        self.similarity_matrix = None
        self.tfidf_vectorizer = None
        
        # Load everything
        self.load_ml_models()
        self.load_catalog()
        self.prepare_similarity_system()


        print("Step 1: Loading ML models...")
        self.load_ml_models()
        print("Step 2: Loading catalog...")
        self.load_catalog()
        print("Step 3: Loading author stats...")
        self.load_author_stats()
        print(f"Author stats loaded: {len(self.author_stats)} authors")
        print("Step 4: Preparing similarity system...")
        self.prepare_similarity_system()
        
    def load_ml_models(self):
        """Load your trained ML models"""
        try:
            model_dir = Path("data/models")
            
            # Load ensemble model
            ensemble_path = model_dir / f"ensemble_{self.model_timestamp}.joblib"
            if ensemble_path.exists():
                self.ml_models = joblib.load(ensemble_path)
                logger.info("✓ ML models loaded successfully!")
            
            # Load metadata
            metadata_path = model_dir / f"model_metadata_{self.model_timestamp}.joblib"
            if metadata_path.exists():
                self.metadata = joblib.load(metadata_path)
                self.feature_columns = self.metadata.get('feature_columns', [])
                self.label_encoders = self.metadata.get('label_encoders', {})
                logger.info("✓ Model metadata loaded")
            
            # Load author statistics for ML features
            self.load_author_stats()
            
        except Exception as e:
            logger.warning(f"Could not load ML models: {e}")
            self.ml_models = {}
    
    def load_author_stats(self):
        """Load comprehensive author statistics from your data"""
        try:
            if self.catalog_df is None or self.catalog_df.empty:
                logger.warning("No catalog data available for author stats")
                return
            
            # Use author_clean if available, otherwise author
            author_col = 'author_clean' if 'author_clean' in self.catalog_df.columns else 'author'
            
            # Filter out empty authors
            valid_authors_df = self.catalog_df[
                self.catalog_df[author_col].notna() & 
                (self.catalog_df[author_col].str.strip() != '')
            ].copy()
            
            if valid_authors_df.empty:
                logger.warning("No valid authors found")
                return
            
            logger.info(f"Processing {len(valid_authors_df)} books from {valid_authors_df[author_col].nunique()} authors")
            
            # Calculate author stats
            for author_name, group in valid_authors_df.groupby(author_col):
                if not author_name or str(author_name).strip() == '':
                    continue
                    
                # Lexile analysis
                lexile_scores = pd.to_numeric(group['lexile_score'], errors='coerce').dropna()
                
                # Age category analysis
                age_ranges = group['age_range_llm'].dropna()
                age_categories = []
                for age_range in age_ranges:
                    age_str = str(age_range).lower()
                    if '3-5' in age_str or '0-3' in age_str:
                        age_categories.append('Early')
                    elif '6-8' in age_str:
                        age_categories.append('Beginning')  
                    elif '9-12' in age_str:
                        age_categories.append('Intermediate')
                    elif '13+' in age_str:
                        age_categories.append('Advanced')
                
                # Store stats
                self.author_stats[author_name] = {
                    'book_count': len(group),
                    'avg_lexile': float(lexile_scores.mean()) if len(lexile_scores) > 0 else None,
                    'std_lexile': float(lexile_scores.std()) if len(lexile_scores) > 1 else 0,
                    'primary_age_category': max(set(age_categories), key=age_categories.count) if age_categories else 'Unknown',
                    'themes': list(set([theme.strip() for themes_str in group['themes'].dropna() 
                                for theme in str(themes_str).split(',') if theme.strip()])),
                    'avg_confidence': float(group.get('reading_confidence_llm', pd.Series([0.7])).mean())
                }
            
            logger.info(f"✓ Loaded statistics for {len(self.author_stats)} authors")
            
            # Log some sample stats
            sample_authors = list(self.author_stats.keys())[:3]
            for author in sample_authors:
                stats = self.author_stats[author]
                logger.info(f"  {author}: {stats['book_count']} books, avg_lexile={stats['avg_lexile']}")
                    
        except Exception as e:
            logger.error(f"Error loading author stats: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def load_catalog(self):
        """Load your book catalog using existing data_io functions and enhance with lexile scores"""
        try:
            # Try to use your existing data loading with correct path
            data_path = ROOT / "data" / "raw" / "books_final_complete.csv"
            self.catalog_df = load_catalog(str(data_path))
            self.catalog_index = build_index(self.catalog_df)
            logger.info(f"✓ Loaded catalog with {len(self.catalog_df)} books")
            
        except Exception as e:
            logger.warning(f"Could not load catalog with existing functions: {e}")
            # Fallback: direct pandas loading
            try:
                data_path = ROOT / "data" / "raw" / "books_final_complete.csv"
                self.catalog_df = pd.read_csv(data_path).fillna("")
                logger.info(f"✓ Loaded catalog directly with {len(self.catalog_df)} books")
            except Exception as e2:
                logger.error(f"Could not load catalog at all: {e2}")
                self.catalog_df = pd.DataFrame()
        
        # Enhance catalog with lexile scores using EnrichedLexilePredictor
        if not self.catalog_df.empty:
            self._enhance_catalog_with_lexile_scores()
    
    def _enhance_catalog_with_lexile_scores(self):
        """Enhance catalog with lexile_score column using EnrichedLexilePredictor"""
        try:
            logger.info("🔍 Enhancing catalog with lexile scores...")
            
            # Initialize the EnrichedLexilePredictor 
            if LEXILE_PREDICTOR_AVAILABLE:
                predictor = predictor_class()
                
                # Keep track of existing scores vs enhanced ones
                enhanced_count = 0
                existing_count = 0
                
                # Get or create enhanced lexile scores column
                enhanced_lexile_scores = []
                
                for _, row in self.catalog_df.iterrows():
                    title = row.get('title', '')
                    author = row.get('author', '')
                    existing_score = row.get('lexile_score')
                    
                    # Get enriched prediction
                    prediction = predictor.predict(title, author)
                    enriched_score = prediction.get('lexile_score', None)
                    
                    # Prefer enriched score if available, otherwise use existing
                    if enriched_score is not None and prediction.get('source') == 'enriched':
                        enhanced_lexile_scores.append(enriched_score)
                        enhanced_count += 1
                    elif existing_score is not None:
                        enhanced_lexile_scores.append(existing_score)
                        existing_count += 1
                    else:
                        # Fall back to ML or default
                        enhanced_lexile_scores.append(enriched_score)
                        if enriched_score is not None:
                            enhanced_count += 1
                
                # Update lexile_score column with enhanced values
                self.catalog_df['lexile_score'] = enhanced_lexile_scores
                
                total_books = len(self.catalog_df)
                logger.info(f"✅ Enhanced catalog: {enhanced_count} enriched, {existing_count} existing, {total_books} total")
                
            else:
                logger.warning("⚠️ No lexile predictor available - keeping existing scores")
                
        except Exception as e:
            logger.error(f"❌ Error enhancing catalog with lexile scores: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def prepare_similarity_system(self):
        """Prepare the similarity matrix for book recommendations"""
        if self.catalog_df is None or self.catalog_df.empty:
            return
            
        try:
            # Create feature vectors for similarity
            features = []
            for _, row in self.catalog_df.iterrows():
                themes = str(row.get('themes', '')).lower()
                tone = str(row.get('tone', '')).lower() 
                age_range = str(row.get('age_range_llm', '') or row.get('age_range', '')).lower()
                
                # Combine features
                combined_features = f"{themes} {tone} {age_range}"
                features.append(combined_features)
            
            # Create TF-IDF matrix
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=1000,
                ngram_range=(1, 2)
            )
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(features)
            self.similarity_matrix = cosine_similarity(tfidf_matrix)
            
            logger.info("✓ Similarity system prepared")
            
        except Exception as e:
            logger.warning(f"Could not prepare similarity system: {e}")
    
    def enhanced_search(self, user_query, n_results=20):
        """
        Fixed search with proper pandas boolean handling
        """
        try:
            if not user_query or self.catalog_df is None or self.catalog_df.empty:
                return pd.DataFrame(), {}
            
            query_lower = user_query.lower()
            df = self.catalog_df.copy()
            
            # Create search mask with proper boolean handling
            title_mask = df['title'].fillna('').str.lower().str.contains(query_lower, na=False, regex=False)
            author_mask = df['author'].fillna('').str.lower().str.contains(query_lower, na=False, regex=False)
            themes_mask = df['themes'].fillna('').str.lower().str.contains(query_lower, na=False, regex=False)
            tone_mask = df['tone'].fillna('').str.lower().str.contains(query_lower, na=False, regex=False)
            age_mask = df['age_range_llm'].fillna('').astype(str).str.lower().str.contains(query_lower, na=False, regex=False)
            
            # Combine masks with | operator
            combined_mask = title_mask | author_mask | themes_mask | tone_mask | age_mask
            
            results = df[combined_mask].copy()
            
            if results.empty:
                logger.warning(f"No matches found for query: {user_query}")
                return pd.DataFrame(), {}
            
            logger.info(f"Found {len(results)} matches for query: {user_query}")
            
            # Simple parsed query
            parsed = {
                'themes': [],
                'tone': '',
                'age_range': '',
                'original_query': user_query
            }
            
            return results.head(n_results), parsed
            
        except Exception as e:
            logger.error(f"Enhanced search error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame(), {}
    
    def find_similar_books(self, book_data, n_recommendations=12):
        """Find similar books using the similarity matrix"""
        if self.similarity_matrix is None or self.catalog_df is None:
            return pd.DataFrame()
        
        try:
            # Find the book in our catalog
            title = book_data.get('title', '').strip()
            author = book_data.get('author', '').strip()
            
            matches = self.catalog_df[
                (self.catalog_df['title'].str.strip() == title) & 
                (self.catalog_df['author'].str.strip() == author)
            ]
            
            if matches.empty:
                # Try title-only match
                matches = self.catalog_df[self.catalog_df['title'].str.strip() == title]
                
            if matches.empty:
                # Fallback to content-based similarity using book features
                return self._find_similar_by_content(book_data, n_recommendations)
            
            # Get the index of the matched book
            book_idx = matches.index[0]
            source_book = self.catalog_df.iloc[book_idx]
            source_lexile = source_book.get('lexile_score', None)
            
            # Get similarity scores for this book
            sim_scores = list(enumerate(self.similarity_matrix[book_idx]))
            
            # Filter out the book itself
            sim_scores = [(i, score) for i, score in sim_scores if i != book_idx]
            
            print(f"DEBUG: Source book '{source_book.get('title', '')}' has Lexile score: {source_lexile}")
            
            # Apply Lexile-based filtering if source book has Lexile score
            if source_lexile and pd.notna(source_lexile) and source_lexile > 0:
                print(f"DEBUG: Applying Lexile filtering with source score: {source_lexile}")
                filtered_scores = []
                for i, content_sim in sim_scores:
                    candidate_book = self.catalog_df.iloc[i]
                    candidate_lexile = candidate_book.get('lexile_score', None)
                    
                    if candidate_lexile and pd.notna(candidate_lexile) and candidate_lexile > 0:
                        lexile_diff = abs(source_lexile - candidate_lexile)
                        # Hard threshold filtering - completely exclude books beyond 250L difference
                        if lexile_diff > 250:
                            print(f"DEBUG: EXCLUDED '{candidate_book.get('title', '')}' Lexile: {candidate_lexile}, diff: {lexile_diff}")
                            continue  # Skip books that are too different
                        
                        # Weighted scoring for books within threshold
                        if lexile_diff <= 100:
                            lexile_weight = 1.0
                        elif lexile_diff <= 200:
                            lexile_weight = 0.8 - (lexile_diff - 100) / 100 * 0.4  # 0.8 to 0.4
                        else:  # 200-250 range
                            lexile_weight = 0.3
                        
                        print(f"DEBUG: INCLUDED '{candidate_book.get('title', '')}' Lexile: {candidate_lexile}, diff: {lexile_diff}, weight: {lexile_weight}")
                    else:
                        # Books without Lexile scores get medium priority
                        lexile_weight = 0.6
                        print(f"DEBUG: NO LEXILE '{candidate_book.get('title', '')}' - using weight: {lexile_weight}")
                    
                    # Balanced scoring: Reading level primary, content themes significant secondary factor
                    lexile_score = lexile_weight * 600   # Primary weight for reading level appropriateness
                    content_score = content_sim * 400    # Significant weight for content theme similarity
                    final_score = lexile_score + content_score
                    filtered_scores.append((i, final_score))
                
                sim_scores = filtered_scores
                print(f"DEBUG: After Lexile filtering: {len(sim_scores)} candidates remaining")
            else:
                print(f"DEBUG: No valid Lexile score for source book, using age-based filtering instead")
                # Fallback: filter by age range if no Lexile score
                source_age = source_book.get('age_range_llm', '')
                if source_age and source_age.strip():
                    print(f"DEBUG: Using age-based filtering for source age: {source_age}")
                    weighted_scores = []
                    for i, content_sim in sim_scores:
                        candidate_book = self.catalog_df.iloc[i]
                        candidate_age = candidate_book.get('age_range_llm', '')
                        
                        # Simple age compatibility check
                        age_weight = 1.0
                        if candidate_age and candidate_age.strip():
                            if candidate_age == source_age:
                                age_weight = 1.0  # Same age range
                            elif self._ages_overlap(source_age, candidate_age):
                                age_weight = 0.8  # Overlapping age ranges
                            else:
                                age_weight = 0.3  # Different age ranges
                            print(f"DEBUG: Candidate '{candidate_book.get('title', '')}' age: {candidate_age}, weight: {age_weight}")
                        
                        final_score = content_sim * (0.7 + 0.3 * age_weight)
                        weighted_scores.append((i, final_score))
                    
                    sim_scores = weighted_scores
            
            # Sort by final similarity score (descending)
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get top N similar books
            top_indices = [i for i, _ in sim_scores[:n_recommendations]]
            similar_books = self.catalog_df.iloc[top_indices].copy()
            
            # Add similarity scores and convert to percentage
            raw_scores = [score for _, score in sim_scores[:n_recommendations]]
            
            # Convert to meaningful percentages
            # The raw scores are composite scores that can range widely based on lexile weighting
            # We'll normalize them to a 0-100% scale based on the highest score in this batch
            if raw_scores:
                max_score = max(raw_scores)
                min_score = min(raw_scores)
                if max_score > min_score:
                    # Scale to 60-100% range (most similar books should be at least 60% similar)
                    similarity_percentages = [
                        60 + (score - min_score) / (max_score - min_score) * 40 
                        for score in raw_scores
                    ]
                else:
                    # If all scores are the same, they're all 100% similar
                    similarity_percentages = [100.0] * len(raw_scores)
                
                # Round to 1 decimal place
                similarity_percentages = [round(pct, 1) for pct in similarity_percentages]
            else:
                similarity_percentages = []
            
            similar_books['similarity_score'] = raw_scores  # Keep raw for backend use
            similar_books['similarity_percentage'] = similarity_percentages
            
            return similar_books
            
        except Exception as e:
            logger.error(f"Error finding similar books: {e}")
            return pd.DataFrame()
    
    def _ages_overlap(self, age1, age2):
        """Check if two age ranges overlap"""
        try:
            def parse_age_range(age_str):
                if not age_str or age_str.strip() == '':
                    return None, None
                age_str = age_str.strip()
                if '-' in age_str:
                    parts = age_str.split('-')
                    return int(parts[0]), int(parts[1])
                else:
                    age = int(age_str)
                    return age, age
            
            min1, max1 = parse_age_range(age1)
            min2, max2 = parse_age_range(age2)
            
            if any(x is None for x in [min1, max1, min2, max2]):
                return False
            
            # Check if ranges overlap
            return max(min1, min2) <= min(max1, max2)
        except:
            return False
    
    def _find_similar_by_content(self, book_data, n_recommendations):
        """Fallback similarity using content features when book not in catalog"""
        try:
            # Simple content-based matching using themes and description
            target_themes = str(book_data.get('themes', '')).lower().strip()
            target_desc = str(book_data.get('description', '')).lower().strip()
            
            if len(target_themes) == 0 and len(target_desc) == 0:
                return pd.DataFrame()
            
            # Score books by content similarity
            scores = []
            for idx, row in self.catalog_df.iterrows():
                score = 0
                
                # Theme similarity
                book_themes = str(row.get('themes', '')).lower().strip()
                if len(target_themes) > 0 and len(book_themes) > 0:
                    theme_words = set(target_themes.split())
                    book_theme_words = set(book_themes.split())
                    theme_overlap = len(theme_words & book_theme_words)
                    score += theme_overlap * 0.6
                
                # Description similarity (simple word overlap)
                book_desc = str(row.get('description', '')).lower().strip()
                if len(target_desc) > 0 and len(book_desc) > 0:
                    desc_words = set(target_desc.split())
                    book_desc_words = set(book_desc.split())
                    desc_overlap = len(desc_words & book_desc_words)
                    score += min(desc_overlap * 0.4, 2.0)  # Cap description score
                
                scores.append((idx, score))
            
            # Sort by score and get top N
            scores = sorted(scores, key=lambda x: x[1], reverse=True)
            top_indices = [i for i, score in scores[:n_recommendations] if score > 0]
            
            if not top_indices:
                return pd.DataFrame()
            
            similar_books = self.catalog_df.iloc[top_indices].copy()
            similar_books['similarity_score'] = [score for _, score in scores[:len(top_indices)]]
            
            return similar_books
            
        except Exception as e:
            logger.error(f"Error in content-based similarity: {e}")
            return pd.DataFrame()
    
    def predict_single_book(self, book_data):
        """Predict reading level for a single book using ML models"""
        if not self.ml_models or not self.feature_columns:
            # Fallback prediction
            return self.create_fallback_prediction(book_data)
        
        try:
            # Engineer features using the same logic as training
            features = self.engineer_book_features(book_data)
            
            # Create feature vector
            feature_vector = []
            for col in self.feature_columns:
                feature_vector.append(features.get(col, 0))
            
            X = np.array(feature_vector).reshape(1, -1)
            
            # Make predictions
            confidence = features.get('avg_confidence', 0.7)
            
            # Tiered assignment
            if confidence >= 0.8:
                tier = "Tier 1: High Confidence"
            elif confidence >= 0.6:
                tier = "Tier 2: Medium Confidence" 
            else:
                tier = "Tier 3: Conservative Estimate"
            
            if tier == "Tier 3: Conservative Estimate":
                lexile_pred = 400
                category_pred_encoded = 1
            else:
                lexile_pred = self.ml_models['lexile_model'].predict(X)[0]
                category_pred_encoded = self.ml_models['category_model'].predict(X)[0]
            
            # Convert category back to label
            category_label = self.label_encoders['category'].classes_[category_pred_encoded]
            
            # Map to age ranges
            age_range_mapping = {
                'Early': '3-5', 'Beginning': '6-8',
                'Intermediate': '9-12', 'Advanced': '13+'
            }
            age_range = age_range_mapping.get(category_label, '6-8')
            
            return {
                'success': True,
                'lexile_score': int(round(lexile_pred)),
                'age_category': category_label,
                'age_range': age_range,
                'confidence_score': round(confidence, 3),
                'assignment_tier': tier
            }
            
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return self.create_fallback_prediction(book_data)
    
    def create_fallback_prediction(self, book_data):
        """Create fallback prediction when ML models aren't available"""
        title = book_data.get('title', '').lower()
        author = book_data.get('author', '').lower()
        themes = book_data.get('themes', '').lower()
        
        # Rule-based prediction
        if 'seuss' in author or 'carle' in author:
            return {
                'success': True,
                'lexile_score': 220,
                'age_category': 'Early',
                'age_range': '3-5',
                'confidence_score': 0.8,
                'assignment_tier': 'Tier 2: Rule-based'
            }
        elif 'rowling' in author or 'harry potter' in title:
            return {
                'success': True,
                'lexile_score': 880,
                'age_category': 'Advanced', 
                'age_range': '9-12',
                'confidence_score': 0.75,
                'assignment_tier': 'Tier 2: Rule-based'
            }
        elif 'magic' in themes or 'adventure' in themes:
            return {
                'success': True,
                'lexile_score': 650,
                'age_category': 'Intermediate',
                'age_range': '9-12',
                'confidence_score': 0.65,
                'assignment_tier': 'Tier 2: Rule-based'
            }
        else:
            return {
                'success': True,
                'lexile_score': 400,
                'age_category': 'Beginning',
                'age_range': '6-8',
                'confidence_score': 0.5,
                'assignment_tier': 'Tier 3: Conservative'
            }
    
    def engineer_book_features(self, book_data):
        """Engineer features for ML prediction (matching training logic)"""
        features = {}
        
        # Author features
        author = book_data.get('author_clean') or book_data.get('author', '')
        if author in self.author_stats:
            stats = self.author_stats[author]
            features['author_book_count'] = stats['book_count']
            features['author_avg_lexile'] = stats.get('avg_lexile', 0) or 0
            features['author_lexile_consistency'] = stats.get('std_lexile', 0)
            features['is_prolific_author'] = 1 if stats['book_count'] >= 5 else 0
            features['is_specialist_author'] = 1 if stats.get('std_lexile', float('inf')) < 50 else 0
            
            # Category encoding
            category_encoding = {'Early': 0, 'Beginning': 1, 'Intermediate': 2, 'Advanced': 3, 'Unknown': 4}
            features['author_primary_category_encoded'] = category_encoding.get(stats.get('primary_age_category', 'Unknown'), 4)
        else:
            features['author_book_count'] = 1
            features['author_avg_lexile'] = 0
            features['author_lexile_consistency'] = 0
            features['is_prolific_author'] = 0
            features['is_specialist_author'] = 0
            features['author_primary_category_encoded'] = 4
        
        # Text features
        title = book_data.get('title', '')
        features['title_length'] = len(title)
        features['title_word_count'] = len(title.split()) if title else 0
        features['title_has_numbers'] = 1 if any(c.isdigit() for c in title) else 0
        
        description = book_data.get('description', '')
        features['description_length'] = len(description)
        features['description_word_count'] = len(description.split()) if description else 0
        features['has_substantial_description'] = 1 if len(description) > 100 else 0
        
        # Theme features
        themes_str = book_data.get('themes', '')
        themes = [t.strip().lower() for t in themes_str.split(',') if t.strip()] if themes_str else []
        features['theme_count'] = len(themes)
        
        # Specific themes
        for theme in ['friendship', 'adventure', 'family', 'imagination', 'magic', 'humor']:
            features[f'has_{theme}_theme'] = 1 if theme in themes else 0
        
        # Complex themes  
        complex_themes = ['identity', 'courage', 'loyalty', 'acceptance']
        features['has_complex_themes'] = 1 if any(t in complex_themes for t in themes) else 0
        
        # Confidence features
        features['reading_confidence_llm'] = book_data.get('reading_confidence_llm', 0.7)
        features['lexile_confidence'] = book_data.get('lexile_confidence', 0.7) 
        features['avg_confidence'] = (features['reading_confidence_llm'] + features['lexile_confidence']) / 2
        features['high_confidence'] = 1 if features['avg_confidence'] >= 0.8 else 0
        
        # Data quality
        features['data_completeness_score'] = (
            (1 if len(description) > 50 else 0) +
            (1 if themes_str else 0) +  
            0  # No summary for new books
        ) / 3
        
        return features

# In-memory storage for demo (replace with database in production)
# File path for persistent user data
USER_DATA_FILE = os.path.join(os.path.dirname(__file__), 'user_books.json')

def load_user_books():
    """Load user books from persistent storage"""
    try:
        if os.path.exists(USER_DATA_FILE):
            with open(USER_DATA_FILE, 'r') as f:
                data = json.load(f)
                print(f"DEBUG: Loaded user books from {USER_DATA_FILE}")
                print(f"DEBUG: Loaded data contains: skipped={len(data.get('skipped', []))}, read={len(data.get('read', []))}, favorites={len(data.get('favorites', []))}")
                return data
    except Exception as e:
        print(f"DEBUG: Error loading user books: {e}")
    
    # Return default structure if file doesn't exist or error occurred
    return {
        'favorites': [],
        'skipped': [],
        'read': []
    }

def save_user_books(user_books):
    """Save user books to persistent storage"""
    try:
        print(f"DEBUG: SAVE FUNCTION CALLED with path: {USER_DATA_FILE}")
        with open(USER_DATA_FILE, 'w') as f:
            json.dump(user_books, f, indent=2)
            print(f"DEBUG: Successfully saved user books to {USER_DATA_FILE}")
            print(f"DEBUG: Saved data contains: skipped={len(user_books.get('skipped', []))}, read={len(user_books.get('read', []))}, favorites={len(user_books.get('favorites', []))}")
    except Exception as e:
        print(f"DEBUG: Error saving user books: {e}")
        import traceback
        traceback.print_exc()

# Load user books from persistent storage at startup
user_books = load_user_books()

def filter_user_books(df):
    """Filter out books that the user has already skipped or read"""
    if df is None or df.empty:
        return df
    
    print(f"DEBUG: filter_user_books called with {len(df)} books")
    print(f"DEBUG: user_books contains: skipped={len(user_books.get('skipped', []))}, read={len(user_books.get('read', []))}")
    
    # Get titles of books to exclude
    excluded_titles = set()
    for book in user_books.get('skipped', []):
        title = book.get('title', '')
        excluded_titles.add(title)
        print(f"DEBUG: Will exclude skipped book: '{title}'")
    for book in user_books.get('read', []):
        title = book.get('title', '')
        excluded_titles.add(title)
        print(f"DEBUG: Will exclude read book: '{title}'")
    
    if excluded_titles:
        print(f"DEBUG: Looking for titles to exclude: {excluded_titles}")
        
        # Filter out books with titles in the excluded set
        mask = ~df['title'].isin(excluded_titles)
        filtered_df = df[mask].copy()
        
        # Debug: show which books were actually filtered out
        excluded_books = df[~mask]
        if not excluded_books.empty:
            print(f"DEBUG: Actually filtered out these books: {list(excluded_books['title'])}")
        else:
            # If no books were filtered, let's check what titles are similar to what we're looking for
            for excluded_title in excluded_titles:
                similar_titles = df[df['title'].str.contains(excluded_title.split()[0], na=False, case=False)]['title'].tolist()[:3]
                if similar_titles:
                    print(f"DEBUG: Couldn't find exact '{excluded_title}', but found similar: {similar_titles}")
        
        excluded_count = len(df) - len(filtered_df)
        if excluded_count > 0:
            print(f"DEBUG: Filtered out {excluded_count} books that were already skipped/read")
        
        return filtered_df
    else:
        print("DEBUG: No excluded titles found, returning original dataframe")
    
    return df

@app.route('/api/save-book', methods=['POST'])
def save_book():
    """Save book to user's list (favorites, skipped, or read)"""
    print("DEBUG: save_book endpoint called!")
    try:
        data = request.get_json()
        book_data = data.get('book')
        list_type = data.get('list_type')  # 'favorites', 'skipped', 'read'
        
        if not book_data or not list_type:
            return jsonify({'success': False, 'error': 'Missing book data or list type'}), 400
        
        if list_type not in ['favorites', 'skipped', 'read']:
            return jsonify({'success': False, 'error': 'Invalid list type'}), 400
        
        # Remove from other lists if it exists
        for other_list in ['favorites', 'skipped', 'read']:
            if other_list != list_type:
                user_books[other_list] = [b for b in user_books[other_list] if b['title'] != book_data['title']]
        
        # Add to the specified list (avoid duplicates)
        existing = [b for b in user_books[list_type] if b['title'] == book_data['title']]
        if not existing:
            user_books[list_type].append(book_data)
        
        # Save to persistent storage
        save_user_books(user_books)
        
        return jsonify({'success': True, 'message': f'Book added to {list_type}'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/get-user-books/<list_type>', methods=['GET'])
def get_user_books(list_type):
    """Get user's books from a specific list"""
    try:
        if list_type not in ['favorites', 'skipped', 'read']:
            return jsonify({'success': False, 'error': 'Invalid list type'}), 400
        
        books = user_books.get(list_type, [])
        return jsonify({'success': True, 'books': books, 'total': len(books)})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/remove-book', methods=['POST'])
def remove_book():
    """Remove book from user's list"""
    try:
        data = request.get_json()
        title = data.get('title')
        list_type = data.get('list_type')
        
        if not title or not list_type:
            return jsonify({'success': False, 'error': 'Missing title or list type'}), 400
        
        user_books[list_type] = [b for b in user_books[list_type] if b['title'] != title]
        
        return jsonify({'success': True, 'message': f'Book removed from {list_type}'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    
    def basic_search(self, user_query, n_results=20):
        """Fallback search when enhanced features fail"""
        if self.catalog_df is None or self.catalog_df.empty:
            return pd.DataFrame(), {}
            
        # Simple text matching
        query_lower = user_query.lower()
        mask = (
            self.catalog_df['title'].str.lower().str.contains(query_lower, na=False) |
            self.catalog_df['author'].str.lower().str.contains(query_lower, na=False) |
            self.catalog_df['themes'].str.lower().str.contains(query_lower, na=False)
        )
        
        results = self.catalog_df[mask].head(n_results)
        return results, {'themes': [], 'tone': '', 'age_range': ''}

    def enhance_with_ml_predictions(self, df):
        """Simplified ML enhancement to avoid boolean array issues"""
        if not self.ml_models or df.empty:
            return df
            
        # Skip ML enhancement for now to avoid errors
        # Just return the original DataFrame
        enhanced_df = df.copy()
        
        # Add placeholder ML predictions
        enhanced_df['predicted_lexile'] = enhanced_df.get('lexile_score', 400)
        enhanced_df['ml_confidence'] = 0.7
        enhanced_df['assignment_tier'] = 'Tier 2: Medium Confidence'
        
        return enhanced_df
    
    # def enhance_with_ml_predictions(self, df):
    #     """Add ML predictions to existing book results"""
    #     if not self.ml_models or df.empty:
    #         return df
            
    #     enhanced_df = df.copy()
    #     ml_predictions = []
        
    #     for _, book in df.iterrows():
    #         try:
    #             prediction = self.predict_single_book(book.to_dict())
    #             ml_predictions.append({
    #                 'predicted_lexile': prediction.get('lexile_score', book.get('lexile_score')),
    #                 'predicted_category': prediction.get('age_category', 'Unknown'),
    #                 'ml_confidence': prediction.get('confidence_score', 0.5),
    #                 'assignment_tier': prediction.get('assignment_tier', 'Tier 3')
    #             })
    #         except Exception as e:
    #             # Fallback for failed predictions
    #             ml_predictions.append({
    #                 'predicted_lexile': book.get('lexile_score', 400),
    #                 'predicted_category': 'Unknown',
    #                 'ml_confidence': 0.5,
    #                 'assignment_tier': 'Tier 3'
    #             })
        
    #     # Add ML predictions to DataFrame
    #     for i, pred in enumerate(ml_predictions):
    #         for key, value in pred.items():
    #             enhanced_df.iloc[i, enhanced_df.columns.get_loc(key) if key in enhanced_df.columns else len(enhanced_df.columns)] = value
        
    #     return enhanced_df
    
    def rerank_results(self, df, parsed_query, original_query):
        """Re-rank results based on ML confidence and query matching"""
        if df.empty:
            return df
            
        scoring_df = df.copy()
        
        # Base score from ML confidence
        scoring_df['relevance_score'] = scoring_df.get('ml_confidence', 0.5)
        
        # Boost for exact theme matches
        if parsed_query.get('themes'):
            for theme in parsed_query['themes']:
                theme_match_mask = scoring_df['themes'].str.lower().str.contains(theme.lower(), na=False)
                scoring_df.loc[theme_match_mask, 'relevance_score'] += 0.2
        
        # Boost for tone matches
        if parsed_query.get('tone'):
            tone_match_mask = scoring_df['tone'].str.lower().str.contains(parsed_query['tone'].lower(), na=False)
            scoring_df.loc[tone_match_mask, 'relevance_score'] += 0.15
        
        # Boost for high-tier ML predictions
        tier1_mask = scoring_df.get('assignment_tier', '').str.contains('Tier 1', na=False)
        scoring_df.loc[tier1_mask, 'relevance_score'] += 0.1
        
        # Sort by relevance score
        scoring_df = scoring_df.sort_values('relevance_score', ascending=False)
        
        return scoring_df

# Initialize the recommendation engine
try:
    recommendation_engine = HybridRecommendationEngine()
    logger.info("✓ Hybrid recommendation engine initialized")
except Exception as e:
    logger.error(f"Failed to initialize recommendation engine: {e}")
    recommendation_engine = None

# Initialize the enhanced Lexile predictor with enriched scores
lexile_predictor = None
if LEXILE_PREDICTOR_AVAILABLE and predictor_class:
    try:
        lexile_predictor = predictor_class()
        logger.info(f"✅ {predictor_class.__name__} initialized successfully")
        # Show stats if available
        if hasattr(lexile_predictor, 'get_stats'):
            stats = lexile_predictor.get_stats()
            logger.info(f"📊 Enriched scores available: {stats.get('enriched_scores_count', 0)}")
            logger.info(f"🤖 ML model available: {stats.get('ml_model_available', False)}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize {predictor_class.__name__}: {e}")
        lexile_predictor = None
else:
    logger.warning("⚠️  Lexile predictor not available - using fallback")




# Flask Routes

@app.route('/')
def index():
    """Serve the main application"""
    return send_from_directory('.', 'app.html')

@app.route('/app.html')
def app_html():
    """Serve the app.html file"""
    return send_from_directory('.', 'app.html')



@app.route('/api/predict', methods=['POST'])
def predict_book():
    """Predict reading level using ML models"""
    if not recommendation_engine:
        return jsonify({'success': False, 'error': 'Prediction models not available'}), 500
    
    try:
        data = request.get_json()
        
        if not data.get('title') or not data.get('author'):
            return jsonify({'success': False, 'error': 'Title and author are required'}), 400
        
        # Make prediction
        result = recommendation_engine.predict_single_book(data)
        
        logger.info(f"Predicted reading level for '{data['title']}': {result.get('lexile_score')}L, {result.get('age_category')}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'success': False, 'error': 'Prediction failed'}), 500


@app.route('/api/similar', methods=['POST'])
def get_similar_books():
    """Find similar books using ML-enhanced similarity"""
    try:
        print("DEBUG: Similar books endpoint called")
        
        if not recommendation_engine:
            print("DEBUG: No recommendation engine")
            return jsonify({'success': False, 'error': 'Similarity engine not available'}), 500
        
        data = request.get_json()
        print(f"DEBUG: Received data: {data}")
        
        if not data.get('title') or not data.get('author'):
            return jsonify({'success': False, 'error': 'Title and author are required'}), 400
        
        print("DEBUG: About to call find_similar_books")
        similar_df = recommendation_engine.find_similar_books(data, n_recommendations=12)
        print(f"DEBUG: Found {len(similar_df)} similar books")
        
        # Filter out skipped and read books
        similar_df = filter_user_books(similar_df)
        print(f"DEBUG: After filtering user books: {len(similar_df)} similar books")
        
        results = []
        for _, book in similar_df.iterrows():
            book_dict = {}
            for col, value in book.items():
                try:
                    # Convert to standard Python types first
                    if isinstance(value, (np.integer, np.floating)):
                        book_dict[col] = float(value)
                    elif value is None or (isinstance(value, float) and np.isnan(value)):
                        book_dict[col] = None
                    else:
                        # Try to check if it's a pandas NA/NaN value
                        str_value = str(value).strip()
                        if str_value.lower() in ['nan', 'none', '<na>', ''] or str_value == 'nan':
                            book_dict[col] = None
                        else:
                            book_dict[col] = str_value
                except Exception:
                    # If anything fails, just convert to string or None
                    try:
                        book_dict[col] = str(value) if value is not None else None
                    except:
                        book_dict[col] = None
            results.append(book_dict)
        
        print(f"DEBUG: Returning {len(results)} results")
        return jsonify({
            'success': True,
            'similar_books': results,
            'total_found': len(results)
        })
        
    except Exception as e:
        print(f"DEBUG: Similar books error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500
# @app.route('/api/similar', methods=['POST'])
# def get_similar_books():
#     """Find similar books using ML-enhanced similarity"""
#     if not recommendation_engine:
#         return jsonify({'success': False, 'error': 'Similarity engine not available'}), 500
    
#     try:
#         data = request.get_json()
        
#         if not data.get('title') or not data.get('author'):
#             return jsonify({'success': False, 'error': 'Title and author are required'}), 400
        
#         similar_df = recommendation_engine.find_similar_books(data, n_recommendations=12)
        
#         results = []
#         for _, book in similar_df.iterrows():
#             book_dict = book.to_dict()
#             # Clean up NaN values
#             for key, value in book_dict.items():
#                 if pd.isna(value):
#                     book_dict[key] = None
#                 elif isinstance(value, (np.integer, np.floating)):
#                     book_dict[key] = float(value)
#             results.append(book_dict)
        
#         return jsonify({
#             'success': True,
#             'similar_books': results,
#             'total_found': len(results)
#         })
        
#     except Exception as e:
#         logger.error(f"Similar books error: {e}")
#         return jsonify({'success': False, 'error': 'Failed to find similar books'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """System health check"""
    status = {
        'status': 'healthy',
        'ml_models_loaded': bool(recommendation_engine and recommendation_engine.ml_models),
        'catalog_loaded': bool(recommendation_engine and recommendation_engine.catalog_df is not None),
        'similarity_ready': bool(recommendation_engine and recommendation_engine.similarity_matrix is not None),
        'timestamp': datetime.now().isoformat()
    }
    
    if recommendation_engine:
        status['catalog_size'] = len(recommendation_engine.catalog_df) if recommendation_engine.catalog_df is not None else 0
        status['authors_known'] = len(recommendation_engine.author_stats)
    
    return jsonify(status)

@app.route('/api/stats', methods=['GET'])
def get_system_stats():
    """Get system statistics for analytics page"""
    if not recommendation_engine:
        return jsonify({'success': False, 'error': 'System not available'}), 500
    
    try:
        stats = {
            'total_books': len(recommendation_engine.catalog_df) if recommendation_engine.catalog_df is not None else 0,
            'authors_analyzed': len(recommendation_engine.author_stats),
            'ml_models_loaded': bool(recommendation_engine.ml_models),
            'similarity_matrix_ready': bool(recommendation_engine.similarity_matrix is not None),
        }
        
        if recommendation_engine.catalog_df is not None:
            # Calculate coverage stats
            df = recommendation_engine.catalog_df
            stats['lexile_coverage'] = float((df['lexile_score'].notna()).mean() * 100)
            stats['theme_coverage'] = float((df['themes'].fillna('').str.len() > 0).mean() * 100)
            stats['age_coverage'] = float((df['age_range_llm'].fillna('').str.len() > 0).mean() * 100)
        
        return jsonify({'success': True, 'stats': stats})
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'success': False, 'error': 'Failed to get stats'}), 500


@app.route('/api/test-search', methods=['GET'])
def test_search():
    """Simple search test"""
    if not recommendation_engine:
        return jsonify({"error": "No engine"})
    
    try:
        # Test the basic search components
        query = "Dr. Seuss"
        df = recommendation_engine.catalog_df
        
        # Simple title/author match
        simple_results = df[
            df['title'].str.contains('Seuss', na=False, case=False) |
            df['author'].str.contains('Seuss', na=False, case=False)
        ]
        
        return jsonify({
            "query": query,
            "simple_search_results": len(simple_results),
            "sample_titles": simple_results['title'].head(3).tolist() if len(simple_results) > 0 else []
        })
        
    except Exception as e:
        return jsonify({"error": str(e)})

def simple_parse_query(query):
    """Enhanced parser for complex queries"""
    query_lower = query.lower()
    result = {
        'themes': [],
        'age_range': None,
        'tone': None,
        'lexile_range': None,
        'parsing_method': 'enhanced_rule_based'
    }
    
    import re
    
    # Age extraction - multiple patterns
    age_patterns = [
        r'ages?\s*(\d+)[-–]\s*(\d+)',  # "ages 6-8" or "age 6–8"
        r'(\d+)[-–]\s*(\d+)\s*years?\s*old',  # "6-8 years old"
        r'ages?\s*(\d+)',  # "ages 15" or "age 15" - must come after range patterns
        r'(\d+)\s*years?\s*old',  # "15 years old" or "5 year old" 
        r'for\s*(\d+)',  # "for 15"
        r'(\d+)\s*year\s*old',  # "5 year old" (keeping for backward compatibility)
    ]
    
    for pattern in age_patterns:
        match = re.search(pattern, query_lower)
        if match:
            if len(match.groups()) == 2:  # Range
                min_age, max_age = match.groups()
                result['age_range'] = f"{min_age}-{max_age}"
            else:  # Single age
                age = int(match.group(1))
                if age <= 3:
                    result['age_range'] = "0-3"
                elif age <= 5:
                    result['age_range'] = "3-5"
                elif age <= 8:
                    result['age_range'] = "6-8"
                elif age <= 12:
                    result['age_range'] = "9-12"
                else:
                    result['age_range'] = "13+"
            break
    
    # Lexile extraction
    lexile_patterns = [
        r'lexile\s*(\d+)[-–]\s*(\d+)',  # "lexile 400-600"
        r'(\d+)[-–]\s*(\d+)\s*(?:L|lexile)',  # "400-600L" 
        r'(\d+)\s*(?:L|lexile)',  # "400L"
    ]
    
    for pattern in lexile_patterns:
        match = re.search(pattern, query_lower)
        if match:
            if len(match.groups()) == 2:  # Range
                min_lex, max_lex = match.groups()
                result['lexile_range'] = (int(min_lex), int(max_lex))
            else:  # Single score - create range ±100
                lexile = int(match.group(1))
                result['lexile_range'] = (lexile - 100, lexile + 100)
            break
    
    # Theme extraction
    theme_keywords = {
        'magic': ['magic', 'magical', 'wizard', 'witch', 'spell', 'fairy'],
        'friendship': ['friend', 'friendship', 'friends'],
        'adventure': ['adventure', 'quest', 'journey', 'exploring'],
        'family': ['family', 'parent', 'mom', 'dad', 'sibling', 'brother', 'sister'],
        'school': ['school', 'classroom', 'teacher', 'student'],
        'animals': ['animal', 'dog', 'cat', 'pet', 'zoo', 'farm'],
        'mystery': ['mystery', 'detective', 'solve', 'clue'],
        'fantasy': ['fantasy', 'dragon', 'kingdom', 'princess', 'prince'],
        'science': ['science', 'experiment', 'space', 'robot'],
        'sports': ['sports', 'soccer', 'basketball', 'baseball', 'football'],
        'bedtime': ['bedtime', 'sleep', 'night', 'dream'],
        'humor': ['humor', 'comedy', 'joke', 'laugh']
    }
    
    for theme, keywords in theme_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            result['themes'].append(theme)
    
    # Tone extraction
    tone_keywords = {
        'funny': ['funny', 'hilarious', 'comedy', 'humor', 'silly', 'humorous'],
        'whimsical': ['whimsical', 'playful', 'imaginative', 'quirky'],
        'scary': ['scary', 'spooky', 'horror', 'frightening'],
        'sad': ['sad', 'crying', 'emotional', 'tearjerker'],
        'inspiring': ['inspiring', 'motivational', 'uplifting'],
        'gentle': ['gentle', 'soft', 'calm', 'peaceful'],
        'exciting': ['exciting', 'thrilling', 'action-packed']
    }
    
    for tone, keywords in tone_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            result['tone'] = tone
            break
    
    return result

def handle_category_search(data):
    """Handle direct category searches (theme groups, tone groups, lexile ranges)"""
    try:
        category = data.get('category', '')
        category_type = data.get('category_type', '')
        page = data.get('page', 1)
        per_page = data.get('per_page', 12)
        
        print(f"DEBUG: Category search - {category_type}: {category}")
        
        if not recommendation_engine or recommendation_engine.catalog_df is None:
            return jsonify({'success': False, 'error': 'No data'}), 500
        
        df = recommendation_engine.catalog_df
        mask = pd.Series([True] * len(df), index=df.index)
        
        if category_type == 'theme':
            # Direct theme group expansion
            if category in THEME_GROUPS:
                expanded_themes = THEME_GROUPS[category]
                print(f"DEBUG: Expanding theme group '{category}' to: {expanded_themes}")
                
                theme_mask = pd.Series([False] * len(df), index=df.index)
                for theme in expanded_themes:
                    theme_match = df['themes'].fillna('').str.contains(
                        theme, case=False, na=False, regex=False
                    )
                    theme_mask = theme_mask | theme_match
                    print(f"DEBUG: Theme '{theme}' found {theme_match.sum()} matches")
                mask = mask & theme_mask
                
        elif category_type == 'tone':
            # Direct tone group expansion
            if category in TONE_GROUPS:
                expanded_tones = TONE_GROUPS[category]
                print(f"DEBUG: Expanding tone group '{category}' to: {expanded_tones}")
                
                tone_mask = pd.Series([False] * len(df), index=df.index)
                for tone in expanded_tones:
                    tone_match = df['tone'].fillna('').str.contains(
                        tone, case=False, na=False, regex=False
                    )
                    tone_mask = tone_mask | tone_match
                    print(f"DEBUG: Tone '{tone}' found {tone_match.sum()} matches")
                mask = mask & tone_mask
                
        elif category_type == 'lexile':
            # Handle lexile ranges like "lexile 400-600" and "lexile 1000+"
            import re
            lexile_range_match = re.search(r'lexile (\d+)-(\d+)', category.lower())
            lexile_plus_match = re.search(r'lexile (\d+)\+', category.lower())
            
            if lexile_range_match:
                min_lexile = int(lexile_range_match.group(1))
                max_lexile = int(lexile_range_match.group(2))
                print(f"DEBUG: Lexile range: {min_lexile}-{max_lexile}")
                
                lexile_values = pd.to_numeric(df['lexile_score'], errors='coerce')
                lexile_mask = (lexile_values >= min_lexile) & (lexile_values <= max_lexile)
                mask = mask & lexile_mask
                print(f"DEBUG: Lexile range filter found {lexile_mask.sum()} matches")
            elif lexile_plus_match:
                min_lexile = int(lexile_plus_match.group(1))
                print(f"DEBUG: Lexile 1000+: >= {min_lexile}")
                
                lexile_values = pd.to_numeric(df['lexile_score'], errors='coerce')
                lexile_mask = lexile_values >= min_lexile
                mask = mask & lexile_mask
                print(f"DEBUG: Lexile 1000+ filter found {lexile_mask.sum()} matches")
        
        # Get filtered results
        filtered_df = df[mask]
        print(f"DEBUG: Total matches after filtering: {len(filtered_df)}")
        
        # Filter out books that user has already skipped or read
        filtered_df = filter_user_books(filtered_df)
        print(f"DEBUG: Total matches after user book filtering: {len(filtered_df)}")
        
        # Apply pagination
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_df = filtered_df.iloc[start_idx:end_idx]
        
        # Convert to results format
        results = []
        for _, book in paginated_df.iterrows():
            goodreads_url = str(book.get('goodreads_url', ''))
            results.append({
                'title': str(book.get('title', '')),
                'author': str(book.get('author', '')),
                'themes': str(book.get('themes', '')),
                'tone': str(book.get('tone', '')),
                'age_range': str(book.get('age_range', '')),
                'lexile_score': str(book.get('lexile_score', '')),
                'cover_url': str(book.get('cover_url', '')) if book.get('cover_url') and str(book.get('cover_url')) != 'nan' else '',
                'description': str(book.get('description', '')),
                'summary_gpt': str(book.get('summary_gpt', '')) if book.get('summary_gpt') and str(book.get('summary_gpt')) != 'nan' else '',
                'goodreads_url': goodreads_url if goodreads_url and goodreads_url != 'nan' else ''
            })
        
        # Calculate pagination info
        total_results = len(filtered_df)
        total_pages = (total_results + per_page - 1) // per_page
        
        pagination = {
            'current_page': page,
            'total_pages': total_pages,
            'total_results': total_results,
            'per_page': per_page,
            'has_prev': page > 1,
            'has_next': page < total_pages
        }
        
        return jsonify({
            'success': True,
            'results': results,
            'pagination': pagination,
            'parsed_criteria': {
                'category': category,
                'category_type': category_type,
                'parsing_method': 'direct_category'
            },
            'query_searched': f"{category_type}: {category}"
        })
        
    except Exception as e:
        print(f"Category search error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/simple-search', methods=['POST'])
def simple_search():
    """Hybrid search with pagination"""
    try:
        data = request.get_json()
        
        # Check if this is a direct category search
        if data.get('category_search', False):
            return handle_category_search(data)
        
        query = data.get('query', '').strip()
        page = data.get('page', 1)
        per_page = data.get('per_page', 12)
        difficulty_level = data.get('difficulty_level', '')
        
        if not query:
            return jsonify({'success': False, 'error': 'Query required'}), 400
        
        if not recommendation_engine or recommendation_engine.catalog_df is None:
            return jsonify({'success': False, 'error': 'No data'}), 500
        
        df = recommendation_engine.catalog_df
        
        # Use the hybrid parser for best of both worlds
        try:
            criteria = hybrid_parse_query(query)
        except Exception as e:
            print(f"DEBUG: Hybrid parser failed, falling back to simple: {e}")
            criteria = simple_parse_query(query)
        print(f"DEBUG: Query: '{query}'")
        print(f"DEBUG: Parsed criteria: {criteria}")
        
        # Apply all filtering logic
        mask = pd.Series([True] * len(df), index=df.index)
        print(f"DEBUG: Starting with {mask.sum()} books")
        
        # Theme filters - with group expansion
        if criteria['themes']:
            print(f"DEBUG: Applying theme filter for: {criteria['themes']}")
            
            # Check if any of the themes match our group names exactly
            # Reconstruct potential group names from the parsed themes
            raw_themes = criteria['themes']
            potential_groups = []
            
            # Try to reconstruct group names from parsed themes
            theme_text = ' '.join(raw_themes).lower()
            for group_name in THEME_GROUPS.keys():
                if group_name.lower() in theme_text or all(word.lower() in theme_text for word in group_name.split() if len(word) > 2):
                    potential_groups.append(group_name)
                    print(f"DEBUG: Found potential group match: {group_name}")
            
            # If we found group matches, use those, otherwise use individual themes
            if potential_groups:
                expanded_themes = expand_theme_groups(potential_groups)
                print(f"DEBUG: Using group expansion for {potential_groups}")
            else:
                expanded_themes = expand_theme_groups(criteria['themes'])
                print(f"DEBUG: Using individual themes")
                
            print(f"DEBUG: Expanded themes: {expanded_themes}")
            theme_mask = pd.Series([False] * len(df), index=df.index)
            for theme in expanded_themes:
                theme_match = df['themes'].fillna('').str.contains(
                    theme, case=False, na=False, regex=False
                )
                theme_mask = theme_mask | theme_match
                print(f"DEBUG: Theme '{theme}' found {theme_match.sum()} matches")
            mask = mask & theme_mask
            print(f"DEBUG: After theme filter: {mask.sum()} books")
        
        # Check if this is a direct title/author search (not a comparison query)
        is_direct_search = criteria.get('is_direct_title_search', False)
        
        # For direct searches of famous books, search for actual titles/authors instead of themes
        if is_direct_search:
            print("DEBUG: Direct book title search detected, searching titles and authors")
            broad_mask = (
                df['title'].fillna('').str.contains(query, case=False, na=False, regex=False) |
                df['author'].fillna('').str.contains(query, case=False, na=False, regex=False)
            )
            mask = broad_mask
            print(f"DEBUG: Direct search found: {mask.sum()} books")
            
            # For direct searches, skip additional filtering - user wants all books by that title/author
            print("DEBUG: Skipping additional filters for direct title search")
            skip_additional_filters = True
        else:
            # Check if we should fall back to broad search for unclear queries
            has_themes = criteria.get('themes') and len(criteria.get('themes', [])) > 0
            has_tone = criteria.get('tone') and criteria['tone'].strip()
            has_age = criteria.get('age_range') and criteria['age_range'].strip()
            has_lexile = criteria.get('lexile_range') and len(criteria.get('lexile_range', [])) == 2
            
            print(f"DEBUG: has_themes={has_themes}, has_tone={has_tone}, has_age={has_age}, has_lexile={has_lexile}")
            
            if not (has_themes or has_tone or has_age or has_lexile):
                print("DEBUG: No specific criteria found, using broad search")
                print(f"DEBUG: Search query: '{query}'")
                title_mask = df['title'].fillna('').str.contains(query, case=False, na=False, regex=False)
                author_mask = df['author'].fillna('').str.contains(query, case=False, na=False, regex=False)
                themes_mask = df['themes'].fillna('').str.contains(query, case=False, na=False, regex=False)
                print(f"DEBUG: Title matches: {title_mask.sum()}")
                print(f"DEBUG: Author matches: {author_mask.sum()}")
                print(f"DEBUG: Themes matches: {themes_mask.sum()}")
                broad_mask = title_mask | author_mask | themes_mask
                mask = broad_mask
                print(f"DEBUG: Broad search found: {mask.sum()} books")
                # Show some sample titles that matched
                if mask.sum() > 0:
                    sample_matches = df[mask]['title'].head(3).tolist()
                    print(f"DEBUG: Sample matches: {sample_matches}")
        

        # Age filter - more precise matching (skip for direct searches)
        skip_additional_filters = locals().get('skip_additional_filters', False)
        
        if criteria['age_range'] and not skip_additional_filters:
            print(f"DEBUG: Applying age filter for: {criteria['age_range']}")
            age_range = criteria['age_range']
            
            def age_range_overlaps(book_age_str, target_age_range):
                """Check if book age range overlaps with target age range"""
                if not book_age_str or pd.isna(book_age_str):
                    return False
                
                book_age_str = str(book_age_str).strip()
                
                # Handle target age range
                if target_age_range == "13+":
                    target_min, target_max = 13, 18
                elif '-' in target_age_range:
                    target_min, target_max = map(int, target_age_range.split('-'))
                else:
                    # Single age
                    target_min = target_max = int(target_age_range)
                
                # Parse book age range
                if "13+" in book_age_str or "teen" in book_age_str.lower():
                    book_min, book_max = 13, 18
                elif '-' in book_age_str:
                    # Extract numbers from ranges like "3-5", "ages 6-8", etc.
                    import re
                    numbers = re.findall(r'\d+', book_age_str)
                    if len(numbers) >= 2:
                        book_min, book_max = int(numbers[0]), int(numbers[1])
                    else:
                        return False
                else:
                    # Single number
                    import re
                    numbers = re.findall(r'\d+', book_age_str)
                    if numbers:
                        book_age = int(numbers[0])
                        book_min = book_max = book_age
                    else:
                        return False
                
                # Check for overlap: ranges overlap if not (one ends before other starts)
                overlap = not (book_max < target_min or book_min > target_max)
                
                print(f"DEBUG: Book age {book_age_str} -> ({book_min}-{book_max}) vs target ({target_min}-{target_max}) = {overlap}")
                return overlap
            
            # Apply age filtering using the overlap function
            age_mask = df['age_range_llm'].apply(lambda x: age_range_overlaps(x, age_range))
            
            print(f"DEBUG: Age matches found: {age_mask.sum()}")
            
            # Debug which books matched
            matched_books = df[age_mask][['title', 'age_range_llm']].head(10)
            print(f"DEBUG: Sample matched books:\n{matched_books}")
            
            mask = mask & age_mask
            print(f"DEBUG: After age filter: {mask.sum()} books")
        elif criteria['age_range'] and skip_additional_filters:
            print(f"DEBUG: Skipping age filter for direct search")
        # Tone filter (skip for direct searches)
        if criteria['tone'] and not skip_additional_filters:
            print(f"DEBUG: Applying tone filter for: {criteria['tone']}")
            
            # Check if the tone matches our tone group names
            tone_input = criteria['tone']
            potential_tone_group = None
            
            # Try to match to tone groups
            # Check both directions: group name in input AND input words in group name
            for group_name in TONE_GROUPS.keys():
                group_lower = group_name.lower()
                tone_lower = tone_input.lower()
                
                # Check if the full group name is in the input (e.g., "light & fun" in "light & fun bedtime")
                if group_lower in tone_lower:
                    potential_tone_group = group_name
                    print(f"DEBUG: Found tone group match (full): {group_name}")
                    break
                    
                # Check if any significant word from the input matches key words in group names
                tone_words = set(tone_lower.split())
                group_words = set(group_lower.split())
                
                # Remove common words that shouldn't trigger matches
                common_words = {'&', 'and', 'or', 'the', 'a', 'an'}
                significant_tone_words = tone_words - common_words
                significant_group_words = group_words - common_words
                
                # If any significant word from tone input is in the group name, it's a match
                if significant_tone_words & significant_group_words:
                    potential_tone_group = group_name
                    print(f"DEBUG: Found tone group match (partial): {group_name} via words {significant_tone_words & significant_group_words}")
                    break
            
            # Expand tone groups to individual tones
            if potential_tone_group:
                expanded_tones = expand_tone_groups([potential_tone_group])
                print(f"DEBUG: Using tone group expansion for {potential_tone_group}")
            else:
                expanded_tones = expand_tone_groups([criteria['tone']])
                print(f"DEBUG: Using individual tone")
                
            print(f"DEBUG: Expanded tones: {expanded_tones}")
            tone_mask = pd.Series([False] * len(df), index=df.index)
            for tone in expanded_tones:
                tone_match = df['tone'].fillna('').str.contains(
                    tone, case=False, na=False, regex=False
                )
                tone_mask = tone_mask | tone_match
                print(f"DEBUG: Tone '{tone}' found {tone_match.sum()} matches")
            mask = mask & tone_mask
            print(f"DEBUG: After tone filter: {mask.sum()} books")
        elif criteria['tone'] and skip_additional_filters:
            print(f"DEBUG: Skipping tone filter for direct search")
        
        # Lexile range filter (skip for direct searches)
        if criteria['lexile_range'] and not skip_additional_filters:
            min_lexile, max_lexile = criteria['lexile_range']
            print(f"DEBUG: Applying lexile filter for: {min_lexile}-{max_lexile}")
            
            # Use enriched Lexile scores when available
            if lexile_predictor and hasattr(lexile_predictor, 'enriched_scores'):
                print(f"DEBUG: Using enriched Lexile scores for filtering")
                lexile_mask = pd.Series([False] * len(df), index=df.index)
                books_checked = 0
                books_within_range = 0
                prediction_errors = 0
                
                for idx, row in df.iterrows():
                    books_checked += 1
                    try:
                        # Get enriched Lexile score from the predictor
                        prediction = lexile_predictor.predict(
                            title=row.get('title', ''),
                            author=row.get('author', '')
                        )
                        
                        if prediction and prediction.get('lexile_score') is not None:
                            lexile_score = prediction['lexile_score']
                            is_in_range = min_lexile <= lexile_score <= max_lexile
                            
                            # Enhanced debugging for problematic books
                            if books_within_range < 10:  # Only debug first 10 matches
                                print(f"DEBUG: Book '{row.get('title', 'N/A')}' - Predicted: {lexile_score}L, In range: {is_in_range}")
                            
                            if is_in_range:
                                lexile_mask.iloc[idx] = True
                                books_within_range += 1
                        else:
                            print(f"DEBUG: No valid prediction for '{row.get('title', 'N/A')}'")
                            
                    except Exception as e:
                        prediction_errors += 1
                        if prediction_errors < 5:  # Only show first 5 errors
                            print(f"DEBUG: Prediction error for '{row.get('title', 'N/A')}': {e}")
                
                print(f"DEBUG: Enriched Lexile - Checked: {books_checked}, In range: {books_within_range}, Errors: {prediction_errors}")
                
                # If we found very few matches, also try original lexile_score as backup
                if lexile_mask.sum() < 5:
                    print(f"DEBUG: Very few enriched matches ({lexile_mask.sum()}), trying original lexile_score as backup")
                    lexile_values = pd.to_numeric(df['lexile_score'], errors='coerce')
                    original_lexile_mask = (lexile_values >= min_lexile) & (lexile_values <= max_lexile)
                    print(f"DEBUG: Original Lexile would have: {original_lexile_mask.sum()} matches")
                    
                    # For now, stick with enriched but add this debug info
                    # lexile_mask = lexile_mask | original_lexile_mask  # Uncomment to use backup
                
            else:
                # Fallback to original lexile_score column
                print(f"DEBUG: Using original lexile_score column for filtering")
                lexile_values = pd.to_numeric(df['lexile_score'], errors='coerce')
                lexile_mask = (lexile_values >= min_lexile) & (lexile_values <= max_lexile)
                print(f"DEBUG: Original Lexile matches found: {lexile_mask.sum()}")
            
            mask = mask & lexile_mask
            print(f"DEBUG: After lexile filter: {mask.sum()} books")
        elif criteria['lexile_range'] and skip_additional_filters:
            print(f"DEBUG: Skipping lexile filter for direct search")

        # Get all results first
        all_results = df[mask]
        
        # Filter out skipped and read books
        all_results = filter_user_books(all_results)
        
        # Sort lexile range results appropriately
        if criteria.get('lexile_range') and not all_results.empty:
            try:
                # Determine difficulty level based on the difficulty_level parameter passed from frontend
                is_challenge = difficulty_level == 'advanced'
                is_easier = difficulty_level == 'lower'
                
                # Get lexile values for sorting
                if lexile_predictor and hasattr(lexile_predictor, 'enriched_scores'):
                    # Use enriched scores for sorting if available
                    lexile_sort_values = []
                    for _, row in all_results.iterrows():
                        try:
                            prediction = lexile_predictor.predict(
                                title=row.get('title', ''),
                                author=row.get('author', '')
                            )
                            if prediction and prediction.get('lexile_score') is not None:
                                lexile_sort_values.append(prediction['lexile_score'])
                            else:
                                # Fallback to original score
                                lexile_sort_values.append(pd.to_numeric(row.get('lexile_score', 0), errors='coerce'))
                        except:
                            lexile_sort_values.append(pd.to_numeric(row.get('lexile_score', 0), errors='coerce'))
                    
                    # Add lexile values as temporary column for sorting
                    all_results = all_results.copy()
                    all_results['temp_sort_lexile'] = lexile_sort_values
                else:
                    # Use original lexile scores
                    all_results = all_results.copy()
                    all_results['temp_sort_lexile'] = pd.to_numeric(all_results['lexile_score'], errors='coerce')
                
                # Sort based on difficulty level requested
                if is_challenge:
                    # For challenge, sort by lexile descending (hardest first)
                    all_results = all_results.sort_values('temp_sort_lexile', ascending=False)
                    print(f"DEBUG: Sorted challenge results by lexile descending")
                elif is_easier:
                    # For easier, sort by lexile ascending (easiest first)
                    all_results = all_results.sort_values('temp_sort_lexile', ascending=True)
                    print(f"DEBUG: Sorted easier results by lexile ascending")
                else:
                    # For similar level, sort by proximity to base lexile (if we can determine it)
                    # For now, just sort ascending
                    all_results = all_results.sort_values('temp_sort_lexile', ascending=True)
                    print(f"DEBUG: Sorted similar level results by lexile ascending")
                
                # Remove temporary column
                all_results = all_results.drop('temp_sort_lexile', axis=1)
                
                print(f"DEBUG: First few lexile values after sorting:")
                for i, (_, row) in enumerate(all_results.head().iterrows()):
                    if lexile_predictor and hasattr(lexile_predictor, 'enriched_scores'):
                        try:
                            prediction = lexile_predictor.predict(
                                title=row.get('title', ''),
                                author=row.get('author', '')
                            )
                            if prediction and prediction.get('lexile_score') is not None:
                                sort_lexile = prediction['lexile_score']
                            else:
                                sort_lexile = row.get('lexile_score', 'N/A')
                        except:
                            sort_lexile = row.get('lexile_score', 'N/A')
                    else:
                        sort_lexile = row.get('lexile_score', 'N/A')
                    print(f"  {i+1}. {row.get('title', 'N/A')} - {sort_lexile}L")
                
            except Exception as e:
                print(f"DEBUG: Error sorting lexile results: {e}")
                # Continue without sorting if there's an error
        
        total_results = len(all_results)
        
        
        # Calculate pagination
        total_pages = (total_results + per_page - 1) // per_page  # Ceiling division
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page

        
        # Get results for current page
        page_results = all_results.iloc[start_idx:end_idx]
        print(f"DEBUG: Pagination - start_idx: {start_idx}, end_idx: {end_idx}")
        print(f"DEBUG: page_results shape: {page_results.shape}")
        
        print("DEBUG: Books being returned:")
        for i, (_, row) in enumerate(page_results.iterrows()):
            print(f"  Book {i}: {row.get('title', 'N/A')} - Age: {row.get('age_range_llm', 'N/A')}")
        print(f"DEBUG: Total results: {total_results}, Page: {page}/{total_pages}")
        
        # Debug: Check DataFrame columns and sample data
        print(f"DEBUG: DataFrame columns: {list(page_results.columns)}")
        if 'goodreads_url' in page_results.columns:
            print("DEBUG: goodreads_url column found in DataFrame")
            sample_goodreads = page_results['goodreads_url'].head(3).tolist()
            print(f"DEBUG: Sample goodreads_url values: {sample_goodreads}")
        else:
            print("DEBUG: goodreads_url column NOT found in DataFrame")
        
        # Convert to response format
        books = []
        for _, row in page_results.iterrows():
            try:  # Add error handling here
                cover_url = str(row.get('cover_url', ''))
                goodreads_url = str(row.get('goodreads_url', ''))
                
                # Get the appropriate lexile score - use enriched when available for lexile searches
                lexile_display_score = str(row.get('lexile_score', ''))  # Default to original
                
                if criteria.get('lexile_range') and lexile_predictor and hasattr(lexile_predictor, 'enriched_scores'):
                    # For lexile range searches, show the enriched score that was used for filtering
                    try:
                        prediction = lexile_predictor.predict(
                            title=row.get('title', ''),
                            author=row.get('author', '')
                        )
                        if prediction and prediction.get('lexile_score') is not None:
                            lexile_display_score = str(int(prediction['lexile_score']))
                            print(f"DEBUG: Using enriched score for '{row.get('title', 'N/A')}': {lexile_display_score}L")
                    except Exception as e:
                        print(f"DEBUG: Failed to get enriched score for '{row.get('title', 'N/A')}': {e}")
                        # Keep original score as fallback
                
                # Debug: print goodreads_url for each book
                print(f"DEBUG: Book '{row.get('title', 'N/A')}' - Raw goodreads_url: '{row.get('goodreads_url', 'MISSING')}' - Processed: '{goodreads_url}'")
                book = {
                    'title': str(row.get('title', '')),
                    'author': str(row.get('author', '')),
                    'themes': str(row.get('themes', '')),
                    'age_range': str(row.get('age_range_llm', '')),
                    'lexile_score': lexile_display_score,
                    'cover_url': cover_url if cover_url and cover_url != 'nan' else '',
                    'tone': str(row.get('tone', '')),
                    'summary_gpt': str(row.get('summary_gpt', '')) if row.get('summary_gpt') and str(row.get('summary_gpt')) != 'nan' else '',
                    'goodreads_url': goodreads_url if goodreads_url and goodreads_url != 'nan' else ''
                }
                books.append(book)
                # Debug: print Goodreads URL for this book
                if "mouse" in book.get('title', '').lower():
                    print(f"DEBUG: Mouse book - Title: {book.get('title')}, Goodreads URL: '{book.get('goodreads_url')}'")
            except Exception as e:
                print(f"DEBUG: Error processing book row: {e}")
                print(f"DEBUG: Row data: {row}")
                continue  # Skip this book and continue with others
        

        # In your /api/simple-search endpoint, add these debug prints:
        print(f"DEBUG: Query: '{query}'")
        print(f"DEBUG: Parsed criteria: {criteria}")
        print(f"DEBUG: Themes: {criteria.get('themes', [])}")
        print(f"DEBUG: Age range: {criteria.get('age_range')}")
        print(f"DEBUG: Tone: {criteria.get('tone')}")

        return jsonify({
            'success': True,
            'results': books,
            'pagination': {
                'current_page': page,
                'total_pages': total_pages,
                'total_results': total_results,
                'per_page': per_page,
                'has_next': page < total_pages,
                'has_prev': page > 1
            },
            'parsed_criteria': criteria,
            'parsing_method': criteria.get('parsing_method', 'hybrid'),
            'query_searched': query
        })
        
    except Exception as e:
        print(f"DEBUG: Search endpoint error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/test-parser', methods=['POST'])
def test_parser():
    data = request.get_json()
    query = data.get('query', '')
    criteria = simple_parse_query(query)
    return jsonify({
        'query': query,
        'parsed_criteria': criteria
    })

@app.route('/api/test-data', methods=['GET'])
def test_data():
    """Test endpoint to verify data loading"""
    try:
        if not recommendation_engine:
            return jsonify({"error": "No recommendation engine"})
        
        if recommendation_engine.catalog_df is None:
            return jsonify({"error": "No catalog data"})
        
        df = recommendation_engine.catalog_df
        
        # Test basic data
        sample_data = {
            "total_books": len(df),
            "columns": df.columns.tolist(),
            "sample_titles": df['title'].head(5).tolist() if 'title' in df.columns else [],
            "sample_authors": df['author'].head(5).tolist() if 'author' in df.columns else [],
            "sample_themes": df['themes'].head(5).tolist() if 'themes' in df.columns else [],
        }
        
        # Test a simple search
        test_query = "magic"
        if 'themes' in df.columns:
            magic_books = df[df['themes'].fillna('').str.contains('magic', case=False, na=False)]
            sample_data["magic_books_found"] = len(magic_books)
            sample_data["sample_magic_titles"] = magic_books['title'].head(3).tolist() if len(magic_books) > 0 else []
        
        return jsonify({"success": True, "data": sample_data})
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        })

@app.route('/api/book-suggestions', methods=['GET'])
def get_book_suggestions():
    """Get book suggestions for autocomplete in AI Book Analysis"""
    try:
        query = request.args.get('q', '').strip()
        if not query or len(query) < 2:
            return jsonify({'success': True, 'suggestions': []})
        
        if not recommendation_engine or recommendation_engine.catalog_df is None:
            return jsonify({'success': False, 'error': 'No data available'})
        
        df = recommendation_engine.catalog_df
        query_lower = query.lower()
        
        # Search in titles and authors
        title_matches = df[
            df['title'].fillna('').str.lower().str.contains(query_lower, na=False, regex=False)
        ].head(10)
        
        author_matches = df[
            df['author'].fillna('').str.lower().str.contains(query_lower, na=False, regex=False)
        ].head(10)
        
        # Combine and deduplicate
        all_matches = pd.concat([title_matches, author_matches]).drop_duplicates(subset=['title', 'author'])
        
        suggestions = []
        for _, book in all_matches.head(15).iterrows():
            cover_url = str(book.get('cover_url', ''))
            suggestions.append({
                'title': str(book.get('title', '')),
                'author': str(book.get('author', '')),
                'themes': str(book.get('themes', '')),
                'age_range': str(book.get('age_range_llm', '')),
                'lexile_score': str(book.get('lexile_score', '')),
                'cover_url': cover_url if cover_url and cover_url != 'nan' else ''
            })
        
        return jsonify({'success': True, 'suggestions': suggestions})
        
    except Exception as e:
        logger.error(f"Book suggestions error: {e}")
        return jsonify({'success': False, 'error': 'Failed to get suggestions'})

@app.route('/api/proxy-image')
def proxy_image():
    """Proxy endpoint to handle OpenLibrary image redirects"""
    import requests
    
    url = request.args.get('url')
    if not url:
        return 'Missing URL parameter', 400
        
    try:
        # Follow redirects and get the final image
        response = requests.get(url, allow_redirects=True, timeout=10, 
                              headers={'User-Agent': 'KidLit Book Curator'})
        
        if response.status_code == 200 and response.headers.get('content-type', '').startswith('image/'):
            return response.content, 200, {
                'Content-Type': response.headers.get('content-type', 'image/jpeg'),
                'Cache-Control': 'public, max-age=3600'  # Cache for 1 hour
            }
        else:
            return 'Image not found', 404
            
    except Exception as e:
        logger.error(f"Image proxy error: {e}")
        return 'Image load failed', 500

# =============================================
# NEW: Lexile Prediction API Endpoints
# =============================================

@app.route('/api/predict-lexile', methods=['POST'])
def predict_lexile():
    """
    Enhanced Lexile prediction with confidence levels and edge case warnings
    
    Expected JSON payload:
    {
        "title": "Book Title",
        "author": "Author Name", 
        "age_min": 3,
        "age_max": 7,
        "book_type": "Standard_Lexile", // or "Adult_Directed", "Graphic_Novel"
        "notes": "Additional book description"
    }
    """
    try:
        data = request.json
        
        if not data or 'title' not in data:
            return jsonify({
                'error': 'Missing required field: title'
            }), 400
        
        # Extract parameters
        title = data.get('title', '')
        author = data.get('author', '')
        age_min = data.get('age_min')
        age_max = data.get('age_max') 
        book_type = data.get('book_type', 'Standard_Lexile')
        notes = data.get('notes', '')
        
        if lexile_predictor is None:
            # Fallback response when predictor is not available
            return jsonify({
                'title': title,
                'predicted_lexile': None,
                'confidence_level': 'unavailable',
                'error': 'Lexile predictor not available',
                'fallback': True
            })
        
        # FIRST: Check if the book exists in our catalog with a known Lexile score
        catalog_score = None
        if recommendation_engine and recommendation_engine.catalog_df is not None:
            df = recommendation_engine.catalog_df
            # Look for exact title and author match
            title_match = df['title'].str.strip().str.lower() == title.lower().strip()
            author_match = df['author'].str.contains(author, case=False, na=False) if author else pd.Series([True] * len(df))
            
            matches = df[title_match & author_match]
            if len(matches) > 0 and 'lexile_score' in matches.columns:
                catalog_score = matches.iloc[0]['lexile_score']
                if pd.notna(catalog_score) and catalog_score > 0:
                    logger.info(f"Found catalog Lexile score for '{title}': {catalog_score}L")
                    return jsonify({
                        'title': title,
                        'author': author,
                        'predicted_lexile': float(catalog_score),
                        'confidence': 0.95,  # High confidence for catalog data
                        'source': 'catalog',
                        'enrichment_source': 'catalog_database',
                        'confidence_level': 'high',
                        'prediction_method': 'catalog_lookup',
                        'success': True,
                        'timestamp': datetime.now().isoformat()
                    })
        
        # SECOND: Check if using EnrichedLexilePredictor (new API) or ProductionLexilePredictor (old API)
        if hasattr(lexile_predictor, 'predict') and hasattr(lexile_predictor, 'enriched_scores'):
            # New EnrichedLexilePredictor API
            prediction = lexile_predictor.predict(title=title, author=author)
            
            result = {
                'title': title,
                'author': author,
                'predicted_lexile': prediction['lexile_score'],
                'confidence': prediction['confidence'],
                'source': prediction['source'],
                'enrichment_source': prediction['enrichment_source'],
                'confidence_level': prediction['confidence_level'],
                'prediction_method': prediction['method'],
                'success': True,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Enhanced Lexile prediction: {title} -> {prediction['lexile_score']}L "
                       f"(source: {prediction['source']}, confidence: {prediction['confidence']:.2f})")
        else:
            # Legacy ProductionLexilePredictor API
            result = lexile_predictor.predict_lexile(
                title=title,
                author=author,
                age_min=age_min,
                age_max=age_max,
                book_type=book_type,
                notes=notes
            )
            
            # Add success indicators
            result['success'] = True
            result['timestamp'] = datetime.now().isoformat()
            
            logger.info(f"Legacy Lexile prediction: {title} -> {result.get('predicted_lexile')}L "
                       f"(confidence: {result.get('confidence_level')})")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Lexile prediction error: {e}")
        return jsonify({
            'error': 'Lexile prediction failed',
            'message': str(e),
            'success': False
        }), 500

@app.route('/api/batch-predict-lexile', methods=['POST'])
def batch_predict_lexile():
    """
    Batch Lexile prediction for multiple books
    
    Expected JSON payload:
    {
        "books": [
            {
                "title": "Book 1",
                "author": "Author 1",
                "age_min": 3,
                "age_max": 7,
                "book_type": "Standard_Lexile"
            },
            // ... more books
        ]
    }
    """
    try:
        data = request.json
        
        if not data or 'books' not in data:
            return jsonify({
                'error': 'Missing required field: books (array)'
            }), 400
        
        books = data.get('books', [])
        
        if len(books) > 50:  # Limit batch size
            return jsonify({
                'error': 'Batch size too large (max 50 books)'
            }), 400
        
        if lexile_predictor is None:
            return jsonify({
                'error': 'Lexile predictor not available',
                'results': [],
                'success': False
            })
        
        # Process each book
        results = []
        successful_predictions = 0
        
        for book in books:
            try:
                if 'title' not in book:
                    results.append({
                        'error': 'Missing title',
                        'success': False
                    })
                    continue
                
                # Check if using EnrichedLexilePredictor (new API) or ProductionLexilePredictor (old API)
                if hasattr(lexile_predictor, 'predict') and hasattr(lexile_predictor, 'enriched_scores'):
                    # New EnrichedLexilePredictor API
                    prediction = lexile_predictor.predict(
                        title=book.get('title', ''),
                        author=book.get('author', '')
                    )
                    
                    result = {
                        'title': book.get('title', ''),
                        'author': book.get('author', ''),
                        'predicted_lexile': prediction['lexile_score'],
                        'confidence': prediction['confidence'],
                        'source': prediction['source'],
                        'enrichment_source': prediction['enrichment_source'],
                        'confidence_level': prediction['confidence_level'],
                        'prediction_method': prediction['method'],
                        'success': True
                    }
                else:
                    # Legacy ProductionLexilePredictor API
                    result = lexile_predictor.predict_lexile(
                        title=book.get('title', ''),
                        author=book.get('author', ''),
                        age_min=book.get('age_min'),
                        age_max=book.get('age_max'),
                        book_type=book.get('book_type', 'Standard_Lexile'),
                        notes=book.get('notes', '')
                    )
                    result['success'] = True
                
                results.append(result)
                successful_predictions += 1
                
            except Exception as e:
                results.append({
                    'title': book.get('title', 'Unknown'),
                    'error': str(e),
                    'success': False
                })
        
        return jsonify({
            'results': results,
            'summary': {
                'total_books': len(books),
                'successful_predictions': successful_predictions,
                'failed_predictions': len(books) - successful_predictions
            },
            'timestamp': datetime.now().isoformat(),
            'success': True
        })
        
    except Exception as e:
        logger.error(f"Batch Lexile prediction error: {e}")
        return jsonify({
            'error': 'Batch prediction failed',
            'message': str(e),
            'success': False
        }), 500

@app.route('/api/lexile-model-info', methods=['GET'])
def lexile_model_info():
    """Get information about the Lexile prediction model"""
    
    model_info = {
        'model_available': lexile_predictor is not None,
        'model_version': 'Extreme Sophistication v1.0',
        'training_date': '2025-09-03',
        'performance_metrics': {
            'overall_mae': '85.6L',
            'adult_directed_mae': '5.6L',
            'r_squared': 0.529,
            'excellent_predictions_pct': 46.2
        },
        'supported_book_types': [
            'Standard_Lexile',
            'Adult_Directed', 
            'Graphic_Novel'
        ],
        'confidence_levels': {
            'high': 'Expected error ±10-50L (Adult Directed, early readers)',
            'medium': 'Expected error ±85L (Standard cases)',
            'low': 'Expected error ±150-330L (Pre-1950 vintage classics)'
        },
        'known_limitations': [
            'Pre-1950 books may have elevated prediction uncertainty',
            'Extreme vintage classics are automatically flagged',
            'Historical language patterns can affect accuracy'
        ]
    }
    
    if lexile_predictor is not None:
        model_info['models_loaded'] = list(lexile_predictor.models.keys()) if hasattr(lexile_predictor, 'models') else []
        model_info['extreme_classics_count'] = len(lexile_predictor.extreme_vintage_classics) if hasattr(lexile_predictor, 'extreme_vintage_classics') else 0
    
    return jsonify(model_info)

@app.route('/api/filter-options', methods=['GET'])
def get_filter_options():
    """Get available filter options based on current selections"""
    try:
        if not recommendation_engine or recommendation_engine.catalog_df is None:
            return jsonify({'success': False, 'error': 'No data available'})
        
        df = recommendation_engine.catalog_df
        
        # Get query parameters for current selections
        selected_theme = request.args.get('theme', '').strip()
        selected_tone = request.args.get('tone', '').strip()
        selected_age = request.args.get('age', '').strip()
        selected_lexile = request.args.get('lexile', '').strip()
        
        # Start with full dataset and filter progressively
        filtered_df = df.copy()
        
        # Apply current filters
        if selected_theme:
            # Map frontend theme keys to searchable keywords
            theme_keywords = {
                'friendship': 'friendship',
                'adventure': 'adventure',
                'family': 'family',
                'courage': 'courage',
                'magic': 'magic',
                'school': 'school',
                'animals': 'animals',
                'humor': 'humor',
                'identity': 'identity',
                'emotions': 'emotions'
            }
            search_keyword = theme_keywords.get(selected_theme, selected_theme)
            theme_filter = filtered_df['themes'].fillna('').str.contains(search_keyword, case=False, na=False)
            filtered_df = filtered_df[theme_filter]
        
        if selected_tone:
            # Map frontend tone keys to searchable keywords  
            tone_keywords = {
                'whimsical': 'whimsical|playful',
                'heartwarming': 'heartwarming|warm',
                'adventurous': 'adventurous|exciting',
                'gentle': 'gentle|calm',
                'humorous': 'humorous|funny|humor',
                'inspiring': 'inspiring|uplifting',
                'mysterious': 'mysterious|mystery',
                'educational': 'educational|informative'
            }
            search_pattern = tone_keywords.get(selected_tone, selected_tone)
            tone_filter = filtered_df['tone'].fillna('').str.contains(search_pattern, case=False, na=False, regex=True)
            filtered_df = filtered_df[tone_filter]
        
        if selected_age:
            try:
                # Handle consolidated age groups
                def get_age_group(age_range):
                    if pd.isna(age_range):
                        return None
                    age_str = str(age_range).lower()
                    
                    # Extract start age for classification
                    start_age = None
                    try:
                        if '+' in age_str:
                            start_age = int(age_str.replace('+', ''))
                        elif '-' in age_str:
                            start_age = int(age_str.split('-')[0])
                        else:
                            start_age = int(''.join(filter(str.isdigit, age_str)))
                    except:
                        return None
                    
                    # Consolidate into 4 main groups
                    if start_age <= 5:
                        return "Ages 0-5"
                    elif start_age <= 8:
                        return "Ages 6-8" 
                    elif start_age <= 12:
                        return "Ages 9-12"
                    else:
                        return "Ages 13+"
                
                def age_matches_group(age_range_str):
                    if not age_range_str or age_range_str == '' or str(age_range_str).lower() == 'nan':
                        return False
                    book_group = get_age_group(age_range_str)
                    return book_group == selected_age
                
                age_filter = filtered_df['age_range_llm'].apply(age_matches_group)
                filtered_df = filtered_df[age_filter]
            except Exception as e:
                print(f"Age filtering error: {e}")
                pass
        
        if selected_lexile:
            if selected_lexile == '1000+':
                lexile_filter = filtered_df['lexile_score'] >= 1000
            else:
                try:
                    lexile_min, lexile_max = map(int, selected_lexile.split('-'))
                    lexile_filter = ((filtered_df['lexile_score'] >= lexile_min) & (filtered_df['lexile_score'] <= lexile_max))
                except:
                    lexile_filter = filtered_df['lexile_score'].notna()
            filtered_df = filtered_df[lexile_filter]
        
        # Extract available options from filtered dataset
        available_options = {}
        
        # Get available themes
        if not selected_theme:
            theme_map = {
                'friendship': 'Friendship',
                'adventure': 'Adventure & Exploration', 
                'family': 'Family',
                'courage': 'Courage & Bravery',
                'magic': 'Fantasy & Magic',
                'school': 'School & Learning',
                'animals': 'Animals & Nature',
                'humor': 'Humor & Fun',
                'identity': 'Identity & Self-Discovery',
                'emotions': 'Emotions & Feelings'
            }
            available_themes = []
            for key, label in theme_map.items():
                if len(filtered_df[filtered_df['themes'].fillna('').str.contains(key, case=False, na=False)]) > 0:
                    available_themes.append({'value': key, 'label': label})
            available_options['themes'] = available_themes
        
        # Get available tones
        if not selected_tone:
            tone_map = {
                'whimsical': ('whimsical|playful', 'Whimsical & Playful'),
                'heartwarming': ('heartwarming|warm', 'Heartwarming & Sweet'),
                'adventurous': ('adventurous|exciting', 'Adventurous & Exciting'),
                'gentle': ('gentle|calm', 'Gentle & Calm'),
                'humorous': ('humorous|funny|humor', 'Humorous & Fun'),
                'inspiring': ('inspiring|uplifting', 'Inspiring & Uplifting'),
                'mysterious': ('mysterious|mystery', 'Mysterious & Intriguing'),
                'educational': ('educational|informative', 'Educational & Informative')
            }
            available_tones = []
            for key, (pattern, label) in tone_map.items():
                if len(filtered_df[filtered_df['tone'].fillna('').str.contains(pattern, case=False, na=False, regex=True)]) > 0:
                    available_tones.append({'value': key, 'label': label})
            available_options['tones'] = available_tones
        
        # Get available age ranges
        if not selected_age:
            age_ranges = ['0-2', '2-4', '3-5', '4-6', '5-7', '6-8', '8-10', '9-12', '10-14', '12-16']
            available_ages = []
            for age_range in age_ranges:
                try:
                    target_age_min, target_age_max = parse_age_span(age_range)
                    
                    def age_overlaps(age_range_str):
                        if not age_range_str or age_range_str == '' or str(age_range_str).lower() == 'nan':
                            return False
                        try:
                            book_age_min, book_age_max = parse_age_span(str(age_range_str))
                            return not (book_age_max < target_age_min or book_age_min > target_age_max)
                        except:
                            return False
                    
                    age_filter = filtered_df['age_range_llm'].apply(age_overlaps)
                    if len(filtered_df[age_filter]) > 0:
                        available_ages.append({'value': age_range, 'label': f'{age_range} years'})
                except Exception as e:
                    print(f"Age availability check error for {age_range}: {e}")
                    pass
            available_options['ages'] = available_ages
        
        # Get available lexile ranges
        if not selected_lexile:
            lexile_ranges = [
                {'value': '0-200', 'label': '0-200L (Beginning)'},
                {'value': '200-400', 'label': '200-400L (Early)'},
                {'value': '400-600', 'label': '400-600L (Developing)'},
                {'value': '600-800', 'label': '600-800L (Intermediate)'},
                {'value': '800-1000', 'label': '800-1000L (Advanced)'},
                {'value': '1000+', 'label': '1000L+ (Expert)'},
                {'value': 'AD', 'label': 'AD (Adult Directed)'}
            ]
            available_lexiles = []
            for lexile_range in lexile_ranges:
                value = lexile_range['value']
                if value == '1000+':
                    lexile_filter = filtered_df['lexile_score'] >= 1000
                elif value == 'AD':
                    lexile_filter = filtered_df['lexile_score'].isna()
                else:
                    try:
                        lexile_min, lexile_max = map(int, value.split('-'))
                        lexile_filter = ((filtered_df['lexile_score'] >= lexile_min) & (filtered_df['lexile_score'] <= lexile_max))
                    except:
                        continue
                
                if len(filtered_df[lexile_filter]) > 0:
                    available_lexiles.append(lexile_range)
            available_options['lexiles'] = available_lexiles
        
        return jsonify({
            'success': True,
            'available_options': available_options,
            'total_books_available': len(filtered_df)
        })
        
    except Exception as e:
        print(f"Filter options error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/lexile-distribution', methods=['GET'])
def get_lexile_distribution():
    """Get lexile score distribution for chart visualization"""
    try:
        if not recommendation_engine or recommendation_engine.catalog_df is None:
            return jsonify({'success': False, 'error': 'Catalog not loaded'})
        
        df = recommendation_engine.catalog_df
        
        # Define lexile ranges for distribution
        ranges = [
            {'label': '0-200L', 'min': 0, 'max': 200},
            {'label': '200-400L', 'min': 200, 'max': 400},
            {'label': '400-600L', 'min': 400, 'max': 600},
            {'label': '600-800L', 'min': 600, 'max': 800},
            {'label': '800-1000L', 'min': 800, 'max': 1000},
            {'label': '1000L+', 'min': 1000, 'max': float('inf')}
        ]
        
        # Count books in each range
        distribution = []
        # Convert lexile_score to numeric, filtering out non-numeric values
        df['lexile_score_numeric'] = pd.to_numeric(df['lexile_score'], errors='coerce')
        valid_lexile_scores = df[df['lexile_score_numeric'].notna()]
        
        for range_info in ranges:
            if range_info['max'] == float('inf'):
                count = len(valid_lexile_scores[valid_lexile_scores['lexile_score_numeric'] >= range_info['min']])
            else:
                count = len(valid_lexile_scores[
                    (valid_lexile_scores['lexile_score_numeric'] >= range_info['min']) & 
                    (valid_lexile_scores['lexile_score_numeric'] < range_info['max'])
                ])
            
            distribution.append({
                'label': range_info['label'],
                'count': count
            })
        
        # Only include ranges with books
        distribution = [item for item in distribution if item['count'] > 0]
        
        return jsonify({
            'success': True,
            'distribution': distribution,
            'total_books': len(valid_lexile_scores),
            'total_catalog': len(df)
        })
        
    except Exception as e:
        print(f"Lexile distribution error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/adaptive-search-options', methods=['POST'])
def get_adaptive_search_options():
    """Get available search options based on current selections"""
    try:
        if not recommendation_engine or recommendation_engine.catalog_df is None:
            return jsonify({'success': False, 'error': 'Catalog not loaded'})
        
        data = request.get_json()
        theme = data.get('theme')
        tone = data.get('tone')
        age = data.get('age')
        lexile = data.get('lexile')
        
        df = recommendation_engine.catalog_df.copy()
        
        # Apply current filters to get subset of books
        filtered_df = df
        
        if theme:
            # Expand theme groups properly
            if theme in THEME_GROUPS:
                expanded_themes = THEME_GROUPS[theme]
                theme_mask = pd.Series([False] * len(df), index=df.index)
                for theme_term in expanded_themes:
                    theme_match = (
                        df['themes'].str.contains(theme_term, case=False, na=False, regex=False)
                    )
                    theme_mask = theme_mask | theme_match
                filtered_df = filtered_df[theme_mask]
            else:
                # Single theme search
                theme_mask = (
                    df['themes'].str.contains(theme, case=False, na=False)
                )
                filtered_df = filtered_df[theme_mask]
        
        if tone:
            # Expand tone groups properly
            if tone in TONE_GROUPS:
                expanded_tones = TONE_GROUPS[tone]
                tone_mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
                for tone_term in expanded_tones:
                    tone_match = (
                        filtered_df['themes'].str.contains(tone_term, case=False, na=False, regex=False) |
                        filtered_df['tone'].str.contains(tone_term, case=False, na=False, regex=False)
                    )
                    tone_mask = tone_mask | tone_match
                filtered_df = filtered_df[tone_mask]
            else:
                # Single tone search
                tone_mask = (
                    filtered_df['themes'].str.contains(tone, case=False, na=False) |
                    filtered_df['tone'].str.contains(tone, case=False, na=False)
                )
                filtered_df = filtered_df[tone_mask]
        
        if age:
            # Filter by age using consolidated groups
            def get_age_group(age_range):
                if pd.isna(age_range):
                    return None
                age_str = str(age_range).lower()
                
                # Extract start age for classification
                start_age = None
                try:
                    if '+' in age_str:
                        start_age = int(age_str.replace('+', ''))
                    elif '-' in age_str:
                        start_age = int(age_str.split('-')[0])
                    else:
                        start_age = int(''.join(filter(str.isdigit, age_str)))
                except:
                    return None
                
                # Consolidate into 4 main groups
                if start_age <= 5:
                    return "Ages 0-5"
                elif start_age <= 8:
                    return "Ages 6-8" 
                elif start_age <= 12:
                    return "Ages 9-12"
                else:
                    return "Ages 13+"
            
            def age_matches_group(age_range_str):
                if not age_range_str or age_range_str == '' or str(age_range_str).lower() == 'nan':
                    return False
                book_group = get_age_group(age_range_str)
                return book_group == age
            
            age_filter = filtered_df['age_range_llm'].apply(age_matches_group)
            filtered_df = filtered_df[age_filter]
        
        if lexile:
            # Filter by lexile level
            if lexile == 'AD':
                filtered_df = filtered_df[filtered_df['lexile_score'] == 'AD']
            elif lexile == '1000+':
                lexile_numeric = pd.to_numeric(filtered_df['lexile_score'], errors='coerce')
                filtered_df = filtered_df[lexile_numeric >= 1000]
            else:
                # Handle ranges like "200-400"
                if '-' in lexile:
                    min_val, max_val = map(int, lexile.split('-'))
                    lexile_numeric = pd.to_numeric(filtered_df['lexile_score'], errors='coerce')
                    filtered_df = filtered_df[
                        (lexile_numeric >= min_val) & (lexile_numeric < max_val)
                    ]
        
        # Now get available options from the filtered dataset
        available_options = {}
        
        # Get available themes (if not already selected)
        if not theme:
            # Filter to only include books that have valid age data (to ensure accurate counts)
            books_with_age_data = filtered_df[filtered_df['age_range_llm'].notna() & (filtered_df['age_range_llm'] != '')]
            
            all_themes = []
            for col in ['themes']:
                themes_series = books_with_age_data[col].dropna()
                for themes_str in themes_series:
                    if isinstance(themes_str, str):
                        themes = [t.strip() for t in themes_str.split(',')]
                        all_themes.extend(themes)
            
            theme_counts = pd.Series(all_themes).value_counts()
            # Filter to most relevant children's book themes
            popular_themes = [
                'friendship', 'family', 'imagination', 'adventure', 'creativity', 
                'animals', 'love', 'acceptance', 'curiosity', 'community',
                'problem-solving', 'courage', 'self-discovery', 'learning',
                'nature', 'magic', 'growing up', 'identity'
            ]
            
            available_options['themes'] = [
                {'value': theme, 'label': theme.title(), 'count': int(count)}
                for theme, count in theme_counts.head(30).items()
                if theme and len(theme) > 2 and any(popular in theme.lower() for popular in popular_themes)
            ][:10]  # Limit to top 10 relevant themes
        
        # Get available tones (if not already selected)
        if not tone:
            available_tones = []
            
            # Filter to only include books that have valid age data (to ensure accurate counts)
            books_with_age_data = filtered_df[filtered_df['age_range_llm'].notna() & (filtered_df['age_range_llm'] != '')]
            
            # Get all unique tones from the actual data
            all_tones = []
            for tone_str in books_with_age_data['tone'].dropna():
                if pd.notna(tone_str) and str(tone_str).strip():
                    # Split by comma and clean up
                    tones = [t.strip() for t in str(tone_str).split(',') if t.strip()]
                    all_tones.extend(tones)
            
            # Count frequency of each tone
            from collections import Counter
            tone_counts = Counter(all_tones)
            
            # Create tone options from most frequent tones
            for tone_name, count in tone_counts.most_common(10):  # Show top 10 most common tones
                if count > 0 and tone_name:  # Ensure valid tone with books
                    available_tones.append({
                        'value': tone_name, 
                        'label': tone_name.title(), 
                        'count': count
                    })
            
            available_options['tones'] = available_tones
        
        # Get available ages (if not already selected)
        if not age:
            # Define consolidated age groups that map individual age ranges
            def get_age_group(age_range):
                if pd.isna(age_range):
                    return None
                age_str = str(age_range).lower()
                
                # Extract start age for classification
                start_age = None
                try:
                    if '+' in age_str:
                        start_age = int(age_str.replace('+', ''))
                    elif '-' in age_str:
                        start_age = int(age_str.split('-')[0])
                    else:
                        start_age = int(''.join(filter(str.isdigit, age_str)))
                except:
                    return None
                
                # Consolidate into 4 main groups
                if start_age <= 5:
                    return "Ages 0-5"
                elif start_age <= 8:
                    return "Ages 6-8" 
                elif start_age <= 12:
                    return "Ages 9-12"
                else:
                    return "Ages 13+"
            
            # Count books in each consolidated group
            age_groups = {}
            for age_range in filtered_df['age_range_llm']:
                group = get_age_group(age_range)
                if group:
                    age_groups[group] = age_groups.get(group, 0) + 1
            
            # Create consolidated age options
            age_options = []
            group_order = ["Ages 0-5", "Ages 6-8", "Ages 9-12", "Ages 13+"]
            for group in group_order:
                if group in age_groups and age_groups[group] > 0:
                    age_options.append({
                        'value': group, 
                        'label': group, 
                        'count': int(age_groups[group])
                    })
            
            available_options['ages'] = age_options
        
        # Get available lexile levels (if not already selected)
        if not lexile:
            # Define lexile ranges
            ranges = [
                {'label': '0-200L', 'value': '0-200', 'min': 0, 'max': 200},
                {'label': '200-400L', 'value': '200-400', 'min': 200, 'max': 400},
                {'label': '400-600L', 'value': '400-600', 'min': 400, 'max': 600},
                {'label': '600-800L', 'value': '600-800', 'min': 600, 'max': 800},
                {'label': '800-1000L', 'value': '800-1000', 'min': 800, 'max': 1000},
                {'label': '1000L+', 'value': '1000+', 'min': 1000, 'max': float('inf')}
            ]
            
            # Convert lexile_score to numeric
            filtered_df['lexile_score_numeric'] = pd.to_numeric(filtered_df['lexile_score'], errors='coerce')
            valid_lexile_df = filtered_df[filtered_df['lexile_score_numeric'].notna()]
            
            available_lexile = []
            for range_info in ranges:
                if range_info['max'] == float('inf'):
                    count = len(valid_lexile_df[valid_lexile_df['lexile_score_numeric'] >= range_info['min']])
                else:
                    count = len(valid_lexile_df[
                        (valid_lexile_df['lexile_score_numeric'] >= range_info['min']) & 
                        (valid_lexile_df['lexile_score_numeric'] < range_info['max'])
                    ])
                
                if count > 0:
                    available_lexile.append({
                        'value': range_info['value'],
                        'label': range_info['label'],
                        'count': int(count)
                    })
            
            # Check for AD books
            ad_count = len(filtered_df[filtered_df['lexile_score'] == 'AD'])
            if ad_count > 0:
                available_lexile.append({
                    'value': 'AD',
                    'label': 'AD (Adult Directed)',
                    'count': int(ad_count)
                })
            
            available_options['lexile'] = available_lexile
        
        return jsonify({
            'success': True,
            'available_options': available_options,
            'total_books': len(filtered_df)
        })
        
    except Exception as e:
        logger.error(f"Adaptive search options error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("🚀 Starting Enhanced KidLit Curator with ML Integration")
    print(f"📊 ML Models: {'✓ Loaded' if recommendation_engine and recommendation_engine.ml_models else '❌ Not loaded'}")
    print(f"📚 Catalog: {'✓ Loaded' if recommendation_engine and recommendation_engine.catalog_df is not None else '❌ Not loaded'}")
    print(f"🎯 Lexile Predictor: {'✅ Ready' if lexile_predictor else '❌ Not available'}")
    print("🌐 Open your browser to: http://127.0.0.1:5001")
    
    app.run(debug=True, host='127.0.0.1', port=5001)