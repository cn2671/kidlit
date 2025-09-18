import pandas as pd
import numpy as np
import pickle
import json
import re
from pathlib import Path
import textstat

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

def load_enhanced_model():
    """Load the enhanced Lexile estimation model"""
    model_path = MODELS_DIR / 'lexile_estimator.pkl'
    scaler_path = MODELS_DIR / 'lexile_scaler.pkl'
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model_data, scaler

def estimate_lexile_from_text(text, model_data, scaler):
    """Estimate Lexile score from raw text using the enhanced model"""
    
    model = model_data['model']
    feature_names = model_data['feature_names']
    
    # Extract features from text (matching CLEAR corpus features)
    try:
        words = len(text.split())
        sentences = text.count('.') + text.count('!') + text.count('?')
        if sentences == 0:
            sentences = 1
        paragraphs = len([p for p in text.split('\n\n') if p.strip()])
        if paragraphs == 0:
            paragraphs = 1
        
        features = [
            textstat.flesch_kincaid_grade(text),       # Flesch-Kincaid-Grade-Level
            textstat.flesch_reading_ease(text),        # Flesch-Reading-Ease  
            textstat.automated_readability_index(text), # Automated Readability Index
            textstat.smog_index(text),                 # SMOG Readability
            textstat.dale_chall_readability_score(text), # New Dale-Chall Readability Formula
            words,                                     # Google WC (word count)
            sentences,                                 # Sentence Count
            paragraphs                                 # Paragraphs
        ]
        
        # Handle any NaN or infinite values
        features = [f if np.isfinite(f) else 0 for f in features]
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        lexile_score = model.predict(features_scaled)[0]
        
        # Calculate confidence
        base_confidence = 0.7
        if 300 <= lexile_score <= 1200:
            range_bonus = 0.2
        elif 100 <= lexile_score <= 1500:
            range_bonus = 0.1
        else:
            range_bonus = -0.1
        
        confidence = max(0.1, min(1.0, base_confidence + range_bonus))
        
        return round(lexile_score), round(confidence, 2)
        
    except Exception as e:
        print(f"Error estimating Lexile for text: {e}")
        return None, 0.0

def update_books_with_enhanced_lexile():
    """Update the books dataset with enhanced Lexile estimates"""
    
    print("🚀 UPDATING BOOKS WITH ENHANCED LEXILE ESTIMATES")
    print("═════════════════════════════════════════════════")
    
    # Load current books data
    books_path = DATA_DIR / "raw" / "books_final_complete.csv"
    df = pd.read_csv(books_path)
    print(f"📖 Loaded {len(df)} books")
    
    # Load enhanced model
    model_data, scaler = load_enhanced_model()
    print(f"🤖 Loaded enhanced model: {model_data['model_name']}")
    print(f"   Performance: MAE {model_data['performance']['test_mae']:.1f}, R² {model_data['performance']['test_r2']:.3f}")
    
    # Create new columns for enhanced estimates
    df['lexile_enhanced'] = None
    df['lexile_confidence_enhanced'] = 0.0
    df['lexile_source_enhanced'] = ''
    
    enhanced_count = 0
    text_based_count = 0
    
    print(f"\n🔍 Processing books...")
    
    for idx, row in df.iterrows():
        lexile_estimate = None
        confidence = 0.0
        source = ''
        
        # Try to use book text for enhanced estimation
        text_content = None
        
        # Check different text fields
        for text_field in ['summary_gpt', 'description', 'excerpt']:
            if text_field in df.columns:
                text = row.get(text_field, '')
                if text and str(text) != 'nan' and len(str(text)) > 50:
                    text_content = str(text)
                    break
        
        if text_content:
            # Use enhanced model with actual text
            lexile_estimate, confidence = estimate_lexile_from_text(text_content, model_data, scaler)
            if lexile_estimate:
                source = 'enhanced_text_analysis'
                text_based_count += 1
        
        # Fallback to age-based estimation if no text available
        if not lexile_estimate:
            # Use original age-based approach as fallback
            reading_estimate = row.get('reading_level_estimate', '')
            if reading_estimate and str(reading_estimate) != 'nan':
                try:
                    estimate_data = json.loads(reading_estimate)
                    age_range_str = estimate_data.get('age_range', '')
                    
                    if age_range_str and str(age_range_str) != 'nan':
                        lexile_estimate, confidence = parse_age_range_to_lexile(age_range_str)
                        if lexile_estimate:
                            source = 'age_range_fallback'
                except json.JSONDecodeError:
                    continue
        
        # Apply estimates
        if lexile_estimate:
            df.at[idx, 'lexile_enhanced'] = lexile_estimate
            df.at[idx, 'lexile_confidence_enhanced'] = confidence
            df.at[idx, 'lexile_source_enhanced'] = source
            enhanced_count += 1
    
    print(f"\n📊 ENHANCEMENT RESULTS")
    print(f"═══════════════════════")
    print(f"📚 Total books: {len(df)}")
    print(f"✅ Books with enhanced Lexile: {enhanced_count} ({(enhanced_count/len(df))*100:.1f}%)")
    print(f"🤖 Text-based estimates: {text_based_count} ({(text_based_count/len(df))*100:.1f}%)")
    print(f"👶 Age-based fallback: {enhanced_count - text_based_count}")
    
    # Show distribution comparison
    print(f"\n📈 LEXILE DISTRIBUTION COMPARISON")
    print(f"═══════════════════════════════════")
    
    ranges = [(0, 300), (300, 600), (600, 900), (900, 1200), (1200, 1500), (1500, 2000)]
    range_names = ['0-300L', '300-600L', '600-900L', '900-1200L', '1200-1500L', '1500L+']
    
    print(f"{'Range':<12} {'Original':<10} {'Enhanced':<10} {'Change':<8}")
    print(f"{'─' * 42}")
    
    for (min_lex, max_lex), name in zip(ranges, range_names):
        # Skip original comparison since lexile_score column doesn't exist yet
        enhanced_count = len(df[(df['lexile_enhanced'] >= min_lex) & (df['lexile_enhanced'] < max_lex)])
        
        print(f"{name:<12} {'N/A':<10} {enhanced_count:<10} {'N/A':<8}")
    
    # Save enhanced dataset
    output_path = DATA_DIR / "books_final_enhanced.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n💾 Enhanced dataset saved to: {output_path}")
    print(f"\n🎯 NEXT STEPS:")
    print(f"1. Update Flask backend to use lexile_enhanced column")
    print(f"2. Display confidence scores in the UI")
    print(f"3. Test the enhanced Lexile filtering")
    
    return df

def parse_age_range_to_lexile(age_range_str):
    """Parse age range string and convert to Lexile estimate (fallback method)"""
    
    age_to_lexile = {
        (0, 2): (0, 100),
        (3, 5): (50, 300),
        (4, 6): (100, 400),
        (5, 7): (200, 500),
        (6, 8): (300, 600),
        (7, 9): (400, 700),
        (8, 10): (500, 800),
        (9, 12): (600, 900),
        (11, 14): (700, 1000),
        (13, 18): (800, 1200),
    }
    
    age_range_str = str(age_range_str).strip()
    
    # Format 1: "6-8" or "3-5" 
    range_match = re.match(r'^(\d+)[\-\–](\d+)$', age_range_str)
    if range_match:
        min_age = int(range_match.group(1))
        max_age = int(range_match.group(2))
        return find_best_lexile_match(min_age, max_age, age_to_lexile)
    
    # Format 2: "13+" 
    plus_match = re.match(r'^(\d+)\+$', age_range_str)
    if plus_match:
        min_age = int(plus_match.group(1))
        max_age = min_age + 5
        return find_best_lexile_match(min_age, max_age, age_to_lexile)
    
    # Format 3: Single age "5"
    single_match = re.match(r'^\d+$', age_range_str)
    if single_match:
        age = int(age_range_str)
        return find_best_lexile_match(age, age + 1, age_to_lexile)
    
    return None, 0.0

def find_best_lexile_match(min_age, max_age, age_to_lexile_mapping):
    """Find the best Lexile estimate for a given age range"""
    
    best_match = None
    best_overlap = 0
    
    for (range_min, range_max), (lexile_min, lexile_max) in age_to_lexile_mapping.items():
        overlap_start = max(min_age, range_min)
        overlap_end = min(max_age, range_max)
        overlap = max(0, overlap_end - overlap_start)
        
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = (lexile_min, lexile_max)
    
    if best_match:
        lexile_estimate = (best_match[0] + best_match[1]) // 2
        
        age_span = max_age - min_age
        if age_span <= 2:
            confidence = 0.4  # Lower confidence for fallback method
        elif age_span <= 4:
            confidence = 0.3
        else:
            confidence = 0.2
            
        return lexile_estimate, confidence
    
    return None, 0.0

if __name__ == "__main__":
    updated_df = update_books_with_enhanced_lexile()