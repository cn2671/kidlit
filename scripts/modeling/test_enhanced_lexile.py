import pandas as pd
import pickle
import numpy as np
import textstat
from pathlib import Path
import random

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
    
    try:
        # Extract features from text
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
        
        return round(lexile_score), round(confidence, 2), features
        
    except Exception as e:
        print(f"Error estimating Lexile for text: {e}")
        return None, 0.0, None

def test_books_sample():
    """Test the enhanced Lexile model on a sample of books"""
    
    print("🧪 TESTING ENHANCED LEXILE MODEL ON BOOKS")
    print("═════════════════════════════════════════")
    
    # Load enhanced dataset
    enhanced_path = DATA_DIR / "books_final_enhanced.csv"
    df = pd.read_csv(enhanced_path)
    print(f"📖 Loaded {len(df)} books with enhanced estimates")
    
    # Load enhanced model
    model_data, scaler = load_enhanced_model()
    print(f"🤖 Loaded model: {model_data['model_name']}")
    
    # Sample books with good text content
    books_with_text = df[
        (df['summary_gpt'].notna()) & 
        (df['summary_gpt'].str.len() > 100)  # At least 100 characters
    ].copy()
    
    print(f"📚 Found {len(books_with_text)} books with substantial text content")
    
    # Sample 15 random books for testing
    if len(books_with_text) > 15:
        sample_books = books_with_text.sample(n=15, random_state=42)
    else:
        sample_books = books_with_text
    
    print(f"\n🔬 TESTING {len(sample_books)} SAMPLE BOOKS")
    print(f"{'='*80}")
    
    results = []
    
    for idx, (_, book) in enumerate(sample_books.iterrows()):
        title = str(book.get('title_clean', 'Unknown'))[:40]
        author = str(book.get('author_clean', 'Unknown'))[:25]
        text = str(book.get('summary_gpt', ''))
        
        # Get existing enhanced estimate
        existing_lexile = book.get('lexile_enhanced', 'N/A')
        existing_confidence = book.get('lexile_confidence_enhanced', 'N/A')
        existing_source = book.get('lexile_source_enhanced', 'N/A')
        
        # Test with fresh ML analysis
        fresh_lexile, fresh_confidence, features = estimate_lexile_from_text(text, model_data, scaler)
        
        print(f"\n📖 Book {idx+1}: {title}")
        print(f"    Author: {author}")
        print(f"    Text length: {len(text)} characters")
        print(f"    Existing estimate: {existing_lexile}L (confidence: {existing_confidence}, source: {existing_source})")
        print(f"    Fresh ML estimate: {fresh_lexile}L (confidence: {fresh_confidence})")
        
        if fresh_lexile and existing_lexile != 'N/A':
            difference = abs(fresh_lexile - existing_lexile)
            print(f"    Difference: {difference}L")
            if difference > 150:
                print(f"    ⚠️  Large difference - may need investigation")
            elif difference > 75:
                print(f"    ⚡ Moderate difference - within expected range")
            else:
                print(f"    ✅ Close agreement")
        
        # Show readability breakdown if available
        if features:
            print(f"    Readability features:")
            feature_names = [
                'Flesch-Kincaid Grade', 'Flesch Reading Ease', 'Auto Readability Index',
                'SMOG Readability', 'Dale-Chall Score', 'Word Count', 'Sentences', 'Paragraphs'
            ]
            for name, value in zip(feature_names[:5], features[:5]):  # Show first 5 features
                print(f"      {name}: {value:.1f}")
        
        results.append({
            'title': title,
            'author': author,
            'existing_lexile': existing_lexile if existing_lexile != 'N/A' else None,
            'fresh_lexile': fresh_lexile,
            'difference': abs(fresh_lexile - existing_lexile) if fresh_lexile and existing_lexile != 'N/A' else None,
            'text_length': len(text),
            'fresh_confidence': fresh_confidence
        })
    
    print(f"\n📊 TESTING SUMMARY")
    print(f"{'='*50}")
    
    # Calculate summary statistics
    valid_results = [r for r in results if r['existing_lexile'] and r['fresh_lexile']]
    
    if valid_results:
        differences = [r['difference'] for r in valid_results if r['difference']]
        avg_difference = sum(differences) / len(differences)
        max_difference = max(differences)
        min_difference = min(differences)
        
        print(f"Books tested: {len(results)}")
        print(f"Valid comparisons: {len(valid_results)}")
        print(f"Average difference: {avg_difference:.1f}L")
        print(f"Min difference: {min_difference:.1f}L")
        print(f"Max difference: {max_difference:.1f}L")
        
        # Categorize agreement levels
        close_agreement = len([d for d in differences if d <= 75])
        moderate_agreement = len([d for d in differences if 75 < d <= 150])
        large_differences = len([d for d in differences if d > 150])
        
        print(f"\nAgreement Analysis:")
        print(f"  Close agreement (≤75L): {close_agreement}/{len(valid_results)} ({100*close_agreement/len(valid_results):.1f}%)")
        print(f"  Moderate difference (75-150L): {moderate_agreement}/{len(valid_results)} ({100*moderate_agreement/len(valid_results):.1f}%)")
        print(f"  Large difference (>150L): {large_differences}/{len(valid_results)} ({100*large_differences/len(valid_results):.1f}%)")
        
        # Show confidence distribution
        confidences = [r['fresh_confidence'] for r in results if r['fresh_confidence']]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            print(f"\nAverage confidence: {avg_confidence:.2f}")
            
            high_conf = len([c for c in confidences if c >= 0.8])
            med_conf = len([c for c in confidences if 0.6 <= c < 0.8])
            low_conf = len([c for c in confidences if c < 0.6])
            
            print(f"Confidence distribution:")
            print(f"  High (≥0.8): {high_conf} books")
            print(f"  Medium (0.6-0.8): {med_conf} books") 
            print(f"  Low (<0.6): {low_conf} books")
    
    print(f"\n🎯 MODEL PERFORMANCE ASSESSMENT")
    print(f"{'='*50}")
    
    if valid_results and avg_difference < 100:
        print(f"✅ Excellent performance: Average difference {avg_difference:.1f}L < 100L")
    elif valid_results and avg_difference < 150:
        print(f"⭐ Good performance: Average difference {avg_difference:.1f}L < 150L")
    elif valid_results:
        print(f"⚠️  Needs investigation: Average difference {avg_difference:.1f}L ≥ 150L")
    else:
        print(f"❓ Unable to assess: Insufficient comparison data")
    
    # Show some interesting examples
    print(f"\n📋 NOTABLE EXAMPLES")
    print(f"{'='*50}")
    
    # Sort by confidence for interesting examples
    sorted_results = sorted([r for r in results if r['fresh_lexile']], key=lambda x: x['fresh_confidence'], reverse=True)
    
    print(f"Highest confidence predictions:")
    for i, result in enumerate(sorted_results[:3]):
        print(f"  {i+1}. {result['title'][:30]}: {result['fresh_lexile']}L (conf: {result['fresh_confidence']})")
    
    if len(sorted_results) > 3:
        print(f"\nLowest confidence predictions:")
        for i, result in enumerate(sorted_results[-3:]):
            print(f"  {i+1}. {result['title'][:30]}: {result['fresh_lexile']}L (conf: {result['fresh_confidence']})")
    
    return results

if __name__ == "__main__":
    results = test_books_sample()