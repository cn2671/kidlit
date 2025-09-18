import pandas as pd
import pickle
import numpy as np
import textstat
from pathlib import Path

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
            textstat.flesch_kincaid_grade(text),
            textstat.flesch_reading_ease(text),
            textstat.automated_readability_index(text),
            textstat.smog_index(text),
            textstat.dale_chall_readability_score(text),
            words,
            sentences,
            paragraphs
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

def find_book_in_dataset(title_search, author_search=None):
    """Find a book in the dataset by title and optionally author"""
    
    # Load enhanced dataset
    enhanced_path = DATA_DIR / "books_final_enhanced.csv"
    df = pd.read_csv(enhanced_path)
    
    # Search for the book
    title_mask = df['title_clean'].str.contains(title_search, case=False, na=False)
    
    if author_search:
        author_mask = df['author_clean'].str.contains(author_search, case=False, na=False)
        matches = df[title_mask & author_mask]
    else:
        matches = df[title_mask]
    
    return matches

def test_known_lexile_books():
    """Test books with known Lexile scores"""
    
    print("🧪 TESTING BOOKS WITH KNOWN LEXILE SCORES")
    print("═════════════════════════════════════════")
    
    # Load enhanced model
    model_data, scaler = load_enhanced_model()
    print(f"🤖 Loaded model: {model_data['model_name']}")
    
    # Books with known Lexile scores
    test_books = [
        {
            "title": "The Help",
            "author": "Kathryn Stockett",
            "known_lexile": 730,
            "search_title": "Help"
        },
        {
            "title": "Horton Hears a Who",
            "author": "Dr. Seuss", 
            "known_lexile": 600,
            "search_title": "Horton"
        },
        {
            "title": "The Scarlet Letter",
            "author": "Nathaniel Hawthorne",
            "known_lexile": 1280,
            "search_title": "Scarlet Letter"
        },
        {
            "title": "The Giving Tree",
            "author": "Shel Silverstein",
            "known_lexile": 530,
            "search_title": "Giving Tree"
        }
    ]
    
    print(f"\n📚 SEARCHING FOR BOOKS IN DATASET")
    print(f"{'='*60}")
    
    results = []
    
    for i, book in enumerate(test_books):
        print(f"\n📖 Book {i+1}: {book['title']} by {book['author']}")
        print(f"    Known Lexile: {book['known_lexile']}L")
        
        # Search for book in dataset
        matches = find_book_in_dataset(book['search_title'], book['author'].split()[0])
        
        if len(matches) == 0:
            # Try broader search
            matches = find_book_in_dataset(book['search_title'])
        
        if len(matches) > 0:
            print(f"    ✅ Found {len(matches)} match(es) in dataset")
            
            # Use the first match
            match = matches.iloc[0]
            print(f"    📚 Dataset match: \"{match['title_clean']}\" by {match['author_clean']}")
            
            # Get text content
            text = str(match.get('summary_gpt', ''))
            if len(text) > 50:
                print(f"    📝 Text available: {len(text)} characters")
                
                # Test with our model
                predicted_lexile, confidence, features = estimate_lexile_from_text(text, model_data, scaler)
                
                if predicted_lexile:
                    # Get existing dataset estimate
                    dataset_lexile = match.get('lexile_enhanced', 'N/A')
                    dataset_confidence = match.get('lexile_confidence_enhanced', 'N/A')
                    
                    print(f"    🤖 Fresh ML estimate: {predicted_lexile}L (confidence: {confidence})")
                    print(f"    🗂️  Dataset estimate: {dataset_lexile}L (confidence: {dataset_confidence})")
                    
                    # Compare with known Lexile
                    difference_from_known = abs(predicted_lexile - book['known_lexile'])
                    print(f"    📏 Difference from known: {difference_from_known}L")
                    
                    if difference_from_known <= 100:
                        print(f"    ✅ Excellent accuracy (within 100L)")
                        accuracy = "Excellent"
                    elif difference_from_known <= 200:
                        print(f"    ⭐ Good accuracy (within 200L)")
                        accuracy = "Good"
                    elif difference_from_known <= 300:
                        print(f"    ⚠️  Moderate accuracy (within 300L)")
                        accuracy = "Moderate"
                    else:
                        print(f"    ❌ Poor accuracy (>300L difference)")
                        accuracy = "Poor"
                    
                    # Show readability breakdown
                    if features:
                        print(f"    📊 Key readability metrics:")
                        print(f"        Flesch-Kincaid Grade: {features[0]:.1f}")
                        print(f"        Reading Ease: {features[1]:.1f}")
                        print(f"        Word Count: {int(features[5])}")
                    
                    results.append({
                        'title': book['title'],
                        'known_lexile': book['known_lexile'],
                        'predicted_lexile': predicted_lexile,
                        'difference': difference_from_known,
                        'accuracy': accuracy,
                        'confidence': confidence,
                        'text_length': len(text)
                    })
                else:
                    print(f"    ❌ Failed to generate ML estimate")
            else:
                print(f"    ⚠️  No substantial text content available")
        else:
            print(f"    ❌ Book not found in dataset")
    
    # Summary analysis
    if results:
        print(f"\n📊 ACCURACY ANALYSIS")
        print(f"{'='*50}")
        
        differences = [r['difference'] for r in results]
        avg_difference = sum(differences) / len(differences)
        
        print(f"Books tested: {len(results)}")
        print(f"Average difference: {avg_difference:.1f}L")
        print(f"Min difference: {min(differences):.1f}L")
        print(f"Max difference: {max(differences):.1f}L")
        
        # Accuracy distribution
        accuracy_counts = {}
        for result in results:
            acc = result['accuracy']
            accuracy_counts[acc] = accuracy_counts.get(acc, 0) + 1
        
        print(f"\nAccuracy distribution:")
        for acc, count in accuracy_counts.items():
            print(f"  {acc}: {count} book(s)")
        
        # Detailed results table
        print(f"\n📋 DETAILED RESULTS")
        print(f"{'='*50}")
        print(f"{'Book':<20} {'Known':<6} {'Pred':<6} {'Diff':<5} {'Acc':<10}")
        print(f"{'-'*50}")
        
        for result in results:
            title = result['title'][:19]
            print(f"{title:<20} {result['known_lexile']:<6}L {result['predicted_lexile']:<6}L {result['difference']:<5.0f}L {result['accuracy']:<10}")
    
    print(f"\n🎯 WOULD YOU LIKE TO TEST MORE BOOKS?")
    print(f"{'='*50}")
    print(f"Please provide more books in this format:")
    print(f"  \"Book Title\" - Author Name - Lexile Score")
    print(f"Example: \"Where the Wild Things Are\" - Maurice Sendak - 740L")

if __name__ == "__main__":
    test_known_lexile_books()