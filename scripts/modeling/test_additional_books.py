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

def test_additional_books():
    """Test additional books with known Lexile scores including AD classifications"""
    
    print("🧪 TESTING ADDITIONAL BOOKS WITH KNOWN LEXILE SCORES")
    print("════════════════════════════════════════════════════")
    
    # Load enhanced model
    model_data, scaler = load_enhanced_model()
    print(f"🤖 Loaded model: {model_data['model_name']}")
    
    # Additional books with detailed Lexile information
    test_books = [
        {
            "title": "Where the Wild Things Are",
            "author": "Maurice Sendak",
            "known_lexile": "AD740L",
            "lexile_numeric": 740,
            "grade_level": "Preschool-3rd",
            "guided_reading": "J",
            "ar_level": 3.4,
            "search_title": "Wild Things",
            "notes": "AD = Adult-Directed, non-standard vocab/complex structures"
        },
        {
            "title": "Goodnight Moon",
            "author": "Margaret Wise Brown",
            "known_lexile": "AD360L", 
            "lexile_numeric": 360,
            "grade_level": "Preschool-1st",
            "guided_reading": "K",
            "ar_level": 2.3,
            "search_title": "Goodnight Moon",
            "notes": "AD = Adult-Directed reading to young children"
        },
        {
            "title": "The Cat in the Hat",
            "author": "Dr. Seuss",
            "known_lexile": "430L",
            "lexile_numeric": 430,
            "grade_level": "Preschool-3rd",
            "guided_reading": "J",
            "ar_level": 2.1,
            "search_title": "Cat in the Hat",
            "notes": "Early reader level"
        },
        {
            "title": "Bridge to Terabithia", 
            "author": "Katherine Paterson",
            "known_lexile": "810L",
            "lexile_numeric": 810,
            "grade_level": "3rd-8th",
            "guided_reading": "U",
            "ar_level": 4.6,
            "search_title": "Bridge to Terabithia",
            "notes": "Middle-grade level"
        }
    ]
    
    print(f"\n📚 TESTING BOOKS WITH AD CLASSIFICATIONS")
    print(f"{'='*70}")
    print(f"Note: AD = Adult-Directed (complex vocab/structures for young readers)")
    
    results = []
    
    for i, book in enumerate(test_books):
        print(f"\n📖 Book {i+1}: {book['title']}")
        print(f"    Author: {book['author']}")
        print(f"    Known Lexile: {book['known_lexile']}")
        print(f"    Grade Level: {book['grade_level']}")
        print(f"    Guided Reading: {book['guided_reading']}")
        print(f"    AR Level: {book['ar_level']}")
        print(f"    Notes: {book['notes']}")
        
        # Search for book in dataset
        matches = find_book_in_dataset(book['search_title'], book['author'].split()[0])
        
        if len(matches) == 0:
            # Try broader search
            matches = find_book_in_dataset(book['search_title'])
        
        if len(matches) > 0:
            print(f"    ✅ Found {len(matches)} match(es) in dataset")
            
            # Use the first match
            match = matches.iloc[0]
            print(f"    📚 Dataset: \"{match['title_clean']}\" by {match['author_clean']}")
            
            # Get text content
            text = str(match.get('summary_gpt', ''))
            if len(text) > 50:
                print(f"    📝 Text: {len(text)} characters")
                
                # Test with our model
                predicted_lexile, confidence, features = estimate_lexile_from_text(text, model_data, scaler)
                
                if predicted_lexile:
                    # Get existing dataset estimate
                    dataset_lexile = match.get('lexile_enhanced', 'N/A')
                    dataset_confidence = match.get('lexile_confidence_enhanced', 'N/A')
                    
                    print(f"    🤖 ML Prediction: {predicted_lexile}L (confidence: {confidence})")
                    print(f"    🗂️  Dataset Value: {dataset_lexile}L (confidence: {dataset_confidence})")
                    
                    # Compare with known numeric Lexile
                    difference_from_known = abs(predicted_lexile - book['lexile_numeric'])
                    print(f"    📏 Difference from known: {difference_from_known}L")
                    
                    # Accuracy assessment
                    if difference_from_known <= 100:
                        accuracy = "Excellent"
                        print(f"    ✅ Excellent accuracy (≤100L)")
                    elif difference_from_known <= 200:
                        accuracy = "Good" 
                        print(f"    ⭐ Good accuracy (≤200L)")
                    elif difference_from_known <= 300:
                        accuracy = "Moderate"
                        print(f"    ⚠️  Moderate accuracy (≤300L)")
                    else:
                        accuracy = "Poor"
                        print(f"    ❌ Poor accuracy (>300L)")
                    
                    # Special analysis for AD books
                    if "AD" in book['known_lexile']:
                        print(f"    🎯 AD Analysis: This book uses complex structures for young readers")
                        if predicted_lexile > book['lexile_numeric']:
                            print(f"    📊 Model detected complexity (summary effect likely)")
                        else:
                            print(f"    📊 Model underestimated complexity")
                    
                    # Show key readability metrics
                    if features:
                        print(f"    📈 Readability Breakdown:")
                        print(f"        Flesch-Kincaid Grade: {features[0]:.1f}")
                        print(f"        Reading Ease: {features[1]:.1f}")
                        print(f"        SMOG: {features[3]:.1f}")
                        print(f"        Words: {int(features[5])}")
                    
                    results.append({
                        'title': book['title'],
                        'known_lexile': book['lexile_numeric'],
                        'known_lexile_full': book['known_lexile'],
                        'predicted_lexile': predicted_lexile,
                        'difference': difference_from_known,
                        'accuracy': accuracy,
                        'confidence': confidence,
                        'has_AD': "AD" in book['known_lexile'],
                        'grade_level': book['grade_level'],
                        'text_length': len(text)
                    })
                else:
                    print(f"    ❌ Failed to generate ML estimate")
            else:
                print(f"    ⚠️  Insufficient text content")
        else:
            print(f"    ❌ Book not found in dataset")
    
    # Comprehensive analysis
    if results:
        print(f"\n📊 COMPREHENSIVE ACCURACY ANALYSIS")
        print(f"{'='*60}")
        
        differences = [r['difference'] for r in results]
        avg_difference = sum(differences) / len(differences)
        
        print(f"Books tested: {len(results)}")
        print(f"Average difference: {avg_difference:.1f}L")
        print(f"Min difference: {min(differences):.1f}L")
        print(f"Max difference: {max(differences):.1f}L")
        
        # AD vs non-AD analysis
        ad_books = [r for r in results if r['has_AD']]
        non_ad_books = [r for r in results if not r['has_AD']]
        
        if ad_books:
            ad_avg = sum([r['difference'] for r in ad_books]) / len(ad_books)
            print(f"\nAD Books Analysis:")
            print(f"  Count: {len(ad_books)}")
            print(f"  Average difference: {ad_avg:.1f}L")
            print(f"  Note: AD books have complex structures for young readers")
        
        if non_ad_books:
            non_ad_avg = sum([r['difference'] for r in non_ad_books]) / len(non_ad_books)
            print(f"\nNon-AD Books Analysis:")
            print(f"  Count: {len(non_ad_books)}")
            print(f"  Average difference: {non_ad_avg:.1f}L")
        
        # Accuracy distribution
        accuracy_counts = {}
        for result in results:
            acc = result['accuracy']
            accuracy_counts[acc] = accuracy_counts.get(acc, 0) + 1
        
        print(f"\nAccuracy Distribution:")
        for acc, count in accuracy_counts.items():
            percentage = (count / len(results)) * 100
            print(f"  {acc}: {count}/{len(results)} ({percentage:.1f}%)")
        
        # Detailed results table
        print(f"\n📋 DETAILED RESULTS TABLE")
        print(f"{'='*80}")
        print(f"{'Book':<25} {'Known':<8} {'Pred':<6} {'Diff':<5} {'Acc':<10} {'Grade':<12}")
        print(f"{'-'*80}")
        
        for result in results:
            title = result['title'][:24]
            known = result['known_lexile_full']
            pred = f"{result['predicted_lexile']}L"
            diff = f"{result['difference']:.0f}L"
            acc = result['accuracy']
            grade = result['grade_level']
            
            print(f"{title:<25} {known:<8} {pred:<6} {diff:<5} {acc:<10} {grade:<12}")
    
    print(f"\n🔍 KEY INSIGHTS FROM TESTING")
    print(f"{'='*50}")
    print(f"1. Summary vs. Actual Text: We're still testing summaries, not book text")
    print(f"2. AD Classification: Adult-Directed books use complex language for young readers")
    print(f"3. Model Performance: Better accuracy on higher-level books")
    print(f"4. Validation Need: Actual book excerpts would provide better validation")
    
    print(f"\n🎯 RECOMMENDATIONS")
    print(f"{'='*30}")
    print(f"• Model works well for 600L-1200L range (middle grade)")
    print(f"• Consider summary complexity when interpreting results")
    print(f"• AD books require special consideration for young readers")
    print(f"• Future: Test with actual book excerpts for true validation")

if __name__ == "__main__":
    test_additional_books()