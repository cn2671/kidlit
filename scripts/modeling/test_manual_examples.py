import pickle
import numpy as np
import textstat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
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

def test_manual_examples():
    """Test the enhanced Lexile model on manually created examples"""
    
    print("🧪 TESTING ENHANCED LEXILE MODEL WITH MANUAL EXAMPLES")
    print("═════════════════════════════════════════════════════")
    
    # Load enhanced model
    model_data, scaler = load_enhanced_model()
    print(f"🤖 Loaded model: {model_data['model_name']}")
    print(f"   Performance: MAE {model_data['performance']['test_mae']:.1f}L, R² {model_data['performance']['test_r2']:.3f}")
    
    # Test examples at different reading levels
    test_examples = [
        {
            "title": "Simple Picture Book Example",
            "expected_range": "200-400L",
            "text": "The cat sat on the mat. The dog ran in the yard. Mom made a big cake. Dad read a book. We had fun at the park."
        },
        {
            "title": "Early Reader Example", 
            "expected_range": "400-600L",
            "text": "Sarah looked at the mysterious box on her doorstep. She wondered who could have left it there. The package was wrapped in bright blue paper with a golden ribbon. When she shook it gently, something rattled inside."
        },
        {
            "title": "Middle Grade Example",
            "expected_range": "600-900L", 
            "text": "The ancient castle stood majestically against the stormy sky. Lightning illuminated its towering spires and crumbling walls. Isabella clutched her flashlight tightly as she approached the enormous wooden door. She had come here searching for answers about her family's mysterious past, but now she was beginning to wonder if some secrets were better left buried."
        },
        {
            "title": "Advanced Chapter Book Example",
            "expected_range": "900-1200L",
            "text": "The revolutionary discoveries in quantum mechanics challenged fundamental assumptions about the nature of reality. Scientists grappled with concepts that seemed to contradict common sense: particles existing in multiple states simultaneously, information traveling faster than light, and observations that appeared to influence the very phenomena being studied. These paradigm-shifting revelations would ultimately transform our understanding of the universe itself."
        },
        {
            "title": "Very Simple Text",
            "expected_range": "100-300L",
            "text": "I see a cat. The cat is big. The cat is black. I like cats."
        },
        {
            "title": "Complex Academic Text",
            "expected_range": "1200L+",
            "text": "The epistemological implications of postmodern deconstructionist theory fundamentally challenge traditional hermeneutical frameworks, necessitating a comprehensive reevaluation of interpretive methodologies within contemporary literary criticism. This paradigmatic shift demands rigorous scrutiny of hegemonic discursive practices and their attendant ideological presumptions."
        }
    ]
    
    print(f"\n📚 TESTING {len(test_examples)} EXAMPLE TEXTS")
    print(f"{'='*80}")
    
    for i, example in enumerate(test_examples):
        print(f"\n📖 Example {i+1}: {example['title']}")
        print(f"    Expected range: {example['expected_range']}")
        print(f"    Text: \"{example['text'][:80]}{'...' if len(example['text']) > 80 else ''}\"")
        
        # Get ML estimate
        lexile_score, confidence, features = estimate_lexile_from_text(example['text'], model_data, scaler)
        
        if lexile_score:
            print(f"    🤖 ML Estimate: {lexile_score}L (confidence: {confidence})")
            
            # Show readability breakdown
            if features:
                print(f"    📊 Readability Analysis:")
                feature_names = [
                    'Flesch-Kincaid Grade', 'Flesch Reading Ease', 'Auto Readability', 
                    'SMOG', 'Dale-Chall', 'Words', 'Sentences', 'Paragraphs'
                ]
                for name, value in zip(feature_names, features):
                    if name in ['Words', 'Sentences', 'Paragraphs']:
                        print(f"        {name}: {int(value)}")
                    else:
                        print(f"        {name}: {value:.1f}")
            
            # Assess prediction quality
            if "100-300L" in example['expected_range'] and 100 <= lexile_score <= 300:
                print(f"    ✅ Excellent prediction - within expected range")
            elif "200-400L" in example['expected_range'] and 200 <= lexile_score <= 400:
                print(f"    ✅ Excellent prediction - within expected range")
            elif "400-600L" in example['expected_range'] and 400 <= lexile_score <= 600:
                print(f"    ✅ Excellent prediction - within expected range")  
            elif "600-900L" in example['expected_range'] and 600 <= lexile_score <= 900:
                print(f"    ✅ Excellent prediction - within expected range")
            elif "900-1200L" in example['expected_range'] and 900 <= lexile_score <= 1200:
                print(f"    ✅ Excellent prediction - within expected range")
            elif "1200L+" in example['expected_range'] and lexile_score >= 1200:
                print(f"    ✅ Excellent prediction - within expected range")
            else:
                print(f"    ⚠️  Prediction outside expected range - may need review")
                
        else:
            print(f"    ❌ Failed to generate estimate")
    
    print(f"\n🎯 ADDITIONAL REAL-WORLD TESTS")
    print(f"{'='*50}")
    
    # Test some famous children's book openings
    famous_examples = [
        {
            "title": "Green Eggs and Ham (Dr. Seuss)",
            "text": "I do not like green eggs and ham. I do not like them, Sam-I-Am. Do you like green eggs and ham? I would not like them here or there. I would not like them anywhere.",
            "known_lexile": "30L (actual)"
        },
        {
            "title": "Charlotte's Web (E.B. White)",
            "text": "Where's Papa going with that ax? said Fern to her mother as they were setting the table for breakfast. Out to the hoghouse, replied Mrs. Arable. Some pigs were born last night.",
            "known_lexile": "680L (actual)"
        },
        {
            "title": "Harry Potter Opening (J.K. Rowling)",
            "text": "Mr. and Mrs. Dursley of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you'd expect to be involved in anything strange or mysterious.",
            "known_lexile": "880L (actual)"
        }
    ]
    
    for example in famous_examples:
        print(f"\n📚 {example['title']}")
        print(f"    Known Lexile: {example['known_lexile']}")
        
        lexile_score, confidence, _ = estimate_lexile_from_text(example['text'], model_data, scaler)
        
        if lexile_score:
            print(f"    🤖 ML Estimate: {lexile_score}L (confidence: {confidence})")
            
            # Extract known lexile for comparison if available
            import re
            known_match = re.search(r'(\d+)L', example['known_lexile'])
            if known_match:
                known_lexile = int(known_match.group(1))
                difference = abs(lexile_score - known_lexile)
                print(f"    📏 Difference: {difference}L")
                
                if difference <= 100:
                    print(f"    ✅ Excellent accuracy (within 100L)")
                elif difference <= 200:
                    print(f"    ⭐ Good accuracy (within 200L)")
                else:
                    print(f"    ⚠️  Large difference - may need investigation")
    
    print(f"\n🔬 MODEL INSIGHTS")
    print(f"{'='*40}")
    print(f"The enhanced Lexile model uses 8 features:")
    print(f"  1. Flesch-Kincaid Grade Level (most important)")
    print(f"  2. Automated Readability Index (secondary)")
    print(f"  3. Flesch Reading Ease")
    print(f"  4. SMOG Readability")  
    print(f"  5. Dale-Chall Readability Score")
    print(f"  6. Word Count")
    print(f"  7. Sentence Count")
    print(f"  8. Paragraph Count")
    print(f"\nThe model was trained on 4,724 verified text samples from")
    print(f"the CommonLit CLEAR corpus with known Lexile ranges.")

if __name__ == "__main__":
    test_manual_examples()