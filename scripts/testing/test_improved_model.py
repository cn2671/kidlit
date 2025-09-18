import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = DATA_DIR / "models"

class ImprovedModelTester:
    """Test improved age-based models with book type handling"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.load_improved_models()
        
    def load_improved_models(self):
        """Load all improved models"""
        try:
            # Load feature names
            feature_names_path = MODELS_DIR / "feature_names.joblib"
            if feature_names_path.exists():
                self.feature_names = joblib.load(feature_names_path)
                print(f"✅ Loaded feature names: {len(self.feature_names)} features")
            
            # Load models
            model_files = list(MODELS_DIR.glob("age_model_*.joblib"))
            for model_file in model_files:
                model_name = model_file.stem.replace("age_model_", "")
                if model_name == "general":
                    book_type = "General"
                elif model_name == "adult_directed":
                    book_type = "Adult_Directed"
                elif model_name == "standard_lexile":
                    book_type = "Standard_Lexile"
                elif model_name == "graphic_novel":
                    book_type = "Graphic_Novel"
                else:
                    continue
                    
                model = joblib.load(model_file)
                self.models[book_type] = model
                print(f"✅ Loaded {book_type} model")
            
            # Load scalers (if they exist)
            scaler_files = list(MODELS_DIR.glob("scaler_*.joblib"))
            for scaler_file in scaler_files:
                scaler_name = scaler_file.stem.replace("scaler_", "")
                if scaler_name == "general":
                    book_type = "General"
                elif scaler_name == "adult_directed":
                    book_type = "Adult_Directed"
                elif scaler_name == "standard_lexile":
                    book_type = "Standard_Lexile"
                else:
                    continue
                    
                scaler = joblib.load(scaler_file)
                self.scalers[book_type] = scaler
                
        except Exception as e:
            print(f"❌ Error loading improved models: {e}")
            
    def extract_age_features_from_tracking(self, title):
        """Extract age features from tracking dataset"""
        tracking_path = DATA_DIR / "lexile_collection_tracking.csv"
        df = pd.read_csv(tracking_path)
        
        # Find the book in our tracking data (flexible matching)
        book_rows = df[df['title'].str.contains(title, case=False, na=False, regex=False)]
        
        if len(book_rows) == 0:
            # Try partial matching
            title_words = title.lower().split()
            if len(title_words) > 1:
                for i in range(len(title_words), 0, -1):
                    partial_title = ' '.join(title_words[:i])
                    book_rows = df[df['title'].str.lower().str.contains(partial_title, na=False, regex=False)]
                    if len(book_rows) > 0:
                        break
        
        if len(book_rows) == 0:
            return None
            
        row = book_rows.iloc[0]
        notes = str(row.get('notes', ''))
        
        return self.extract_age_features(notes), row['title'], row['author']
    
    def extract_age_features(self, notes_text):
        """Extract age and grade information from notes"""
        if pd.isna(notes_text):
            return None
            
        notes_lower = str(notes_text).lower()
        
        # Extract age ranges
        age_match = re.search(r'ages?\s+(\d+)[-–to]*\s*(\d+)', notes_lower)
        min_age = None
        max_age = None
        
        if age_match:
            min_age = int(age_match.group(1))
            max_age = int(age_match.group(2)) if age_match.group(2) else min_age
        else:
            # Try single age
            single_age = re.search(r'ages?\s+(\d+)', notes_lower)
            if single_age:
                min_age = max_age = int(single_age.group(1))
        
        # Extract grade ranges
        grade_match = re.search(r'grades?\s+([k\d]+)[-–to]*\s*([k\d]*)', notes_lower)
        min_grade = None
        max_grade = None
        
        if grade_match:
            min_grade_str = grade_match.group(1)
            max_grade_str = grade_match.group(2) if grade_match.group(2) else min_grade_str
            
            # Convert K to 0, numbers to int
            min_grade = 0 if min_grade_str == 'k' else int(min_grade_str) if min_grade_str.isdigit() else None
            max_grade = 0 if max_grade_str == 'k' else int(max_grade_str) if max_grade_str.isdigit() else min_grade
        
        # Extract AR level
        ar_level = None
        ar_match = re.search(r'ar\s*:?\s*(\d+\.?\d*)', notes_lower)
        if ar_match:
            ar_level = float(ar_match.group(1))
            
        return min_age, max_age, min_grade, max_grade, ar_level
    
    def extract_additional_features(self, title, author, notes):
        """Extract additional predictive features (enhanced)"""
        features = {}
        
        # Book type indicators
        title_lower = str(title).lower()
        notes_lower = str(notes).lower() if pd.notna(notes) else ""
        author_lower = str(author).lower() if pd.notna(author) else ""
        
        # Series book indicator
        features['is_series'] = 1 if any(indicator in title_lower for indicator in 
                                       ['#', 'book', 'volume', 'series']) else 0
        
        # Picture book indicators (enhanced)
        picture_indicators = ['picture book', 'pages', 'illustrations', 'caldecott', 'i can read', 
                            'beginner reader', 'early reader', 'my first']
        features['is_picture_book'] = 1 if any(indicator in notes_lower for indicator in picture_indicators) else 0
        
        # Classic/award book (enhanced)
        classic_indicators = ['newbery', 'caldecott', 'award', 'medal', 'classic', 'honor', 'winner']
        features['is_classic'] = 1 if any(award in notes_lower for award in classic_indicators) else 0
        
        # Author popularity (enhanced)
        popular_authors = ['dr. seuss', 'roald dahl', 'eric carle', 'maurice sendak', 'mo willems', 
                          'arnold lobel', 'cynthia rylant', 'beverly cleary', 'mary pope osborne']
        features['popular_author'] = 1 if any(auth.lower() in author_lower for auth in popular_authors) else 0
        
        # Very early reader indicators
        early_indicators = ['pre-k', 'preschool', 'toddler', 'baby', 'first words', 'board book']
        features['is_very_early'] = 1 if any(indicator in notes_lower for indicator in early_indicators) else 0
        
        # Chapter book indicator
        chapter_indicators = ['chapter book', 'early chapter', 'beginning chapter']
        features['is_chapter_book'] = 1 if any(indicator in notes_lower for indicator in chapter_indicators) else 0
        
        return features
    
    def predict_with_improved_model(self, features, book_type='Standard_Lexile'):
        """Make prediction using appropriate improved model"""
        
        # Select appropriate model
        if book_type in self.models:
            model = self.models[book_type]
            model_name = book_type
        else:
            print(f"    ⚠️  No model for {book_type}, using general model")
            model = self.models.get('General', list(self.models.values())[0])
            model_name = 'General'
        
        # Create feature array in correct order
        X = np.array([[features[col] for col in self.feature_names]])
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        return prediction, model_name
    
    def test_improved_model(self):
        """Test improved models against comprehensive test cases"""
        
        # Load comprehensive test cases
        test_cases_path = DATA_DIR / "comprehensive_model_test_cases.csv"
        test_df = pd.read_csv(test_cases_path)
        
        print(f"🧪 TESTING IMPROVED BOOK-TYPE-AWARE MODELS")
        print(f"{'='*55}")
        print(f"Test cases: {len(test_df)}")
        print(f"Available models: {list(self.models.keys())}")
        
        results = []
        successful_tests = 0
        failed_tests = 0
        
        for idx, test_case in test_df.iterrows():
            title = test_case['title']
            verified_lexile = test_case['lexile_numeric']
            book_type = test_case['book_type']
            
            if idx < 10:  # Show first 10 for debugging
                print(f"\n📖 Testing: {title}")
                print(f"   Verified: {verified_lexile}L ({book_type})")
            elif idx == 10:
                print(f"\n   ... (continuing with remaining {len(test_df) - 10} books)")
            
            # Try to find age data in our tracking dataset
            age_data = self.extract_age_features_from_tracking(title)
            
            if age_data is None or age_data[0] is None:
                if idx < 10:
                    print(f"   ❌ No age data found")
                failed_tests += 1
                continue
                
            age_features, full_title, author = age_data
            min_age, max_age, min_grade, max_grade, ar_level = age_features
            
            if idx < 10:
                print(f"   Found: {full_title}")
                print(f"   Age: {min_age}-{max_age} years")
            
            # Get tracking data for additional features
            tracking_path = DATA_DIR / "lexile_collection_tracking.csv"
            tracking_df = pd.read_csv(tracking_path)
            book_row = tracking_df[tracking_df['title'] == full_title]
            
            if len(book_row) == 0:
                if idx < 10:
                    print(f"   ❌ Book not found in tracking data")
                failed_tests += 1
                continue
                
            notes = str(book_row.iloc[0].get('notes', ''))
            additional_features = self.extract_additional_features(full_title, author, notes)
            
            # Prepare feature vector (enhanced)
            feature_vector = {
                'min_age': min_age,
                'max_age': max_age if max_age else min_age,
                'avg_age': (min_age + (max_age if max_age else min_age)) / 2,
                'age_range': (max_age if max_age else min_age) - min_age,
                'min_grade': min_grade if min_grade is not None else -1,
                'max_grade': max_grade if max_grade is not None else -1,
                'ar_level': ar_level if ar_level is not None else -1,
                'is_ad_book': 1 if book_type == 'Adult_Directed' else 0,
                'is_gn_book': 1 if book_type == 'Graphic_Novel' else 0,
                **additional_features
            }
            
            # Make prediction with improved model
            predicted_lexile, model_used = self.predict_with_improved_model(feature_vector, book_type)
            error = abs(predicted_lexile - verified_lexile)
            
            if idx < 10:
                print(f"   🔮 Predicted: {predicted_lexile:.0f}L (using {model_used} model)")
                print(f"   📏 Error: {error:.0f}L")
            
            # Store result
            result = {
                'title': title,
                'verified_lexile': verified_lexile,
                'predicted_lexile': predicted_lexile,
                'error': error,
                'book_type': book_type,
                'model_used': model_used,
                'age_range': f"{min_age}-{max_age if max_age else min_age}"
            }
            results.append(result)
            successful_tests += 1
        
        if len(results) > 0:
            results_df = pd.DataFrame(results)
            
            # Calculate overall accuracy metrics
            mae = mean_absolute_error(results_df['verified_lexile'], results_df['predicted_lexile'])
            r2 = r2_score(results_df['verified_lexile'], results_df['predicted_lexile'])
            
            print(f"\n📊 IMPROVED MODEL RESULTS")
            print(f"{'='*40}")
            print(f"✅ Successful tests: {successful_tests}")
            print(f"❌ Failed tests: {failed_tests}")
            print(f"📈 Overall MAE: {mae:.1f}L")
            print(f"📈 Overall R²: {r2:.3f}")
            
            # Compare with original model
            original_mae = 122.5  # From previous comprehensive test
            improvement = ((original_mae - mae) / original_mae) * 100
            print(f"🚀 Improvement over original: {improvement:.1f}% better")
            
            # Analyze by book type
            print(f"\n📚 ACCURACY BY BOOK TYPE:")
            for book_type in ['Standard_Lexile', 'Adult_Directed', 'Graphic_Novel']:
                type_results = results_df[results_df['book_type'] == book_type]
                if len(type_results) > 0:
                    type_mae = type_results['error'].mean()
                    type_r2 = r2_score(type_results['verified_lexile'], type_results['predicted_lexile']) if len(type_results) > 1 else 0
                    
                    # Compare with original performance
                    if book_type == 'Standard_Lexile':
                        original_type_mae = 110.8
                    elif book_type == 'Adult_Directed':
                        original_type_mae = 181.4
                    else:
                        original_type_mae = 72.3
                    
                    type_improvement = ((original_type_mae - type_mae) / original_type_mae) * 100
                    
                    print(f"  {book_type}: {len(type_results)} books")
                    print(f"    New MAE: {type_mae:.1f}L vs Old: {original_type_mae:.1f}L ({type_improvement:+.1f}%)")
                    print(f"    R²: {type_r2:.3f}")
            
            # Error distribution
            print(f"\n🎯 ERROR DISTRIBUTION:")
            excellent = len(results_df[results_df['error'] < 50])
            good = len(results_df[(results_df['error'] >= 50) & (results_df['error'] < 100)])
            fair = len(results_df[(results_df['error'] >= 100) & (results_df['error'] < 200)])
            poor = len(results_df[results_df['error'] >= 200])
            
            print(f"  Excellent (<50L): {excellent} ({100*excellent/len(results_df):.1f}%)")
            print(f"  Good (50-100L): {good} ({100*good/len(results_df):.1f}%)")
            print(f"  Fair (100-200L): {fair} ({100*fair/len(results_df):.1f}%)")
            print(f"  Poor (>200L): {poor} ({100*poor/len(results_df):.1f}%)")
            
            # Show best and worst predictions
            print(f"\n✅ BEST PREDICTIONS:")
            best_predictions = results_df.nsmallest(5, 'error')
            for _, row in best_predictions.iterrows():
                title_short = row['title'][:35]
                print(f"  {title_short}: {row['verified_lexile']:.0f}L → {row['predicted_lexile']:.0f}L (±{row['error']:.0f}L)")
            
            print(f"\n⚠️ WORST PREDICTIONS:")
            worst_predictions = results_df.nlargest(5, 'error')
            for _, row in worst_predictions.iterrows():
                title_short = row['title'][:35]
                print(f"  {title_short}: {row['verified_lexile']:.0f}L → {row['predicted_lexile']:.0f}L (±{row['error']:.0f}L)")
            
            # Save results
            results_path = DATA_DIR / f"improved_model_test_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            results_df.to_csv(results_path, index=False)
            print(f"\n💾 Results saved: {results_path}")
            
            return results_df, mae, r2
        
        else:
            print("❌ No successful tests completed")
            return None, None, None

def main():
    tester = ImprovedModelTester()
    
    if len(tester.models) > 0:
        results_df, mae, r2 = tester.test_improved_model()
        return results_df, mae, r2
    else:
        print("❌ No improved models found")
        return None, None, None

if __name__ == "__main__":
    main()