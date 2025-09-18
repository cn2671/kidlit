import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

class ComprehensiveModelValidator:
    """Test age-based model against comprehensive set of verified Lexile scores"""
    
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.load_model()
        
    def load_model(self):
        """Load the trained age-based model"""
        model_path = DATA_DIR / "age_model.joblib"
        if model_path.exists():
            self.model = joblib.load(model_path)
            self.feature_names = ['min_age', 'max_age', 'avg_age', 'age_range', 'min_grade', 'max_grade', 
                                'ar_level', 'is_ad_book', 'is_series', 'is_picture_book', 'is_classic', 'popular_author']
            print("✅ Loaded trained age-based model")
        else:
            print("❌ No trained model found!")
            
    def extract_age_features_from_tracking(self, title):
        """Extract age features from our tracking dataset"""
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
        """Extract additional predictive features"""
        features = {}
        
        # Book type indicators
        title_lower = str(title).lower()
        notes_lower = str(notes).lower() if pd.notna(notes) else ""
        
        # Series book indicator
        features['is_series'] = 1 if any(indicator in title_lower for indicator in 
                                       ['#', 'book', 'volume', 'series']) else 0
        
        # Picture book indicators
        features['is_picture_book'] = 1 if any(indicator in notes_lower for indicator in 
                                             ['picture book', 'pages', 'illustrations', 'caldecott', 'i can read']) else 0
        
        # Classic/award book
        features['is_classic'] = 1 if any(award in notes_lower for award in 
                                        ['newbery', 'caldecott', 'award', 'medal', 'classic', 'honor']) else 0
        
        # Author popularity (rough proxy)
        popular_authors = ['dr. seuss', 'roald dahl', 'eric carle', 'maurice sendak', 'mo willems']
        features['popular_author'] = 1 if any(auth.lower() in str(author).lower() 
                                             for auth in popular_authors) else 0
        
        return features
    
    def comprehensive_model_test(self):
        """Test model against comprehensive set of verified Lexile scores"""
        
        # Load comprehensive test cases
        test_cases_path = DATA_DIR / "comprehensive_model_test_cases.csv"
        test_df = pd.read_csv(test_cases_path)
        
        print(f"🧪 COMPREHENSIVE MODEL VALIDATION")
        print(f"{'='*50}")
        print(f"Test cases to evaluate: {len(test_df)}")
        
        results = []
        successful_tests = 0
        failed_tests = 0
        
        for idx, test_case in test_df.iterrows():
            title = test_case['title']
            verified_lexile = test_case['lexile_numeric']
            book_type = test_case['book_type']
            
            if idx < 10:  # Show first 10 for debugging
                print(f"\n📖 Testing: {title}")
                print(f"   Verified Lexile: {verified_lexile}L ({book_type})")
            elif idx == 10:
                print(f"\n   ... (continuing with remaining {len(test_df) - 10} books)")
            
            # Try to find age data in our tracking dataset
            age_data = self.extract_age_features_from_tracking(title)
            
            if age_data is None or age_data[0] is None:
                if idx < 10:
                    print(f"   ❌ No age data found in tracking dataset")
                failed_tests += 1
                continue
                
            age_features, full_title, author = age_data
            min_age, max_age, min_grade, max_grade, ar_level = age_features
            
            if idx < 10:
                print(f"   Found match: {full_title}")
                print(f"   Age data: {min_age}-{max_age} years")
            
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
            
            # Prepare feature vector
            feature_vector = {
                'min_age': min_age,
                'max_age': max_age if max_age else min_age,
                'avg_age': (min_age + (max_age if max_age else min_age)) / 2,
                'age_range': (max_age if max_age else min_age) - min_age,
                'min_grade': min_grade if min_grade is not None else -1,
                'max_grade': max_grade if max_grade is not None else -1,
                'ar_level': ar_level if ar_level is not None else -1,
                'is_ad_book': 1 if book_type == 'Adult_Directed' else 0,
                **additional_features
            }
            
            # Create feature array in correct order
            X = np.array([[feature_vector[col] for col in self.feature_names]])
            
            # Make prediction
            predicted_lexile = self.model.predict(X)[0]
            error = abs(predicted_lexile - verified_lexile)
            
            if idx < 10:
                print(f"   🔮 Predicted: {predicted_lexile:.0f}L")
                print(f"   📏 Error: {error:.0f}L")
            
            # Store result
            result = {
                'title': title,
                'verified_lexile': verified_lexile,
                'predicted_lexile': predicted_lexile,
                'error': error,
                'book_type': book_type,
                'age_range': f"{min_age}-{max_age if max_age else min_age}"
            }
            results.append(result)
            successful_tests += 1
        
        if len(results) > 0:
            results_df = pd.DataFrame(results)
            
            # Calculate overall accuracy metrics
            mae = mean_absolute_error(results_df['verified_lexile'], results_df['predicted_lexile'])
            r2 = r2_score(results_df['verified_lexile'], results_df['predicted_lexile'])
            
            print(f"\n📊 COMPREHENSIVE VALIDATION RESULTS")
            print(f"{'='*45}")
            print(f"✅ Successful tests: {successful_tests}")
            print(f"❌ Failed tests: {failed_tests}")
            print(f"📈 Overall Mean Absolute Error: {mae:.1f}L")
            print(f"📈 Overall R² Score: {r2:.3f}")
            
            # Analyze by book type
            print(f"\n📚 ACCURACY BY BOOK TYPE:")
            for book_type in ['Standard_Lexile', 'Adult_Directed', 'Graphic_Novel']:
                type_results = results_df[results_df['book_type'] == book_type]
                if len(type_results) > 0:
                    type_mae = type_results['error'].mean()
                    type_r2 = r2_score(type_results['verified_lexile'], type_results['predicted_lexile']) if len(type_results) > 1 else 0
                    print(f"  {book_type}: {len(type_results)} books, {type_mae:.1f}L avg error, R²: {type_r2:.3f}")
            
            # Error distribution analysis
            print(f"\n🎯 ERROR DISTRIBUTION:")
            excellent = len(results_df[results_df['error'] < 50])
            good = len(results_df[(results_df['error'] >= 50) & (results_df['error'] < 100)])
            fair = len(results_df[(results_df['error'] >= 100) & (results_df['error'] < 200)])
            poor = len(results_df[results_df['error'] >= 200])
            
            print(f"  Excellent (<50L error): {excellent} books ({100*excellent/len(results_df):.1f}%)")
            print(f"  Good (50-100L error): {good} books ({100*good/len(results_df):.1f}%)")
            print(f"  Fair (100-200L error): {fair} books ({100*fair/len(results_df):.1f}%)")
            print(f"  Poor (>200L error): {poor} books ({100*poor/len(results_df):.1f}%)")
            
            # Show worst predictions for analysis
            print(f"\n⚠️ WORST PREDICTIONS (for analysis):")
            worst_predictions = results_df.nlargest(10, 'error')
            for _, row in worst_predictions.iterrows():
                title_short = row['title'][:40]
                verified = row['verified_lexile']
                predicted = row['predicted_lexile']
                error = row['error']
                book_type = row['book_type']
                age_range = row['age_range']
                print(f"  {title_short}: {verified:.0f}L → {predicted:.0f}L (±{error:.0f}L) {book_type} Ages {age_range}")
            
            # Show best predictions
            print(f"\n✅ BEST PREDICTIONS:")
            best_predictions = results_df.nsmallest(10, 'error')
            for _, row in best_predictions.iterrows():
                title_short = row['title'][:40]
                verified = row['verified_lexile']
                predicted = row['predicted_lexile']
                error = row['error']
                book_type = row['book_type']
                age_range = row['age_range']
                print(f"  {title_short}: {verified:.0f}L → {predicted:.0f}L (±{error:.0f}L) {book_type} Ages {age_range}")
            
            # Save comprehensive test results
            test_results_path = DATA_DIR / f"comprehensive_test_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            results_df.to_csv(test_results_path, index=False)
            print(f"\n💾 Comprehensive test results saved: {test_results_path}")
            
            return results_df, mae, r2
        
        else:
            print("❌ No successful tests completed")
            return None, None, None

def main():
    validator = ComprehensiveModelValidator()
    
    if validator.model is not None:
        results_df, mae, r2 = validator.comprehensive_model_test()
        return results_df, mae, r2
    else:
        print("❌ Cannot run tests without trained model")
        return None, None, None

if __name__ == "__main__":
    main()