import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = DATA_DIR / "models"

class SophisticatedModelTester:
    """Test sophisticated Lexile predictor with picture book detection"""
    
    def __init__(self):
        self.age_bands = {
            (5, 7): (0, 300),      # BR to 300L
            (7, 8): (300, 600),    # 300L - 600L
            (7, 9): (420, 820),    # 420L - 820L
            (9, 11): (740, 1010),  # 740L - 1010L
            (11, 14): (925, 1185), # 925L - 1185L
            (14, 16): (1050, 1335),# 1050L - 1335L
            (16, 99): (1185, 1385) # 1185L - 1385L
        }
        
        # Classic/sophisticated picture books that read above their age level
        self.sophisticated_books = {
            'the poky little puppy', 'make way for ducklings', 'madeline',
            'where the wild things are', 'the giving tree', 'harold and the purple crayon',
            'corduroy', 'the very hungry caterpillar', 'goodnight moon',
            'the country bunny and the little gold shoes', 'stellaluna',
            'the velveteen rabbit', 'the little prince', 'alexander and the terrible',
            'mike mulligan and his steam shovel', 'the story of ferdinand',
            'caps for sale', 'curious george', 'frog and toad'
        }
        
        # Award indicators for sophisticated content
        self.sophistication_awards = {
            'caldecott', 'newbery', 'coretta scott king', 'boston globe',
            'national book award', 'pulitzer', 'hans christian andersen'
        }
        
        self.models = {}
        self.feature_names = []
        self.load_sophisticated_models()
        
    def load_sophisticated_models(self):
        """Load sophisticated models"""
        try:
            # Load feature names
            feature_names_path = MODELS_DIR / "sophisticated_feature_names.joblib"
            if feature_names_path.exists():
                self.feature_names = joblib.load(feature_names_path)
            
            # Load models
            model_files = list(MODELS_DIR.glob("sophisticated_age_model_*.joblib"))
            for model_file in model_files:
                model_name = model_file.stem.replace("sophisticated_age_model_", "")
                if model_name == "general":
                    book_type = "General"
                elif model_name == "adult_directed":
                    book_type = "Adult_Directed"
                elif model_name == "standard_lexile":
                    book_type = "Standard_Lexile"
                else:
                    continue
                    
                model = joblib.load(model_file)
                self.models[book_type] = model
                
            print(f"✅ Loaded {len(self.models)} sophisticated models")
            
        except Exception as e:
            print(f"❌ Error loading sophisticated models: {e}")
    
    def detect_sophisticated_picture_book(self, title, author, notes, publication_year=None):
        """Detect if a book is a sophisticated picture book that reads above age level"""
        sophistication_score = 0
        reasons = []
        
        title_lower = str(title).lower()
        notes_lower = str(notes).lower() if pd.notna(notes) else ""
        author_lower = str(author).lower() if pd.notna(author) else ""
        
        # 1. Check against known sophisticated books list
        for sophisticated_title in self.sophisticated_books:
            if sophisticated_title in title_lower:
                sophistication_score += 3
                reasons.append(f"Known sophisticated book: {sophisticated_title}")
                break
        
        # 2. Classic book indicators
        classic_indicators = [
            'classic', 'timeless', 'beloved', 'award-winning', 'acclaimed',
            'masterpiece', 'landmark', 'enduring', 'treasured'
        ]
        for indicator in classic_indicators:
            if indicator in notes_lower:
                sophistication_score += 1
                reasons.append(f"Classic indicator: {indicator}")
        
        # 3. Award detection (more sophisticated than basic awards)
        for award in self.sophistication_awards:
            if award in notes_lower:
                sophistication_score += 2
                reasons.append(f"Prestigious award: {award}")
        
        # 4. Publication era (pre-1980 picture books often more sophisticated)
        if publication_year and publication_year < 1980:
            sophistication_score += 1
            reasons.append(f"Vintage publication: {publication_year}")
        elif 'published' in notes_lower:
            # Try to extract publication year from notes
            year_match = re.search(r'published.*?(\d{4})', notes_lower)
            if year_match and int(year_match.group(1)) < 1980:
                sophistication_score += 1
                reasons.append(f"Vintage publication from notes")
        
        # 5. Sophisticated vocabulary indicators
        vocab_indicators = [
            'sophisticated vocabulary', 'rich language', 'poetic', 'lyrical',
            'elegant prose', 'literary', 'complex themes', 'philosophical'
        ]
        for indicator in vocab_indicators:
            if indicator in notes_lower:
                sophistication_score += 2
                reasons.append(f"Vocabulary indicator: {indicator}")
        
        # 6. Author reputation (known for sophisticated writing)
        sophisticated_authors = [
            'maurice sendak', 'chris van allsburg', 'william steig',
            'ezra jack keats', 'robert mccloskey', 'ludwig bemelmans',
            'margaret wise brown', 'crockett johnson', 'marcia brown'
        ]
        for auth in sophisticated_authors:
            if auth in author_lower:
                sophistication_score += 2
                reasons.append(f"Sophisticated author: {auth}")
                break
        
        # 7. Picture book with unusually long text
        length_indicators = [
            'lengthy text', 'substantial story', 'detailed narrative',
            'complex plot', 'multiple themes'
        ]
        for indicator in length_indicators:
            if indicator in notes_lower:
                sophistication_score += 1
                reasons.append(f"Length indicator: {indicator}")
        
        # 8. Multi-generational appeal
        appeal_indicators = [
            'all ages', 'adults and children', 'multi-generational',
            'grown-ups', 'parents will enjoy', 'family favorite'
        ]
        for indicator in appeal_indicators:
            if indicator in notes_lower:
                sophistication_score += 1
                reasons.append(f"Multi-generational appeal: {indicator}")
        
        # Determine sophistication level
        is_sophisticated = sophistication_score >= 3
        sophistication_level = min(sophistication_score, 10)  # Cap at 10
        
        return is_sophisticated, sophistication_level, reasons
    
    def get_expected_lexile_range(self, min_age, max_age, is_sophisticated=False, sophistication_level=0):
        """Get expected Lexile range based on age, with sophistication adjustment"""
        avg_age = (min_age + max_age) / 2
        
        # Find the most appropriate age band
        base_min, base_max = None, None
        for (band_min, band_max), (lex_min, lex_max) in self.age_bands.items():
            if band_min <= avg_age <= band_max:
                base_min, base_max = lex_min, lex_max
                break
        
        # Fallback for very young or very old
        if base_min is None:
            if avg_age < 5:
                base_min, base_max = 0, 200
            elif avg_age > 16:
                base_min, base_max = 1185, 1385
            else:
                # Find closest band
                closest_band = min(self.age_bands.keys(), 
                                 key=lambda x: abs((x[0] + x[1])/2 - avg_age))
                base_min, base_max = self.age_bands[closest_band]
        
        # Adjust for sophistication
        if is_sophisticated and sophistication_level > 0:
            # Sophisticated books can read 100-400L higher than typical for age
            sophistication_boost = min(sophistication_level * 40, 400)
            adjusted_max = base_max + sophistication_boost
            adjusted_min = base_min + min(sophistication_boost // 2, 200)
            return adjusted_min, adjusted_max
        
        return base_min, base_max
    
    def adjust_age_for_ad_books(self, min_age, max_age, lexile_score):
        """Adjust age downward for AD books based on Lexile level"""
        if lexile_score < 300:
            age_shift = 1
        elif lexile_score < 600:
            age_shift = 2
        else:
            age_shift = 3
            
        adjusted_min = max(2, min_age - age_shift)
        adjusted_max = max(adjusted_min, max_age - age_shift)
        
        return adjusted_min, adjusted_max
            
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
        """Extract additional predictive features with sophistication detection"""
        features = {}
        
        title_lower = str(title).lower()
        notes_lower = str(notes).lower() if pd.notna(notes) else ""
        author_lower = str(author).lower() if pd.notna(author) else ""
        
        # Sophistication detection
        is_sophisticated, sophistication_level, reasons = self.detect_sophisticated_picture_book(
            title, author, notes
        )
        features['is_sophisticated'] = 1 if is_sophisticated else 0
        features['sophistication_level'] = sophistication_level
        
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
                          'arnold lobel', 'cynthia rylant', 'beverly cleary', 'mary pope osborne', 'e.b. white']
        features['popular_author'] = 1 if any(auth.lower() in author_lower for auth in popular_authors) else 0
        
        # Very early reader indicators
        early_indicators = ['pre-k', 'preschool', 'toddler', 'baby', 'first words', 'board book']
        features['is_very_early'] = 1 if any(indicator in notes_lower for indicator in early_indicators) else 0
        
        # Chapter book indicator
        chapter_indicators = ['chapter book', 'early chapter', 'beginning chapter']
        features['is_chapter_book'] = 1 if any(indicator in notes_lower for indicator in chapter_indicators) else 0
        
        # Wordless book indicator (should have very low Lexile)
        features['is_wordless'] = 1 if 'wordless' in notes_lower else 0
        
        return features, is_sophisticated, sophistication_level, reasons
    
    def predict_with_sophisticated_model(self, features, book_type='Standard_Lexile'):
        """Make prediction using appropriate sophisticated model"""
        
        # Select appropriate model
        if book_type in self.models:
            model = self.models[book_type]
            model_name = book_type
        else:
            model = self.models.get('General', list(self.models.values())[0])
            model_name = 'General'
        
        # Create feature array in correct order
        X = np.array([[features[col] for col in self.feature_names]])
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        return prediction, model_name
    
    def test_sophisticated_model(self):
        """Test sophisticated model against new Lexile scores"""
        
        # Load new test scores
        new_scores_path = DATA_DIR / "additional_test_scores.csv"
        test_df = pd.read_csv(new_scores_path)
        
        print(f"🧪 TESTING SOPHISTICATED PICTURE BOOK MODEL")
        print(f"{'='*55}") 
        print(f"New test cases: {len(test_df)}")
        
        results = []
        successful_tests = 0
        failed_tests = 0
        
        for idx, test_case in test_df.iterrows():
            title = test_case['title']
            
            # Skip books without Lexile scores (just age data)
            if pd.isna(test_case.get('verified_lexile')) or test_case.get('verified_lexile') == '' or pd.isna(test_case.get('lexile_numeric')):
                print(f"\n📖 Skipping {title} - only age data provided")
                continue
                
            verified_lexile = float(test_case['lexile_numeric'])
            book_type = test_case['book_type']
            
            print(f"\n📖 Testing: {title}")
            print(f"   Verified: {verified_lexile}L ({book_type})")
            
            # Try to find age data in our tracking dataset
            age_data = self.extract_age_features_from_tracking(title)
            
            if age_data is None or age_data[0] is None:
                print(f"   ❌ No age data found in tracking dataset")
                failed_tests += 1
                continue
                
            age_features, full_title, author = age_data
            min_age, max_age, min_grade, max_grade, ar_level = age_features
            
            # Handle missing age data
            if min_age is None:
                print(f"   ❌ No valid age data found")
                failed_tests += 1
                continue
            
            print(f"   Found: {full_title}")
            print(f"   Original Age: {min_age}-{max_age if max_age else min_age} years")
            
            # Get tracking data for additional features
            tracking_path = DATA_DIR / "lexile_collection_tracking.csv"
            tracking_df = pd.read_csv(tracking_path)
            book_row = tracking_df[tracking_df['title'] == full_title]
            
            if len(book_row) == 0:
                print(f"   ❌ Book not found in tracking data")
                failed_tests += 1
                continue
                
            notes = str(book_row.iloc[0].get('notes', ''))
            additional_features, is_sophisticated, sophistication_level, reasons = self.extract_additional_features(full_title, author, notes)
            
            # Show sophistication detection
            if is_sophisticated:
                print(f"   🎭 Sophisticated Book Detected (Level {sophistication_level}):")
                for reason in reasons[:3]:  # Show top 3 reasons
                    print(f"      • {reason}")
            
            # Adjust age for AD books
            original_min_age, original_max_age = min_age, max_age if max_age else min_age
            if book_type == 'Adult_Directed':
                min_age, max_age = self.adjust_age_for_ad_books(min_age, max_age if max_age else min_age, verified_lexile)
                print(f"   AD Adjusted Age: {min_age}-{max_age} years (shifted younger)")
            
            # Get expected Lexile range with sophistication adjustment
            expected_min, expected_max = self.get_expected_lexile_range(
                min_age, max_age if max_age else min_age, 
                is_sophisticated, sophistication_level
            )
            if is_sophisticated:
                print(f"   Expected Range: {expected_min}L-{expected_max}L for age {min_age}-{max_age} (sophistication adjusted)")
            else:
                print(f"   Expected Range: {expected_min}L-{expected_max}L for age {min_age}-{max_age}")
            
            # Prepare sophisticated feature vector
            feature_vector = {
                'min_age': min_age,
                'max_age': max_age if max_age else min_age,
                'avg_age': (min_age + (max_age if max_age else min_age)) / 2,
                'age_range': (max_age if max_age else min_age) - min_age,
                'min_grade': min_grade if min_grade is not None else -1,
                'max_grade': max_grade if max_grade is not None else -1,
                'ar_level': ar_level if ar_level is not None else -1,
                'expected_lexile_min': expected_min,
                'expected_lexile_max': expected_max,
                'expected_lexile_mid': (expected_min + expected_max) / 2,
                'lexile_age_deviation': abs(verified_lexile - (expected_min + expected_max) / 2),
                'is_ad_book': 1 if book_type == 'Adult_Directed' else 0,
                'is_gn_book': 1 if book_type == 'Graphic_Novel' else 0,
                'age_adjusted_for_ad': 1 if book_type == 'Adult_Directed' else 0,
                **additional_features
            }
            
            # Make prediction with sophisticated model
            predicted_lexile, model_used = self.predict_with_sophisticated_model(feature_vector, book_type)
            error = abs(predicted_lexile - verified_lexile)
            
            print(f"   🔮 Predicted: {predicted_lexile:.0f}L (using {model_used} model)")
            print(f"   📏 Error: {error:.0f}L")
            
            # Compare with original ML estimate
            original_ml = book_row.iloc[0]['current_ml_estimate'] if len(book_row) > 0 else None
            if pd.notna(original_ml):
                original_error = abs(original_ml - verified_lexile)
                improvement = original_error - error
                print(f"   📊 Original ML: {original_ml:.0f}L (±{original_error:.0f}L)")
                print(f"   📈 Improvement: {improvement:+.0f}L")
            
            # Store result
            result = {
                'title': title,
                'verified_lexile': verified_lexile,
                'predicted_lexile': predicted_lexile,
                'error': error,
                'book_type': book_type,
                'model_used': model_used,
                'original_ml_estimate': original_ml if pd.notna(original_ml) else None,
                'original_error': original_error if pd.notna(original_ml) else None,
                'improvement': improvement if pd.notna(original_ml) else None,
                'age_range': f"{min_age}-{max_age if max_age else min_age}",
                'expected_range': f"{expected_min}-{expected_max}",
                'is_sophisticated': is_sophisticated,
                'sophistication_level': sophistication_level,
                'age_adjusted_for_ad': book_type == 'Adult_Directed'
            }
            results.append(result)
            successful_tests += 1
        
        if len(results) > 0:
            results_df = pd.DataFrame(results)
            
            # Calculate overall accuracy metrics
            mae = mean_absolute_error(results_df['verified_lexile'], results_df['predicted_lexile'])
            r2 = r2_score(results_df['verified_lexile'], results_df['predicted_lexile'])
            
            print(f"\n📊 SOPHISTICATED MODEL TEST RESULTS")
            print(f"{'='*45}")
            print(f"✅ Successful tests: {successful_tests}")
            print(f"❌ Failed tests: {failed_tests}")
            print(f"📈 Overall MAE: {mae:.1f}L")
            print(f"📈 Overall R²: {r2:.3f}")
            
            # Compare with previous model
            previous_mae = 88.9  # From enhanced model
            improvement = ((previous_mae - mae) / previous_mae) * 100
            print(f"🚀 Improvement over enhanced model: {improvement:.1f}% {'better' if improvement > 0 else 'change'}")
            
            # Compare with original ML estimates
            valid_improvements = results_df[results_df['improvement'].notna()]
            if len(valid_improvements) > 0:
                avg_improvement = valid_improvements['improvement'].mean()
                print(f"🚀 Average improvement over original ML: {avg_improvement:+.1f}L")
            
            # Analyze sophisticated book performance
            sophisticated_books = results_df[results_df['is_sophisticated'] == True]
            if len(sophisticated_books) > 0:
                soph_mae = sophisticated_books['error'].mean()
                print(f"🎭 Sophisticated books MAE: {soph_mae:.1f}L ({len(sophisticated_books)} books)")
            
            # Analyze by book type
            print(f"\n📚 ACCURACY BY BOOK TYPE:")
            for book_type in ['Standard_Lexile', 'Adult_Directed']:
                type_results = results_df[results_df['book_type'] == book_type]
                if len(type_results) > 0:
                    type_mae = type_results['error'].mean()
                    print(f"  {book_type}: {len(type_results)} books, {type_mae:.1f}L avg error")
            
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
            
            # Show all results with sophistication info
            print(f"\n📋 DETAILED RESULTS:")
            for _, row in results_df.iterrows():
                title_short = row['title'][:35]
                verified = row['verified_lexile']
                predicted = row['predicted_lexile']
                error = row['error']
                book_type = row['book_type']
                expected = row['expected_range']
                
                status = "✅" if error < 50 else "⚠️" if error < 100 else "❌"
                ad_marker = " (AD adj)" if row['age_adjusted_for_ad'] else ""
                soph_marker = f" 🎭{row['sophistication_level']}" if row['is_sophisticated'] else ""
                print(f"  {status} {title_short}: {verified:.0f}L → {predicted:.0f}L (±{error:.0f}L) [{book_type}]{ad_marker}{soph_marker}")
                print(f"      Expected: {expected}L")
            
            # Save results
            results_path = DATA_DIR / f"sophisticated_model_test_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            results_df.to_csv(results_path, index=False)
            print(f"\n💾 Results saved: {results_path}")
            
            return results_df, mae, r2
        
        else:
            print("❌ No successful tests completed")
            return None, None, None

def main():
    tester = SophisticatedModelTester()
    
    if len(tester.models) > 0:
        results_df, mae, r2 = tester.test_sophisticated_model()
        return results_df, mae, r2
    else:
        print("❌ No sophisticated models found")
        return None, None, None

if __name__ == "__main__":
    main()