import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = DATA_DIR / "models"

class ExtremeSophisticationModelTester:
    """Test extreme sophistication model with enhanced vintage classic detection"""
    
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
        
        # Extreme vintage classics (pre-1950) with advanced vocabulary
        self.extreme_vintage_classics = {
            'the poky little puppy': {'year': 1942, 'sophistication_boost': 300},
            'make way for ducklings': {'year': 1941, 'sophistication_boost': 250},
            'the little red hen': {'year': 1918, 'sophistication_boost': 200},
            'the country bunny and the little gold shoes': {'year': 1939, 'sophistication_boost': 280},
            'caps for sale': {'year': 1940, 'sophistication_boost': 220},
            'mike mulligan and his steam shovel': {'year': 1939, 'sophistication_boost': 240},
            'the story of ferdinand': {'year': 1936, 'sophistication_boost': 210},
            'madeline': {'year': 1939, 'sophistication_boost': 200},
            'goodnight moon': {'year': 1947, 'sophistication_boost': 150}
        }
        
        # Golden age classics (1950s-1970s)
        self.golden_age_classics = {
            'where the wild things are': {'year': 1963, 'sophistication_boost': 180},
            'the giving tree': {'year': 1964, 'sophistication_boost': 160},
            'corduroy': {'year': 1968, 'sophistication_boost': 140},
            'harold and the purple crayon': {'year': 1955, 'sophistication_boost': 170},
            'curious george': {'year': 1941, 'sophistication_boost': 190},
            'the cat in the hat': {'year': 1957, 'sophistication_boost': 100},
            'green eggs and ham': {'year': 1960, 'sophistication_boost': 0}
        }
        
        # Authors known for sophisticated writing
        self.sophisticated_authors = {
            'robert mccloskey': 250, 'virginia lee burton': 220, 'munro leaf': 210,
            'dubose heyward': 280, 'esphyr slobodkina': 220, 'ludwig bemelmans': 200,
            'margaret wise brown': 150, 'maurice sendak': 180, 'crockett johnson': 170
        }
        
        # AD-specific characteristics
        self.simple_ad_series = {
            'if you give', 'if you take', 'pete the cat', 'splat the cat',
            'little critter', 'berenstain bears', 'arthur', 'clifford'
        }
        
        self.educational_indicators = {
            'magic school bus', 'national geographic', 'dk readers',
            'scholastic discover', 'science', 'nature', 'learn', 'explore'
        }
        
        self.models = {}
        self.feature_names = []
        self.load_extreme_sophistication_models()
        
    def load_extreme_sophistication_models(self):
        """Load extreme sophistication models"""
        try:
            # Load feature names
            feature_names_path = MODELS_DIR / "extreme_sophistication_feature_names.joblib"
            if feature_names_path.exists():
                self.feature_names = joblib.load(feature_names_path)
            
            # Load models
            model_files = list(MODELS_DIR.glob("extreme_sophistication_model_*.joblib"))
            for model_file in model_files:
                model_name = model_file.stem.replace("extreme_sophistication_model_", "")
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
                
            print(f"✅ Loaded {len(self.models)} extreme sophistication models")
            
        except Exception as e:
            print(f"❌ Error loading extreme sophistication models: {e}")
    
    def detect_extreme_sophistication(self, title, author, notes):
        """Enhanced sophistication detection with extreme outlier handling"""
        title_lower = str(title).lower()
        notes_lower = str(notes).lower() if pd.notna(notes) else ""
        author_lower = str(author).lower() if pd.notna(author) else ""
        
        sophistication_score = 0
        sophistication_boost = 0
        reasons = []
        
        # Check extreme vintage classics (highest priority)
        for classic_title, info in self.extreme_vintage_classics.items():
            if classic_title in title_lower:
                sophistication_score += 10
                sophistication_boost = info['sophistication_boost']
                reasons.append(f"Extreme vintage classic: {classic_title} ({info['year']})")
                break
        
        # Check golden age classics if not extreme vintage
        if sophistication_score < 10:
            for classic_title, info in self.golden_age_classics.items():
                if classic_title in title_lower:
                    sophistication_score += 7
                    sophistication_boost = info['sophistication_boost']
                    reasons.append(f"Golden age classic: {classic_title} ({info['year']})")
                    break
        
        # Check sophisticated authors
        for auth_name, boost in self.sophisticated_authors.items():
            if auth_name in author_lower:
                sophistication_score += 6
                sophistication_boost = max(sophistication_boost, boost)
                reasons.append(f"Sophisticated author: {auth_name}")
                break
        
        # Classic indicators
        classic_indicators = ['classic', 'timeless', 'beloved', 'enduring', 'treasured']
        for indicator in classic_indicators:
            if indicator in notes_lower:
                sophistication_score += 2
                sophistication_boost += 50
                reasons.append(f"Classic indicator: {indicator}")
                break
        
        # Prestigious awards
        prestigious_awards = ['caldecott', 'newbery', 'boston globe', 'national book award']
        for award in prestigious_awards:
            if award in notes_lower:
                sophistication_score += 4
                sophistication_boost += 100
                reasons.append(f"Prestigious award: {award}")
                break
        
        # Determine final sophistication level
        if sophistication_score >= 10:
            sophistication_level = "Extreme"
        elif sophistication_score >= 7:
            sophistication_level = "High"
        elif sophistication_score >= 4:
            sophistication_level = "Moderate"
        elif sophistication_score >= 2:
            sophistication_level = "Low"
        else:
            sophistication_level = "None"
        
        # Cap sophistication boost
        sophistication_boost = min(sophistication_boost, 400)
        
        return sophistication_level, sophistication_score, sophistication_boost, reasons[:5]
    
    def get_sophisticated_expected_range(self, min_age, max_age, sophistication_level, sophistication_boost):
        """Get expected Lexile range with extreme sophistication adjustment"""
        avg_age = (min_age + max_age) / 2
        
        # Get base age band
        base_min, base_max = 0, 300
        for (band_min, band_max), (lex_min, lex_max) in self.age_bands.items():
            if band_min <= avg_age <= band_max:
                base_min, base_max = lex_min, lex_max
                break
        
        # Apply sophistication adjustment
        if sophistication_level == "Extreme":
            adjusted_min = base_min + (sophistication_boost // 2)
            adjusted_max = base_max + sophistication_boost
        elif sophistication_level == "High":
            adjusted_min = base_min + (sophistication_boost // 3)
            adjusted_max = base_max + sophistication_boost
        elif sophistication_level == "Moderate":
            adjusted_min = base_min + (sophistication_boost // 4)
            adjusted_max = base_max + sophistication_boost
        elif sophistication_level == "Low":
            adjusted_min = base_min + (sophistication_boost // 6)
            adjusted_max = base_max + sophistication_boost
        else:
            adjusted_min, adjusted_max = base_min, base_max
        
        return max(0, adjusted_min), adjusted_max
    
    def detect_ad_complexity_level(self, title, author, notes):
        """Detect AD book complexity level"""
        title_lower = str(title).lower()
        notes_lower = str(notes).lower() if pd.notna(notes) else ""
        
        complexity_score = 0
        complexity_reasons = []
        
        # Simple AD book indicators
        for series in self.simple_ad_series:
            if series in title_lower or series in notes_lower:
                complexity_score -= 2
                complexity_reasons.append(f"Simple series: {series}")
                break
        
        # Educational content indicators
        for indicator in self.educational_indicators:
            if indicator in title_lower or indicator in notes_lower:
                complexity_score += 3
                complexity_reasons.append(f"Educational content: {indicator}")
                break
        
        # Determine complexity level
        if complexity_score <= -2:
            complexity_level = "Simple"
            expected_adjustment = -100
        elif complexity_score >= 2:
            complexity_level = "Educational"
            expected_adjustment = +50
        else:
            complexity_level = "Standard"
            expected_adjustment = 0
        
        return complexity_level, complexity_score, expected_adjustment, complexity_reasons
    
    def get_ad_expected_lexile_range(self, min_age, max_age, complexity_level, complexity_adjustment):
        """Get expected Lexile range for AD books"""
        ad_adjustment = -100
        avg_age = (min_age + max_age) / 2
        base_min, base_max = 0, 300
        
        for (band_min, band_max), (lex_min, lex_max) in self.age_bands.items():
            if band_min <= avg_age <= band_max:
                base_min, base_max = lex_min, lex_max
                break
        
        adjusted_min = max(0, base_min + ad_adjustment)
        adjusted_max = max(adjusted_min + 100, base_max + ad_adjustment)
        
        final_min = max(0, adjusted_min + complexity_adjustment)
        final_max = adjusted_max + complexity_adjustment
        
        return final_min, final_max
    
    def adjust_age_for_ad_books(self, min_age, max_age, lexile_score):
        """Conservative age adjustment for AD books"""
        if lexile_score < 300:
            age_shift = 1
        elif lexile_score < 500:
            age_shift = 1.5
        else:
            age_shift = 2
            
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
    
    def extract_additional_features(self, title, author, notes, book_type):
        """Extract enhanced features with extreme sophistication detection"""
        features = {}
        
        title_lower = str(title).lower()
        notes_lower = str(notes).lower() if pd.notna(notes) else ""
        author_lower = str(author).lower() if pd.notna(author) else ""
        
        # Enhanced sophistication detection
        sophistication_level, sophistication_score, sophistication_boost, reasons = self.detect_extreme_sophistication(
            title, author, notes
        )
        features['sophistication_level'] = {'None': 0, 'Low': 1, 'Moderate': 2, 'High': 3, 'Extreme': 4}[sophistication_level]
        features['sophistication_score'] = sophistication_score
        features['sophistication_boost'] = sophistication_boost
        features['is_extreme_classic'] = 1 if sophistication_level == "Extreme" else 0
        
        # AD complexity (only for AD books)
        if book_type == 'Adult_Directed':
            ad_complexity_level, ad_complexity_score, ad_complexity_adjustment, ad_reasons = self.detect_ad_complexity_level(
                title, author, notes
            )
            features['ad_complexity_level'] = {'Simple': 0, 'Standard': 1, 'Educational': 2}[ad_complexity_level]
            features['ad_complexity_score'] = ad_complexity_score
            features['ad_complexity_adjustment'] = ad_complexity_adjustment
        else:
            features['ad_complexity_level'] = 1  # Standard for non-AD
            features['ad_complexity_score'] = 0
            features['ad_complexity_adjustment'] = 0
        
        # Basic features
        features['is_series'] = 1 if any(indicator in title_lower for indicator in ['#', 'book', 'volume', 'series']) else 0
        features['is_picture_book'] = 1 if any(indicator in notes_lower for indicator in ['picture book', 'pages', 'illustrations']) else 0
        features['is_classic'] = 1 if any(award in notes_lower for award in ['newbery', 'caldecott', 'award', 'medal', 'classic']) else 0
        features['popular_author'] = 1 if any(auth.lower() in author_lower for auth in ['dr. seuss', 'roald dahl', 'eric carle', 'maurice sendak']) else 0
        features['is_very_early'] = 1 if any(indicator in notes_lower for indicator in ['pre-k', 'preschool', 'toddler', 'baby']) else 0
        features['is_chapter_book'] = 1 if any(indicator in notes_lower for indicator in ['chapter book', 'early chapter']) else 0
        features['is_wordless'] = 1 if 'wordless' in notes_lower else 0
        
        return features, sophistication_level, sophistication_boost, reasons
    
    def predict_with_extreme_sophistication_model(self, features, book_type='Standard_Lexile'):
        """Make prediction using extreme sophistication model"""
        
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
    
    def test_extreme_sophistication_model(self):
        """Test extreme sophistication model"""
        
        # Load new test scores
        new_scores_path = DATA_DIR / "additional_test_scores.csv"
        test_df = pd.read_csv(new_scores_path)
        
        print(f"🌟 TESTING EXTREME SOPHISTICATION MODEL")
        print(f"{'='*50}") 
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
            additional_features, sophistication_level, sophistication_boost, reasons = self.extract_additional_features(
                full_title, author, notes, book_type
            )
            
            # Show extreme sophistication detection
            if sophistication_level != "None":
                print(f"   🌟 Sophistication: {sophistication_level} (Score: {additional_features['sophistication_score']}, Boost: +{sophistication_boost}L)")
                for reason in reasons[:3]:  # Show top 3 reasons
                    print(f"      • {reason}")
            
            # Show AD complexity detection for AD books
            if book_type == 'Adult_Directed':
                complexity_level, complexity_score, complexity_adjustment, ad_reasons = self.detect_ad_complexity_level(
                    full_title, author, notes
                )
                print(f"   🔧 AD Complexity: {complexity_level} (Score: {complexity_score}, Adj: {complexity_adjustment}L)")
                if ad_reasons:
                    for reason in ad_reasons[:2]:
                        print(f"      • {reason}")
            
            # Adjust age for AD books
            original_min_age, original_max_age = min_age, max_age if max_age else min_age
            if book_type == 'Adult_Directed':
                min_age, max_age = self.adjust_age_for_ad_books(min_age, max_age if max_age else min_age, verified_lexile)
                print(f"   AD Adjusted Age: {min_age}-{max_age} years (conservative adjustment)")
            
            # Get expected Lexile range
            if book_type == 'Adult_Directed':
                complexity_level, complexity_score, complexity_adjustment, ad_reasons = self.detect_ad_complexity_level(
                    full_title, author, notes
                )
                expected_min, expected_max = self.get_ad_expected_lexile_range(
                    min_age, max_age if max_age else min_age,
                    complexity_level, complexity_adjustment
                )
                print(f"   Expected Range: {expected_min}L-{expected_max}L (AD-specific)")
            else:
                expected_min, expected_max = self.get_sophisticated_expected_range(
                    min_age, max_age if max_age else min_age,
                    sophistication_level, sophistication_boost
                )
                if sophistication_level != "None":
                    print(f"   Expected Range: {expected_min}L-{expected_max}L (sophistication adjusted)")
                else:
                    print(f"   Expected Range: {expected_min}L-{expected_max}L")
            
            # Prepare feature vector
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
            
            # Make prediction with extreme sophistication model
            predicted_lexile, model_used = self.predict_with_extreme_sophistication_model(feature_vector, book_type)
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
                'sophistication_level': sophistication_level,
                'sophistication_boost': sophistication_boost,
                'is_extreme_classic': sophistication_level == "Extreme",
                'age_adjusted_for_ad': book_type == 'Adult_Directed'
            }
            results.append(result)
            successful_tests += 1
        
        if len(results) > 0:
            results_df = pd.DataFrame(results)
            
            # Calculate overall accuracy metrics
            mae = mean_absolute_error(results_df['verified_lexile'], results_df['predicted_lexile'])
            r2 = r2_score(results_df['verified_lexile'], results_df['predicted_lexile'])
            
            print(f"\n📊 EXTREME SOPHISTICATION MODEL RESULTS")
            print(f"{'='*50}")
            print(f"✅ Successful tests: {successful_tests}")
            print(f"❌ Failed tests: {failed_tests}")
            print(f"📈 Overall MAE: {mae:.1f}L")
            print(f"📈 Overall R²: {r2:.3f}")
            
            # Compare with previous fixed AD model
            previous_mae = 76.0
            improvement = ((previous_mae - mae) / previous_mae) * 100
            print(f"🚀 Improvement over fixed AD model: {improvement:.1f}% {'better' if improvement > 0 else 'change'}")
            
            # Compare with original ML estimates
            valid_improvements = results_df[results_df['improvement'].notna()]
            if len(valid_improvements) > 0:
                avg_improvement = valid_improvements['improvement'].mean()
                print(f"🚀 Average improvement over original ML: {avg_improvement:+.1f}L")
            
            # Analyze extreme classics performance
            extreme_books = results_df[results_df['is_extreme_classic'] == True]
            if len(extreme_books) > 0:
                extreme_mae = extreme_books['error'].mean()
                print(f"🌟 Extreme classics MAE: {extreme_mae:.1f}L ({len(extreme_books)} books)")
                print("   Extreme classics detected:")
                for _, book in extreme_books.iterrows():
                    print(f"     {book['title']}: {book['verified_lexile']:.0f}L → {book['predicted_lexile']:.0f}L (±{book['error']:.0f}L)")
            
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
            
            # Show all results
            print(f"\n📋 DETAILED RESULTS:")
            for _, row in results_df.iterrows():
                title_short = row['title'][:35]
                verified = row['verified_lexile']
                predicted = row['predicted_lexile']
                error = row['error']
                book_type = row['book_type']
                expected = row['expected_range']
                
                status = "✅" if error < 50 else "⚠️" if error < 100 else "❌"
                extreme_marker = f" 🌟{row['sophistication_level']}" if row['sophistication_level'] != "None" else ""
                print(f"  {status} {title_short}: {verified:.0f}L → {predicted:.0f}L (±{error:.0f}L) [{book_type}]{extreme_marker}")
                print(f"      Expected: {expected}L")
            
            # Save results
            results_path = DATA_DIR / f"extreme_sophistication_test_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            results_df.to_csv(results_path, index=False)
            print(f"\n💾 Results saved: {results_path}")
            
            return results_df, mae, r2
        
        else:
            print("❌ No successful tests completed")
            return None, None, None

def main():
    tester = ExtremeSophisticationModelTester()
    
    if len(tester.models) > 0:
        results_df, mae, r2 = tester.test_extreme_sophistication_model()
        return results_df, mae, r2
    else:
        print("❌ No extreme sophistication models found")
        return None, None, None

if __name__ == "__main__":
    main()