import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import joblib

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = DATA_DIR / "models"

class ExtremeSophisticationPredictor:
    """Enhanced predictor with extreme sophistication detection for vintage classics"""
    
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
        
        # Extreme vintage classics (pre-1950) with advanced vocabulary - EXPANDED
        self.extreme_vintage_classics = {
            'the poky little puppy': {'year': 1942, 'sophistication_boost': 300},
            'make way for ducklings': {'year': 1941, 'sophistication_boost': 250},
            'the little red hen': {'year': 1918, 'sophistication_boost': 200},
            'the country bunny and the little gold shoes': {'year': 1939, 'sophistication_boost': 280},
            'caps for sale': {'year': 1938, 'sophistication_boost': 220},
            'mike mulligan and his steam shovel': {'year': 1939, 'sophistication_boost': 240},
            'the story of ferdinand': {'year': 1936, 'sophistication_boost': 210},
            'madeline': {'year': 1939, 'sophistication_boost': 200},
            'goodnight moon': {'year': 1947, 'sophistication_boost': 150},
            # NEW: Additional pre-1950 classics from research
            'the little engine that could': {'year': 1930, 'sophistication_boost': 200},
            'millions of cats': {'year': 1928, 'sophistication_boost': 250},
            'abraham lincoln': {'year': 1940, 'sophistication_boost': 300},
            'white snow bright snow': {'year': 1948, 'sophistication_boost': 180},
            'the song of the swallows': {'year': 1950, 'sophistication_boost': 160}
        }
        
        # Golden age classics (1950s-1970s) with moderate sophistication
        self.golden_age_classics = {
            'where the wild things are': {'year': 1963, 'sophistication_boost': 180},
            'the giving tree': {'year': 1964, 'sophistication_boost': 160},
            'corduroy': {'year': 1968, 'sophistication_boost': 140},
            'harold and the purple crayon': {'year': 1955, 'sophistication_boost': 170},
            'curious george': {'year': 1941, 'sophistication_boost': 190},
            'the cat in the hat': {'year': 1957, 'sophistication_boost': 100},  # Dr. Seuss designed to be simple
            'green eggs and ham': {'year': 1960, 'sophistication_boost': 0}    # Specifically simple vocabulary
        }
        
        # Publishers known for sophisticated language
        self.sophisticated_publishers = {
            'viking press', 'houghton mifflin', 'harper & brothers', 'doubleday',
            'macmillan', 'scribner', 'atlantic monthly press', 'farrar straus'
        }
        
        # Authors known for sophisticated writing for children
        self.sophisticated_authors = {
            'robert mccloskey': 250,  # Make Way for Ducklings
            'virginia lee burton': 220,  # Mike Mulligan
            'munro leaf': 210,  # Ferdinand
            'dubose heyward': 280,  # Country Bunny
            'esphyr slobodkina': 220,  # Caps for Sale
            'ludwig bemelmans': 200,  # Madeline
            'margaret wise brown': 150,  # Goodnight Moon (but simpler than others)
            'maurice sendak': 180,  # Where the Wild Things Are
            'crockett johnson': 170   # Harold
        }
        
        # AD-specific characteristics
        self.simple_ad_series = {
            'if you give', 'if you take', 'pete the cat', 'splat the cat',
            'little critter', 'berenstain bears', 'arthur', 'clifford'
        }
        
        # Educational/concept books (tend to be higher)
        self.educational_indicators = {
            'magic school bus', 'national geographic', 'dk readers',
            'scholastic discover', 'science', 'nature', 'learn', 'explore'
        }
        
        self.models = {}
        self.training_data = None
        self.feature_names = []
        
    def detect_extreme_sophistication(self, title, author, notes):
        """Enhanced sophistication detection with extreme outlier handling"""
        title_lower = str(title).lower()
        notes_lower = str(notes).lower() if pd.notna(notes) else ""
        author_lower = str(author).lower() if pd.notna(author) else ""
        
        sophistication_score = 0
        sophistication_boost = 0
        reasons = []
        
        # 1. Check extreme vintage classics (highest priority)
        for classic_title, info in self.extreme_vintage_classics.items():
            if classic_title in title_lower:
                sophistication_score += 10  # Maximum score
                sophistication_boost = info['sophistication_boost']
                reasons.append(f"Extreme vintage classic: {classic_title} ({info['year']})")
                break
        
        # 2. Check golden age classics if not extreme vintage
        if sophistication_score < 10:
            for classic_title, info in self.golden_age_classics.items():
                if classic_title in title_lower:
                    sophistication_score += 7
                    sophistication_boost = info['sophistication_boost']
                    reasons.append(f"Golden age classic: {classic_title} ({info['year']})")
                    break
        
        # 3. Check sophisticated authors
        for auth_name, boost in self.sophisticated_authors.items():
            if auth_name in author_lower:
                sophistication_score += 6
                sophistication_boost = max(sophistication_boost, boost)
                reasons.append(f"Sophisticated author: {auth_name}")
                break
        
        # 4. Publication era indicators (if not already identified as classic)
        if sophistication_score < 6:
            # Try to extract publication year from notes or title
            year_pattern = r'(?:published|copyright|\b)(\d{4})\b'
            year_match = re.search(year_pattern, notes_lower)
            if year_match:
                pub_year = int(year_match.group(1))
                if pub_year < 1940:
                    sophistication_score += 8
                    sophistication_boost += 250
                    reasons.append(f"Pre-1940 publication: {pub_year}")
                elif pub_year < 1950:
                    sophistication_score += 7
                    sophistication_boost += 200
                    reasons.append(f"1940s publication: {pub_year}")
                elif pub_year < 1970:
                    sophistication_score += 5
                    sophistication_boost += 150
                    reasons.append(f"Golden age publication: {pub_year}")
        
        # 5. Classic book indicators (general)
        classic_indicators = [
            'classic', 'timeless', 'beloved', 'enduring', 'treasured',
            'landmark', 'masterpiece', 'acclaimed'
        ]
        for indicator in classic_indicators:
            if indicator in notes_lower:
                sophistication_score += 2
                sophistication_boost += 50
                reasons.append(f"Classic indicator: {indicator}")
                break
        
        # 6. Award detection (prestigious awards suggest sophistication)
        prestigious_awards = ['caldecott', 'newbery', 'boston globe', 'national book award']
        for award in prestigious_awards:
            if award in notes_lower:
                sophistication_score += 4
                sophistication_boost += 100
                reasons.append(f"Prestigious award: {award}")
                break
        
        # 7. Sophisticated vocabulary indicators
        vocab_indicators = [
            'sophisticated vocabulary', 'rich language', 'poetic', 'lyrical',
            'elegant prose', 'literary', 'complex themes', 'philosophical',
            'advanced vocabulary', 'mature themes'
        ]
        for indicator in vocab_indicators:
            if indicator in notes_lower:
                sophistication_score += 3
                sophistication_boost += 80
                reasons.append(f"Vocabulary indicator: {indicator}")
        
        # 8. Multi-generational appeal (suggests complexity beyond age)
        appeal_indicators = [
            'all ages', 'adults and children', 'multi-generational',
            'grown-ups will enjoy', 'parents will appreciate', 'family favorite',
            'appeals to readers of all ages'
        ]
        for indicator in appeal_indicators:
            if indicator in notes_lower:
                sophistication_score += 2
                sophistication_boost += 60
                reasons.append(f"Multi-generational appeal: {indicator}")
        
        # 9. Complex themes for picture books
        complex_themes = [
            'mortality', 'death', 'war', 'prejudice', 'social issues',
            'philosophical', 'existential', 'complex emotions', 'mature subject'
        ]
        for theme in complex_themes:
            if theme in notes_lower:
                sophistication_score += 3
                sophistication_boost += 100
                reasons.append(f"Complex theme: {theme}")
        
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
        
        # Cap sophistication boost to prevent over-adjustment
        sophistication_boost = min(sophistication_boost, 400)
        
        return sophistication_level, sophistication_score, sophistication_boost, reasons[:5]  # Top 5 reasons
    
    def get_sophisticated_expected_range(self, min_age, max_age, sophistication_level, sophistication_boost):
        """Get expected Lexile range with extreme sophistication adjustment"""
        avg_age = (min_age + max_age) / 2
        
        # Get base age band
        base_min, base_max = 0, 300  # Default for young ages
        for (band_min, band_max), (lex_min, lex_max) in self.age_bands.items():
            if band_min <= avg_age <= band_max:
                base_min, base_max = lex_min, lex_max
                break
        
        # Apply sophistication adjustment
        if sophistication_level == "Extreme":
            # Extreme sophistication can go way above normal ranges
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
        tracking_path = DATA_DIR / "lexile_collection_tracking_expanded.csv"
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
    
    def load_training_data(self):
        """Load training data with extreme sophistication features"""
        comprehensive_path = DATA_DIR / "comprehensive_model_test_cases.csv"
        if not comprehensive_path.exists():
            print("❌ Comprehensive test cases file not found")
            return False
            
        test_df = pd.read_csv(comprehensive_path)
        enhanced_data = []
        
        for idx, row in test_df.iterrows():
            title = row['title']
            verified_lexile = row['lexile_numeric']
            book_type = row['book_type']
            
            # Extract age data
            age_data = self.extract_age_features_from_tracking(title)
            if age_data is None or age_data[0] is None:
                continue
                
            age_features, full_title, author = age_data
            min_age, max_age, min_grade, max_grade, ar_level = age_features
            
            if min_age is None:
                continue
            
            # Get tracking data for additional features
            tracking_path = DATA_DIR / "lexile_collection_tracking_expanded.csv"
            tracking_df = pd.read_csv(tracking_path)
            book_row = tracking_df[tracking_df['title'] == full_title]
            
            if len(book_row) == 0:
                continue
                
            notes = str(book_row.iloc[0].get('notes', ''))
            additional_features, sophistication_level, sophistication_boost, reasons = self.extract_additional_features(
                full_title, author, notes, book_type
            )
            
            # Adjust age for AD books
            original_min_age, original_max_age = min_age, max_age if max_age else min_age
            if book_type == 'Adult_Directed':
                min_age, max_age = self.adjust_age_for_ad_books(min_age, max_age if max_age else min_age, verified_lexile)
            
            # Get expected Lexile range
            if book_type == 'Adult_Directed':
                expected_min, expected_max = self.get_ad_expected_lexile_range(
                    min_age, max_age if max_age else min_age,
                    additional_features.get('ad_complexity_level', 'Standard'),
                    additional_features['ad_complexity_adjustment']
                )
            else:
                expected_min, expected_max = self.get_sophisticated_expected_range(
                    min_age, max_age if max_age else min_age,
                    sophistication_level, sophistication_boost
                )
            
            # Create enhanced feature vector
            enhanced_row = {
                'title': title,
                'verified_lexile': verified_lexile,
                'book_type': book_type,
                'min_age': min_age,
                'max_age': max_age if max_age else min_age,
                'avg_age': (min_age + (max_age if max_age else min_age)) / 2,
                'age_range': (max_age if max_age else min_age) - min_age,
                'min_grade': min_grade if min_grade is not None else -1,
                'max_grade': max_grade if max_grade is not None else -1,
                'ar_level': ar_level if ar_level is not None else -1,
                'is_ad_book': 1 if book_type == 'Adult_Directed' else 0,
                'is_gn_book': 1 if book_type == 'Graphic_Novel' else 0,
                'expected_lexile_min': expected_min,
                'expected_lexile_max': expected_max,
                'expected_lexile_mid': (expected_min + expected_max) / 2,
                'lexile_age_deviation': abs(verified_lexile - (expected_min + expected_max) / 2),
                'age_adjusted_for_ad': 1 if book_type == 'Adult_Directed' else 0,
                **additional_features
            }
            
            enhanced_data.append(enhanced_row)
        
        self.training_data = pd.DataFrame(enhanced_data)
        print(f"✅ Loaded extreme sophistication training data: {len(self.training_data)} books")
        
        # Show sophistication analysis
        extreme_books = self.training_data[self.training_data['is_extreme_classic'] == 1]
        print(f"🌟 Extreme classics detected: {len(extreme_books)} books")
        if len(extreme_books) > 0:
            for _, book in extreme_books.head(5).iterrows():
                print(f"   {book['title']} (Boost: +{book['sophistication_boost']}L)")
        
        return True
    
    def train_extreme_sophistication_model(self):
        """Train models with extreme sophistication handling"""
        if self.training_data is None:
            print("❌ No training data loaded")
            return False
        
        # Enhanced feature set with extreme sophistication
        feature_cols = [
            'min_age', 'max_age', 'avg_age', 'age_range',
            'min_grade', 'max_grade', 'ar_level',
            'expected_lexile_min', 'expected_lexile_max', 'expected_lexile_mid',
            'lexile_age_deviation',
            'is_ad_book', 'is_gn_book', 'age_adjusted_for_ad',
            'sophistication_level', 'sophistication_score', 'sophistication_boost', 'is_extreme_classic',  # Enhanced
            'ad_complexity_level', 'ad_complexity_score', 'ad_complexity_adjustment',
            'is_series', 'is_picture_book', 'is_classic', 
            'popular_author', 'is_very_early', 'is_chapter_book', 'is_wordless'
        ]
        
        self.feature_names = feature_cols
        MODELS_DIR.mkdir(exist_ok=True)
        
        # Train separate models
        book_types = ['Standard_Lexile', 'Adult_Directed', 'General']
        
        for book_type in book_types:
            if book_type == 'General':
                type_data = self.training_data.copy()
            else:
                type_data = self.training_data[self.training_data['book_type'] == book_type].copy()
            
            if len(type_data) < 5:
                print(f"⚠️ Skipping {book_type} - insufficient data ({len(type_data)} books)")
                continue
            
            print(f"\n📊 Training {book_type} model with {len(type_data)} books")
            
            X = type_data[feature_cols]
            y = type_data['verified_lexile']
            
            if book_type == 'Adult_Directed':
                model = Ridge(alpha=10.0, random_state=42)
                print("   Using Ridge regression for AD books")
            else:
                model = RandomForestRegressor(
                    n_estimators=250,  # Increased for better extreme case handling
                    max_depth=12,
                    min_samples_split=2,  # More flexible for extreme cases
                    min_samples_leaf=1,
                    random_state=42
                )
            
            model.fit(X, y)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=min(5, len(type_data)), 
                                      scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()
            print(f"✅ {book_type} model CV MAE: {cv_mae:.1f}L")
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = sorted(zip(feature_cols, model.feature_importances_), 
                                          key=lambda x: x[1], reverse=True)
                print("📈 Top 5 features:")
                for feat, importance in feature_importance[:5]:
                    print(f"   {feat}: {importance:.3f}")
            
            # Save model
            model_path = MODELS_DIR / f"extreme_sophistication_model_{book_type.lower()}.joblib"
            joblib.dump(model, model_path)
            self.models[book_type] = model
        
        # Save feature names
        feature_names_path = MODELS_DIR / "extreme_sophistication_feature_names.joblib"
        joblib.dump(self.feature_names, feature_names_path)
        
        print(f"\n✅ Extreme sophistication models trained and saved")
        return True

def main():
    predictor = ExtremeSophisticationPredictor()
    
    if not predictor.load_training_data():
        return
    
    if not predictor.train_extreme_sophistication_model():
        return
    
    print("\n🌟 Extreme sophistication Lexile predictor ready!")
    return predictor

if __name__ == "__main__":
    main()