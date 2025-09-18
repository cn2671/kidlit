import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import joblib

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = DATA_DIR / "models"

class EnhancedLexilePredictor:
    """Enhanced Lexile predictor with age-band awareness and AD book handling"""
    
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
        self.models = {}
        self.training_data = None
        self.feature_names = []
        
    def get_expected_lexile_range(self, min_age, max_age):
        """Get expected Lexile range based on age"""
        avg_age = (min_age + max_age) / 2
        
        # Find the most appropriate age band
        for (band_min, band_max), (lex_min, lex_max) in self.age_bands.items():
            if band_min <= avg_age <= band_max:
                return lex_min, lex_max
        
        # Fallback for very young or very old
        if avg_age < 5:
            return 0, 200
        elif avg_age > 16:
            return 1185, 1385
        else:
            # Find closest band
            closest_band = min(self.age_bands.keys(), 
                             key=lambda x: abs((x[0] + x[1])/2 - avg_age))
            return self.age_bands[closest_band]
    
    def adjust_age_for_ad_books(self, min_age, max_age, lexile_score):
        """Adjust age downward for AD books based on Lexile level"""
        # AD books are typically read by younger children than the Lexile suggests
        if lexile_score < 300:
            age_shift = 1  # Shift down 1 year for low Lexile AD books
        elif lexile_score < 600:
            age_shift = 2  # Shift down 2 years for medium Lexile AD books
        else:
            age_shift = 3  # Shift down 3 years for high Lexile AD books
            
        adjusted_min = max(2, min_age - age_shift)  # Don't go below age 2
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
        """Extract additional predictive features"""
        features = {}
        
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
                          'arnold lobel', 'cynthia rylant', 'beverly cleary', 'mary pope osborne', 'e.b. white']
        features['popular_author'] = 1 if any(auth.lower() in author_lower for auth in popular_authors) else 0
        
        # Very early reader indicators
        early_indicators = ['pre-k', 'preschool', 'toddler', 'baby', 'first words', 'board book']
        features['is_very_early'] = 1 if any(indicator in notes_lower for indicator in early_indicators) else 0
        
        # Chapter book indicator
        chapter_indicators = ['chapter book', 'early chapter', 'beginning chapter']
        features['is_chapter_book'] = 1 if any(indicator in notes_lower for indicator in chapter_indicators) else 0
        
        return features
    
    def load_training_data(self):
        """Load and prepare enhanced training data"""
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
            tracking_path = DATA_DIR / "lexile_collection_tracking.csv"
            tracking_df = pd.read_csv(tracking_path)
            book_row = tracking_df[tracking_df['title'] == full_title]
            
            if len(book_row) == 0:
                continue
                
            notes = str(book_row.iloc[0].get('notes', ''))
            additional_features = self.extract_additional_features(full_title, author, notes)
            
            # Adjust age for AD books
            original_min_age, original_max_age = min_age, max_age if max_age else min_age
            if book_type == 'Adult_Directed':
                min_age, max_age = self.adjust_age_for_ad_books(min_age, max_age if max_age else min_age, verified_lexile)
            
            # Get expected Lexile range for age validation
            expected_min, expected_max = self.get_expected_lexile_range(min_age, max_age if max_age else min_age)
            
            # Create enhanced feature vector
            enhanced_row = {
                'title': title,
                'verified_lexile': verified_lexile,
                'book_type': book_type,
                'min_age': min_age,
                'max_age': max_age if max_age else min_age,
                'original_min_age': original_min_age,
                'original_max_age': original_max_age,
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
        print(f"✅ Loaded enhanced training data: {len(self.training_data)} books")
        
        return True
    
    def train_enhanced_model(self):
        """Train enhanced models with age-band awareness"""
        if self.training_data is None:
            print("❌ No training data loaded")
            return False
        
        # Enhanced feature set
        feature_cols = [
            'min_age', 'max_age', 'avg_age', 'age_range',
            'min_grade', 'max_grade', 'ar_level',
            'expected_lexile_min', 'expected_lexile_max', 'expected_lexile_mid',
            'lexile_age_deviation',
            'is_ad_book', 'is_gn_book', 'age_adjusted_for_ad',
            'is_series', 'is_picture_book', 'is_classic', 
            'popular_author', 'is_very_early', 'is_chapter_book'
        ]
        
        self.feature_names = feature_cols
        
        # Create models directory if it doesn't exist
        MODELS_DIR.mkdir(exist_ok=True)
        
        # Train separate models for different book types
        book_types = ['Standard_Lexile', 'Adult_Directed', 'General']
        
        for book_type in book_types:
            if book_type == 'General':
                # General model trained on all data
                type_data = self.training_data.copy()
            else:
                type_data = self.training_data[self.training_data['book_type'] == book_type].copy()
            
            if len(type_data) < 5:
                print(f"⚠️ Skipping {book_type} - insufficient data ({len(type_data)} books)")
                continue
            
            print(f"\n📊 Training {book_type} model with {len(type_data)} books")
            
            X = type_data[feature_cols]
            y = type_data['verified_lexile']
            
            # Use Random Forest for better handling of complex relationships
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=42
            )
            
            # Train model
            model.fit(X, y)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X, y, cv=min(5, len(type_data)), 
                                      scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()
            
            print(f"✅ {book_type} model CV MAE: {cv_mae:.1f}L")
            
            # Feature importance
            feature_importance = sorted(zip(feature_cols, model.feature_importances_), 
                                      key=lambda x: x[1], reverse=True)
            print("📈 Top 5 features:")
            for feat, importance in feature_importance[:5]:
                print(f"   {feat}: {importance:.3f}")
            
            # Save model
            model_path = MODELS_DIR / f"enhanced_age_model_{book_type.lower()}.joblib"
            joblib.dump(model, model_path)
            self.models[book_type] = model
        
        # Save feature names
        feature_names_path = MODELS_DIR / "enhanced_feature_names.joblib"
        joblib.dump(self.feature_names, feature_names_path)
        
        print(f"\n✅ Enhanced models trained and saved")
        return True
    
    def predict_lexile(self, title, book_type='Standard_Lexile'):
        """Predict Lexile score for a book"""
        # Extract age data
        age_data = self.extract_age_features_from_tracking(title)
        if age_data is None or age_data[0] is None:
            return None, "No age data found"
            
        age_features, full_title, author = age_data
        min_age, max_age, min_grade, max_grade, ar_level = age_features
        
        if min_age is None:
            return None, "No valid age data"
        
        # Get additional features
        tracking_path = DATA_DIR / "lexile_collection_tracking.csv"
        tracking_df = pd.read_csv(tracking_path)
        book_row = tracking_df[tracking_df['title'] == full_title]
        
        if len(book_row) == 0:
            return None, "Book not found in tracking data"
            
        notes = str(book_row.iloc[0].get('notes', ''))
        additional_features = self.extract_additional_features(full_title, author, notes)
        
        # Adjust age for AD books (using estimated Lexile from age band)
        original_min_age, original_max_age = min_age, max_age if max_age else min_age
        expected_min, expected_max = self.get_expected_lexile_range(min_age, max_age if max_age else min_age)
        estimated_lexile = (expected_min + expected_max) / 2
        
        if book_type == 'Adult_Directed':
            min_age, max_age = self.adjust_age_for_ad_books(min_age, max_age if max_age else min_age, estimated_lexile)
            # Recalculate expected range with adjusted age
            expected_min, expected_max = self.get_expected_lexile_range(min_age, max_age if max_age else min_age)
        
        # Create feature vector
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
            'lexile_age_deviation': 0,  # Will be calculated during training
            'is_ad_book': 1 if book_type == 'Adult_Directed' else 0,
            'is_gn_book': 1 if book_type == 'Graphic_Novel' else 0,
            'age_adjusted_for_ad': 1 if book_type == 'Adult_Directed' else 0,
            **additional_features
        }
        
        # Select appropriate model
        if book_type in self.models:
            model = self.models[book_type]
            model_name = book_type
        else:
            model = self.models.get('General', list(self.models.values())[0])
            model_name = 'General'
        
        # Make prediction
        X = np.array([[feature_vector[col] for col in self.feature_names]])
        prediction = model.predict(X)[0]
        
        return prediction, model_name

def main():
    predictor = EnhancedLexilePredictor()
    
    # Load training data
    if not predictor.load_training_data():
        return
    
    # Train enhanced model
    if not predictor.train_enhanced_model():
        return
    
    print("\n🎉 Enhanced Lexile predictor ready!")
    return predictor

if __name__ == "__main__":
    main()