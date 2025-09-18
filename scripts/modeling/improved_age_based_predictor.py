import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

class ImprovedAgeBasedPredictor:
    """Improved age-based Lexile predictor with book type handling"""
    
    def __init__(self):
        self.models = {}  # Separate models for each book type
        self.scalers = {}  # Separate scalers for each book type
        self.feature_names = []
        self.training_data = None
        
    def extract_age_features(self, notes_text):
        """Extract age and grade information from notes"""
        if pd.isna(notes_text):
            return None, None, None, None
            
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
        
        return min_age, max_age, min_grade, max_grade
    
    def extract_ar_level(self, notes_text):
        """Extract AR (Accelerated Reader) level"""
        if pd.isna(notes_text):
            return None
            
        notes_lower = str(notes_text).lower()
        ar_match = re.search(r'ar\s*:?\s*(\d+\.?\d*)', notes_lower)
        if ar_match:
            return float(ar_match.group(1))
        
        # Try ATOS format
        atos_match = re.search(r'atos\s+(?:level\s+)?(\d+\.?\d*)', notes_lower)
        if atos_match:
            return float(atos_match.group(1))
            
        return None
    
    def extract_additional_features(self, title, author, notes):
        """Extract additional predictive features"""
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
    
    def prepare_improved_training_data(self):
        """Prepare enhanced training data from verified books"""
        
        tracking_path = DATA_DIR / "lexile_collection_tracking.csv"
        df = pd.read_csv(tracking_path)
        
        # Get completed books with verified scores
        completed = df[df['status'] == 'Complete'].copy()
        
        print(f"📊 PREPARING IMPROVED TRAINING DATA")
        print(f"{'='*50}")
        print(f"Processing {len(completed)} verified books...")
        
        training_rows = []
        
        for _, row in completed.iterrows():
            # Extract age and grade features
            min_age, max_age, min_grade, max_grade = self.extract_age_features(row['notes'])
            ar_level = self.extract_ar_level(row['notes'])
            additional_features = self.extract_additional_features(row['title'], row['author'], row['notes'])
            
            # Skip if we don't have age data
            if min_age is None:
                print(f"  ⚠️  Skipping {row['title'][:30]} - no age data")
                continue
            
            # Determine book type with enhanced logic
            book_type = 'Standard_Lexile'  # default
            if pd.notna(row['book_type']):
                book_type = row['book_type']
            elif pd.notna(row['lexile_prefix']):
                if row['lexile_prefix'] == 'AD':
                    book_type = 'Adult_Directed'
                elif row['lexile_prefix'] == 'GN':
                    book_type = 'Graphic_Novel'
            
            # Prepare feature vector
            features = {
                'title': row['title'],
                'verified_lexile': row['lexile_numeric'],
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
                **additional_features
            }
            
            training_rows.append(features)
            
            print(f"  ✅ {row['title'][:30]}: Age {min_age}-{max_age}, Lexile {row['lexile_numeric']}L ({book_type})")
        
        self.training_data = pd.DataFrame(training_rows)
        
        print(f"\n📈 TRAINING DATA SUMMARY:")
        print(f"  Books with age data: {len(self.training_data)}")
        print(f"  Age range: {self.training_data['min_age'].min()}-{self.training_data['max_age'].max()} years")
        print(f"  Lexile range: {self.training_data['verified_lexile'].min():.0f}-{self.training_data['verified_lexile'].max():.0f}L")
        
        # Show book type distribution
        book_type_counts = self.training_data['book_type'].value_counts()
        print(f"\n📚 BOOK TYPE DISTRIBUTION:")
        for book_type, count in book_type_counts.items():
            avg_lexile = self.training_data[self.training_data['book_type'] == book_type]['verified_lexile'].mean()
            print(f"  {book_type}: {count} books (avg {avg_lexile:.0f}L)")
        
        return self.training_data
    
    def train_improved_model(self):
        """Train separate models for different book types"""
        
        if self.training_data is None:
            print("❌ No training data prepared!")
            return None
        
        print(f"\n🤖 TRAINING IMPROVED BOOK-TYPE-AWARE MODELS")
        print(f"{'='*55}")
        
        # Define feature columns (enhanced)
        feature_cols = ['min_age', 'max_age', 'avg_age', 'age_range', 'min_grade', 'max_grade', 
                       'ar_level', 'is_ad_book', 'is_gn_book', 'is_series', 'is_picture_book', 
                       'is_classic', 'popular_author', 'is_very_early', 'is_chapter_book']
        
        self.feature_names = feature_cols
        
        # Train separate models for each book type
        book_types = self.training_data['book_type'].unique()
        
        for book_type in book_types:
            print(f"\n🎯 Training model for {book_type}:")
            
            # Get data for this book type
            type_data = self.training_data[self.training_data['book_type'] == book_type].copy()
            
            if len(type_data) < 3:
                print(f"  ⚠️  Insufficient data ({len(type_data)} books) - using general model")
                continue
            
            X = type_data[feature_cols]
            y = type_data['verified_lexile']
            
            print(f"  Training samples: {len(X)}")
            print(f"  Lexile range: {y.min():.0f}-{y.max():.0f}L")
            
            # Try different models
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Linear Regression': LinearRegression()
            }
            
            best_model = None
            best_score = float('inf')
            best_name = None
            
            for name, model in models.items():
                # Cross-validation (adjust cv based on sample size)
                cv_folds = min(5, len(X))
                if cv_folds < 2:
                    cv_folds = 2
                    
                cv_scores = cross_val_score(model, X, y, cv=cv_folds, 
                                          scoring='neg_mean_absolute_error')
                avg_error = -cv_scores.mean()
                
                print(f"    {name}: {avg_error:.1f}L average error")
                
                if avg_error < best_score:
                    best_score = avg_error
                    best_model = model
                    best_name = name
            
            # Train best model on full dataset
            best_model.fit(X, y)
            
            # Feature scaling for this book type
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Store model and scaler
            self.models[book_type] = best_model
            self.scalers[book_type] = scaler
            
            # Validation on training data
            y_pred = best_model.predict(X)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred) if len(y) > 1 else 0
            
            print(f"    ✅ Best: {best_name}")
            print(f"    Training MAE: {mae:.1f}L")
            print(f"    Training R²: {r2:.3f}")
            print(f"    CV MAE: {best_score:.1f}L")
        
        # Train a general fallback model for cases where book type is unknown
        print(f"\n🔄 Training general fallback model:")
        
        X_all = self.training_data[feature_cols]
        y_all = self.training_data['verified_lexile']
        
        general_model = RandomForestRegressor(n_estimators=100, random_state=42)
        general_model.fit(X_all, y_all)
        
        general_scaler = StandardScaler()
        X_all_scaled = general_scaler.fit_transform(X_all)
        
        self.models['General'] = general_model
        self.scalers['General'] = general_scaler
        
        mae_general = mean_absolute_error(y_all, general_model.predict(X_all))
        r2_general = r2_score(y_all, general_model.predict(X_all))
        
        print(f"  General model MAE: {mae_general:.1f}L")
        print(f"  General model R²: {r2_general:.3f}")
        
        return self.models
    
    def predict_with_book_type(self, features, book_type='Standard_Lexile'):
        """Make prediction using appropriate model for book type"""
        
        if book_type in self.models:
            model = self.models[book_type]
        else:
            print(f"  ⚠️  No model for {book_type}, using general model")
            model = self.models.get('General', list(self.models.values())[0])
        
        # Create feature array
        X = np.array([[features[col] for col in self.feature_names]])
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        return prediction
    
    def save_improved_models(self):
        """Save all models and scalers"""
        
        models_dir = DATA_DIR / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Save each model
        for book_type, model in self.models.items():
            model_path = models_dir / f"age_model_{book_type.lower().replace(' ', '_')}.joblib"
            joblib.dump(model, model_path)
            print(f"💾 Saved {book_type} model: {model_path}")
        
        # Save scalers
        for book_type, scaler in self.scalers.items():
            scaler_path = models_dir / f"scaler_{book_type.lower().replace(' ', '_')}.joblib"
            joblib.dump(scaler, scaler_path)
        
        # Save training data and feature names
        if self.training_data is not None:
            results_path = DATA_DIR / "improved_age_model_results.csv"
            self.training_data.to_csv(results_path, index=False)
            print(f"💾 Training results saved: {results_path}")
        
        # Save feature names
        feature_names_path = models_dir / "feature_names.joblib"
        joblib.dump(self.feature_names, feature_names_path)
        print(f"💾 Feature names saved: {feature_names_path}")
        
        return models_dir

def main():
    predictor = ImprovedAgeBasedPredictor()
    
    # Prepare improved training data
    training_data = predictor.prepare_improved_training_data()
    
    if training_data is not None and len(training_data) > 5:
        # Train improved models
        models = predictor.train_improved_model()
        
        # Save models
        models_dir = predictor.save_improved_models()
        
        print(f"\n🎉 IMPROVED MODEL TRAINING COMPLETE!")
        print(f"  ✅ Trained separate models for each book type")
        print(f"  ✅ Enhanced feature engineering")
        print(f"  ✅ Models saved to: {models_dir}")
        
        return predictor, training_data, models
    else:
        print("❌ Insufficient training data for improved models")
        return None, None, None

if __name__ == "__main__":
    main()