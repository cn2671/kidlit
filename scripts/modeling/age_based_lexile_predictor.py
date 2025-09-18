import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

class AgeBasedLexilePredictor:
    """Build Lexile predictions using age and metadata features"""
    
    def __init__(self):
        self.model = None
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
        ar_match = re.search(r'ar\s+(?:level\s+)?(\d+\.?\d*)', notes_lower)
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
        
        # Series book indicator
        features['is_series'] = 1 if any(indicator in title_lower for indicator in 
                                       ['#', 'book', 'volume', 'series']) else 0
        
        # Picture book indicators
        features['is_picture_book'] = 1 if any(indicator in notes_lower for indicator in 
                                             ['picture book', 'pages', 'illustrations']) else 0
        
        # Classic/award book
        features['is_classic'] = 1 if any(award in notes_lower for award in 
                                        ['newbery', 'caldecott', 'award', 'medal', 'classic']) else 0
        
        # Author popularity (rough proxy)
        popular_authors = ['dr. seuss', 'roald dahl', 'eric carle', 'maurice sendak']
        features['popular_author'] = 1 if any(auth.lower() in str(author).lower() 
                                             for auth in popular_authors) else 0
        
        return features
    
    def prepare_training_data(self):
        """Prepare training data from verified books"""
        
        tracking_path = DATA_DIR / "lexile_collection_tracking.csv"
        df = pd.read_csv(tracking_path)
        
        # Get completed books with verified scores
        completed = df[df['status'] == 'Complete'].copy()
        
        print(f"📊 PREPARING AGE-BASED TRAINING DATA")
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
            
            # Prepare feature vector
            features = {
                'title': row['title'],
                'verified_lexile': row['lexile_numeric'],
                'book_type': row['book_type'],
                'min_age': min_age,
                'max_age': max_age if max_age else min_age,
                'avg_age': (min_age + (max_age if max_age else min_age)) / 2,
                'age_range': (max_age if max_age else min_age) - min_age,
                'min_grade': min_grade if min_grade is not None else -1,  # Use -1 for missing
                'max_grade': max_grade if max_grade is not None else -1,
                'ar_level': ar_level if ar_level is not None else -1,  # Use -1 for missing
                'is_ad_book': 1 if row['book_type'] == 'Adult_Directed' else 0,
                **additional_features
            }
            
            training_rows.append(features)
            
            print(f"  ✅ {row['title'][:30]}: Age {min_age}-{max_age}, Lexile {row['lexile_numeric']}L")
        
        self.training_data = pd.DataFrame(training_rows)
        
        print(f"\n📈 TRAINING DATA SUMMARY:")
        print(f"  Books with age data: {len(self.training_data)}")
        print(f"  Age range: {self.training_data['min_age'].min()}-{self.training_data['max_age'].max()} years")
        print(f"  Lexile range: {self.training_data['verified_lexile'].min():.0f}-{self.training_data['verified_lexile'].max():.0f}L")
        
        return self.training_data
    
    def train_age_model(self):
        """Train age-based Lexile prediction model"""
        
        if self.training_data is None:
            print("❌ No training data prepared!")
            return None
        
        print(f"\n🤖 TRAINING AGE-BASED LEXILE PREDICTION MODEL")
        print(f"{'='*50}")
        
        # Define feature columns
        feature_cols = ['min_age', 'max_age', 'avg_age', 'age_range', 'min_grade', 'max_grade', 
                       'ar_level', 'is_ad_book', 'is_series', 'is_picture_book', 'is_classic', 'popular_author']
        
        self.feature_names = feature_cols
        
        X = self.training_data[feature_cols]
        y = self.training_data['verified_lexile']
        
        print(f"Features: {len(feature_cols)}")
        print(f"Training samples: {len(X)}")
        
        # Try different models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        best_model = None
        best_score = float('inf')
        best_name = None
        
        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=min(5, len(X)), 
                                      scoring='neg_mean_absolute_error')
            avg_error = -cv_scores.mean()
            
            print(f"  {name}: {avg_error:.1f}L average error")
            
            if avg_error < best_score:
                best_score = avg_error
                best_model = model
                best_name = name
        
        # Train best model on full dataset
        self.model = best_model
        self.model.fit(X, y)
        
        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance = list(zip(feature_cols, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n🎯 FEATURE IMPORTANCE ({best_name}):")
            for feature, importance in feature_importance[:5]:
                print(f"  {feature}: {importance:.3f}")
        
        # Validation on training data
        y_pred = self.model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        print(f"\n📊 MODEL PERFORMANCE:")
        print(f"  Best model: {best_name}")
        print(f"  Training MAE: {mae:.1f}L")
        print(f"  Training R²: {r2:.3f}")
        print(f"  Cross-validation MAE: {best_score:.1f}L")
        
        # Show predictions vs actual
        print(f"\n🎯 PREDICTIONS vs ACTUAL:")
        self.training_data['predicted_lexile'] = y_pred
        self.training_data['prediction_error'] = abs(y_pred - y)
        
        for _, row in self.training_data.head(10).iterrows():
            title = row['title'][:30]
            actual = row['verified_lexile']
            predicted = row['predicted_lexile']
            error = row['prediction_error']
            print(f"  {title}: {actual:.0f}L → {predicted:.0f}L (error: {error:.0f}L)")
        
        return self.model
    
    def predict_remaining_books(self):
        """Use trained model to predict Lexile scores for remaining books"""
        
        if self.model is None:
            print("❌ No trained model available!")
            return None
        
        print(f"\n🔮 PREDICTING LEXILE SCORES FOR REMAINING BOOKS")
        print(f"{'='*50}")
        
        # Load full dataset
        tracking_path = DATA_DIR / "lexile_collection_tracking.csv"
        df = pd.read_csv(tracking_path)
        
        # Get books without verified scores
        unverified = df[df['status'] != 'Complete'].copy()
        
        print(f"Books to predict: {len(unverified)}")
        
        predictions = []
        
        for _, row in unverified.iterrows():
            # Try to extract features (would need age data in original dataset)
            # For now, we'll use the ML estimates as a baseline comparison
            
            prediction_data = {
                'title': row['title'],
                'author': row['author'],
                'current_ml_estimate': row['current_ml_estimate'],
                'age_based_prediction': None,  # Would need age data
                'confidence': 'Low - No age data'
            }
            
            predictions.append(prediction_data)
        
        predictions_df = pd.DataFrame(predictions)
        
        print(f"\n⚠️  Need age/grade data for remaining books to make predictions")
        print(f"Current ML estimates available as baseline")
        
        return predictions_df
    
    def save_model_results(self):
        """Save model and results"""
        
        if self.training_data is not None:
            # Save training results
            results_path = DATA_DIR / "age_based_model_results.csv"
            self.training_data.to_csv(results_path, index=False)
            print(f"💾 Model results saved: {results_path}")
            
        return results_path

def main():
    predictor = AgeBasedLexilePredictor()
    
    # Prepare training data
    training_data = predictor.prepare_training_data()
    
    if training_data is not None and len(training_data) > 3:
        # Train model
        model = predictor.train_age_model()
        
        # Save results
        predictor.save_model_results()
        
        # Try to predict remaining books
        predictions = predictor.predict_remaining_books()
        
        return predictor, training_data, predictions
    else:
        print("❌ Insufficient training data for age-based model")
        return None, None, None

if __name__ == "__main__":
    main()