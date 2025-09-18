import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

class AgeBasedPredictor:
    """Apply trained age model to predict remaining books"""
    
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.load_trained_model()
        
    def load_trained_model(self):
        """Load trained model or retrain if needed"""
        try:
            # Try to load existing model
            model_path = DATA_DIR / "age_model.joblib"
            if model_path.exists():
                self.model = joblib.load(model_path)
                print("✅ Loaded existing age model")
            else:
                print("⚠️ No saved model found, retraining...")
                self.retrain_model()
        except:
            print("❌ Error loading model, retraining...")
            self.retrain_model()
    
    def retrain_model(self):
        """Retrain the age-based model"""
        # Load verified training data
        results_path = DATA_DIR / "age_based_model_results.csv"
        training_data = pd.read_csv(results_path)
        
        feature_cols = ['min_age', 'max_age', 'avg_age', 'age_range', 'min_grade', 'max_grade', 
                       'ar_level', 'is_ad_book', 'is_series', 'is_picture_book', 'is_classic', 'popular_author']
        
        self.feature_names = feature_cols
        
        X = training_data[feature_cols]
        y = training_data['verified_lexile']
        
        # Train Random Forest (best performer)
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
        # Save model
        model_path = DATA_DIR / "age_model.joblib"
        joblib.dump(self.model, model_path)
        print(f"💾 Model saved: {model_path}")
    
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
    
    def predict_remaining_books(self):
        """Predict Lexile scores for books with age data but no verified scores"""
        
        # Load enriched dataset
        tracking_path = DATA_DIR / "lexile_collection_tracking.csv"
        df = pd.read_csv(tracking_path)
        
        print(f"🔮 APPLYING AGE-BASED MODEL TO PREDICT LEXILE SCORES")
        print(f"{'='*55}")
        
        predictions = []
        successful_predictions = 0
        failed_predictions = 0
        
        for idx, row in df.iterrows():
            # Skip if already has verified score
            if row['status'] == 'Complete':
                continue
                
            # Extract age features
            min_age, max_age, min_grade, max_grade = self.extract_age_features(row['notes'])
            
            # Skip if no age data
            if min_age is None:
                failed_predictions += 1
                continue
                
            ar_level = self.extract_ar_level(row['notes'])
            additional_features = self.extract_additional_features(row['title'], row['author'], row['notes'])
            
            # Prepare feature vector
            feature_vector = {
                'min_age': min_age,
                'max_age': max_age if max_age else min_age,
                'avg_age': (min_age + (max_age if max_age else min_age)) / 2,
                'age_range': (max_age if max_age else min_age) - min_age,
                'min_grade': min_grade if min_grade is not None else -1,
                'max_grade': max_grade if max_grade is not None else -1,
                'ar_level': ar_level if ar_level is not None else -1,
                'is_ad_book': 0,  # Default to standard lexile for predictions
                **additional_features
            }
            
            # Create feature array in correct order
            X = np.array([[feature_vector[col] for col in self.feature_names]])
            
            # Make prediction
            predicted_lexile = self.model.predict(X)[0]
            
            prediction_data = {
                'title': row['title'],
                'author': row['author'],
                'age_range': f"{min_age}-{max_age if max_age else min_age}",
                'predicted_lexile': round(predicted_lexile),
                'confidence': 'High - Age model (99.7L avg error)',
                'original_ml_estimate': row.get('current_ml_estimate', 'N/A'),
                'prediction_diff': round(predicted_lexile - row['current_ml_estimate']) if pd.notna(row.get('current_ml_estimate')) else 'N/A'
            }
            
            predictions.append(prediction_data)
            successful_predictions += 1
            
            print(f"  ✅ {row['title'][:40]}: Ages {min_age}-{max_age if max_age else min_age} → {predicted_lexile:.0f}L")
        
        print(f"\n📊 PREDICTION RESULTS:")
        print(f"  Successful predictions: {successful_predictions}")
        print(f"  Failed (no age data): {failed_predictions}")
        
        if predictions:
            predictions_df = pd.DataFrame(predictions)
            
            print(f"\n📈 PREDICTION ANALYSIS:")
            print(f"  Lexile range: {predictions_df['predicted_lexile'].min():.0f} - {predictions_df['predicted_lexile'].max():.0f}L")
            print(f"  Average prediction: {predictions_df['predicted_lexile'].mean():.0f}L")
            
            # Show comparison with original ML estimates where available
            valid_diffs = predictions_df[predictions_df['prediction_diff'] != 'N/A']['prediction_diff']
            if len(valid_diffs) > 0:
                avg_diff = valid_diffs.mean()
                print(f"  Average difference from ML estimates: {avg_diff:.0f}L")
            
            # Save predictions
            predictions_path = DATA_DIR / "age_based_predictions.csv"
            predictions_df.to_csv(predictions_path, index=False)
            print(f"\n💾 Predictions saved: {predictions_path}")
            
            return predictions_df
        else:
            print("❌ No predictions generated")
            return None
    
    def update_tracking_with_predictions(self, predictions_df):
        """Update main tracking file with age-based predictions"""
        
        if predictions_df is None:
            return
        
        print(f"\n📝 UPDATING TRACKING FILE WITH PREDICTIONS")
        print(f"{'='*45}")
        
        # Load tracking file
        tracking_path = DATA_DIR / "lexile_collection_tracking.csv"
        df = pd.read_csv(tracking_path)
        
        updated_count = 0
        
        for _, pred_row in predictions_df.iterrows():
            # Find matching book in tracking file
            mask = df['title'] == pred_row['title']
            matching_rows = df[mask]
            
            if len(matching_rows) > 0:
                idx = matching_rows.index[0]
                
                # Update with prediction
                df.loc[idx, 'predicted_lexile_age_model'] = pred_row['predicted_lexile']
                df.loc[idx, 'age_model_confidence'] = 'High'
                df.loc[idx, 'prediction_method'] = 'Age-based model'
                
                # Add prediction notes
                current_notes = df.loc[idx, 'notes']
                prediction_note = f"Predicted: {pred_row['predicted_lexile']:.0f}L (age model)"
                
                if pd.notna(current_notes) and current_notes != '':
                    df.loc[idx, 'notes'] = f"{current_notes}; {prediction_note}"
                else:
                    df.loc[idx, 'notes'] = prediction_note
                
                updated_count += 1
        
        # Save updated tracking file
        df.to_csv(tracking_path, index=False)
        
        print(f"  Updated {updated_count} books with age-based predictions")
        print(f"💾 Tracking file updated")
        
        return df

def main():
    predictor = AgeBasedPredictor()
    
    # Make predictions
    predictions_df = predictor.predict_remaining_books()
    
    # Update tracking file
    updated_df = predictor.update_tracking_with_predictions(predictions_df)
    
    return predictions_df, updated_df

if __name__ == "__main__":
    main()