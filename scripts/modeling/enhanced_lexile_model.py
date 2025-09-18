import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"

def load_clear_features():
    """Load preprocessed CLEAR corpus features and targets"""
    X = np.load(PROCESSED_DIR / 'clear_features.npy')
    y = np.load(PROCESSED_DIR / 'clear_targets.npy')
    feature_names = pd.read_csv(PROCESSED_DIR / 'feature_names.csv')['feature'].tolist()
    
    print(f"Loaded CLEAR corpus: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features: {feature_names}")
    
    return X, y, feature_names

def calculate_confidence(prediction, feature_vector, model):
    """Calculate confidence score for a Lexile prediction"""
    # Base confidence on model type and feature quality
    base_confidence = 0.7
    
    # Adjust based on feature completeness
    missing_features = np.sum(np.isnan(feature_vector)) if np.any(np.isnan(feature_vector)) else 0
    completeness_penalty = missing_features * 0.1
    
    # Adjust based on prediction range (more confident for common ranges)
    if 300 <= prediction <= 1200:  # Common range
        range_bonus = 0.2
    elif 100 <= prediction <= 1500:  # Extended common range
        range_bonus = 0.1
    else:  # Extreme ranges
        range_bonus = -0.1
    
    confidence = base_confidence - completeness_penalty + range_bonus
    return max(0.1, min(1.0, confidence))  # Clamp between 0.1 and 1.0

def train_ensemble_model():
    """Train an enhanced Lexile estimation model using CLEAR corpus data"""
    
    print("🚀 TRAINING ENHANCED LEXILE ESTIMATION MODEL")
    print("═════════════════════════════════════════════")
    
    # Load CLEAR corpus data
    X, y, feature_names = load_clear_features()
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train multiple models and compare performance
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        ),
        'Ridge Regression': Ridge(alpha=1.0),
        'Linear Regression': LinearRegression()
    }
    
    best_model = None
    best_score = float('inf')
    best_name = ""
    model_results = {}
    
    print("\n📊 MODEL PERFORMANCE COMPARISON")
    print("═══════════════════════════════")
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                  scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Test set performance
        y_pred = model.predict(X_test)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_r2 = r2_score(y_test, y_pred)
        
        model_results[name] = {
            'model': model,
            'cv_mae': cv_mae,
            'cv_std': cv_std,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2
        }
        
        print(f"{name}:")
        print(f"  CV MAE: {cv_mae:.1f} ± {cv_std:.1f}")
        print(f"  Test MAE: {test_mae:.1f}")
        print(f"  Test RMSE: {test_rmse:.1f}")
        print(f"  Test R²: {test_r2:.3f}")
        print()
        
        # Track best model (lowest test MAE)
        if test_mae < best_score:
            best_score = test_mae
            best_model = model
            best_name = name
    
    print(f"🏆 Best model: {best_name} (MAE: {best_score:.1f})")
    
    # Feature importance for best model (if available)
    if hasattr(best_model, 'feature_importances_'):
        print(f"\n📈 FEATURE IMPORTANCE ({best_name})")
        print("═══════════════════════════════════")
        importances = best_model.feature_importances_
        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        for feature, importance in feature_importance:
            print(f"  {feature}: {importance:.3f}")
    
    
    # Save model and scaler
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save best model
    model_path = MODELS_DIR / 'lexile_estimator.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': best_model,
            'model_name': best_name,
            'feature_names': feature_names,
            'performance': model_results[best_name],
            'confidence_function': 'calculate_confidence'
        }, f)
    
    # Load scaler parameters for inference
    scaler_params = pd.read_csv(PROCESSED_DIR / 'scaler_params.csv')
    scaler = StandardScaler()
    scaler.mean_ = scaler_params['mean'].values
    scaler.scale_ = scaler_params['scale'].values
    
    # Save scaler
    scaler_path = MODELS_DIR / 'lexile_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\n💾 Model saved to: {model_path}")
    print(f"💾 Scaler saved to: {scaler_path}")
    
    # Test the saved model
    print(f"\n🧪 TESTING SAVED MODEL")
    print("════════════════════════")
    test_saved_model(X_test[:5], y_test[:5], feature_names)
    
    return best_model, feature_names, model_results

def test_saved_model(X_sample, y_true, feature_names):
    """Test the saved model with sample data"""
    
    # Load saved model
    with open(MODELS_DIR / 'lexile_estimator.pkl', 'rb') as f:
        saved_data = pickle.load(f)
    
    model = saved_data['model']
    
    print("Sample predictions:")
    for i in range(len(X_sample)):
        features = X_sample[i]
        prediction = model.predict([features])[0]
        confidence = calculate_confidence(prediction, features, model)
        actual = y_true[i]
        error = abs(prediction - actual)
        
        print(f"  Predicted: {prediction:.0f}L, Actual: {actual:.0f}L, "
              f"Error: {error:.0f}, Confidence: {confidence:.2f}")

def create_inference_function():
    """Create a function for making Lexile predictions on new text"""
    
    inference_code = '''
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import textstat
from textstat import flesch_kincaid_grade_level, flesch_reading_ease, automated_readability_index, smog_index, dale_chall_readability_score

def estimate_lexile_from_text(text, models_dir="/Users/chaerinnoh/Desktop/kidlit/models"):
    """
    Estimate Lexile score from raw text using the trained model
    
    Args:
        text (str): The text to analyze
        models_dir (str): Path to saved models directory
    
    Returns:
        dict: Contains lexile_score, confidence, and feature_breakdown
    """
    
    # Load saved model and scaler
    model_path = Path(models_dir) / 'lexile_estimator.pkl'
    scaler_path = Path(models_dir) / 'lexile_scaler.pkl'
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    model = model_data['model']
    feature_names = model_data['feature_names']
    confidence_fn = model_data['confidence_function']
    
    # Extract features from text (matching CLEAR corpus features)
    words = len(text.split())
    sentences = len([s for s in text.split('.') if s.strip()])
    paragraphs = len([p for p in text.split('\\n\\n') if p.strip()])
    
    features = [
        flesch_kincaid_grade_level(text),      # Flesch-Kincaid-Grade-Level
        flesch_reading_ease(text),             # Flesch-Reading-Ease  
        automated_readability_index(text),     # Automated Readability Index
        smog_index(text),                      # SMOG Readability
        dale_chall_readability_score(text),    # New Dale-Chall Readability Formula
        words,                                 # Google WC (word count)
        sentences,                             # Sentence Count
        paragraphs                             # Paragraphs
    ]
    
    # Handle any NaN values
    features = [f if not np.isnan(f) else 0 for f in features]
    features_array = np.array(features).reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features_array)
    
    # Make prediction
    lexile_score = model.predict(features_scaled)[0]
    confidence = confidence_fn(lexile_score, features_array[0], model)
    
    # Feature breakdown for debugging
    feature_breakdown = dict(zip(feature_names, features))
    
    return {
        'lexile_score': round(lexile_score),
        'confidence': round(confidence, 2),
        'feature_breakdown': feature_breakdown
    }

# Example usage:
# result = estimate_lexile_from_text("Your book text goes here...")
# print(f"Estimated Lexile: {result['lexile_score']}L (confidence: {result['confidence']})")
'''
    
    # Save inference function
    inference_path = MODELS_DIR / 'lexile_inference.py'
    with open(inference_path, 'w') as f:
        f.write(inference_code)
    
    print(f"📄 Inference function saved to: {inference_path}")

if __name__ == "__main__":
    # Ensure models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Train the enhanced model
    model, features, results = train_ensemble_model()
    
    # Create inference function
    create_inference_function()
    
    print(f"\n✅ ENHANCED LEXILE MODEL READY!")
    print(f"   - Trained on {len(load_clear_features()[0])} CLEAR corpus samples")
    print(f"   - Best model: {results[list(results.keys())[0]]['model'].__class__.__name__}")
    print(f"   - Inference function available in models/lexile_inference.py")