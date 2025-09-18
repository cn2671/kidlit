
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
    paragraphs = len([p for p in text.split('\n\n') if p.strip()])
    
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
