#!/usr/bin/env python3
"""
Script to evaluate the accuracy of the age prediction model.
This will test the model against books that already have age data to see how well it performs.
"""

import pandas as pd
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import matplotlib.pyplot as plt
import seaborn as sns

def clean_text(text):
    """Clean text for feature extraction"""
    if pd.isna(text):
        return ""
    # Convert to string and lowercase
    text = str(text).lower()
    # Remove special characters and extra spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_age_model():
    """Load the existing age prediction model"""
    try:
        model = joblib.load('../data/age_model.joblib')
        print("✓ Age prediction model loaded successfully")
        return model
    except Exception as e:
        print(f"✗ Error loading age model: {e}")
        return None

def prepare_features(df):
    """Prepare feature text for model evaluation"""
    feature_texts = []
    for _, row in df.iterrows():
        # Combine title, author, description, themes, and other available text
        text_parts = []
        
        if pd.notna(row.get('title')):
            text_parts.append(clean_text(row['title']))
        if pd.notna(row.get('author')):
            text_parts.append(clean_text(row['author']))
        if pd.notna(row.get('description')):
            text_parts.append(clean_text(row['description']))
        if pd.notna(row.get('themes')):
            text_parts.append(clean_text(row['themes']))
        if pd.notna(row.get('tone')):
            text_parts.append(clean_text(row['tone']))
        if pd.notna(row.get('genre')):
            text_parts.append(clean_text(row['genre']))
            
        feature_text = ' '.join(text_parts)
        feature_texts.append(feature_text)
    
    return feature_texts

def normalize_age_ranges(age_ranges):
    """Normalize age ranges to consistent format"""
    normalized = []
    for age in age_ranges:
        age_str = str(age).strip()
        
        # Map common variations to standard format
        if age_str in ['3-5', '3-5 years']:
            normalized.append('3-5')
        elif age_str in ['6-8', '6-8 years']:
            normalized.append('6-8')
        elif age_str in ['9-12', '9-12 years']:
            normalized.append('9-12')
        elif age_str in ['13+', '13+ years', 'teen', 'young adult']:
            normalized.append('13+')
        elif age_str in ['4-8', '4-8 years']:
            normalized.append('4-8')
        elif age_str in ['3-6', '3-6 years']:
            normalized.append('3-6')
        elif age_str in ['3-8', '3-8 years']:
            normalized.append('3-8')
        elif age_str in ['8-12', '8-12 years']:
            normalized.append('8-12')
        elif age_str in ['1-3', '1-3 years', 'toddler']:
            normalized.append('1-3')
        elif age_str in ['0-3', '0-3 years', 'baby', 'infant']:
            normalized.append('0-3')
        else:
            normalized.append(age_str)
    
    return normalized

def evaluate_model():
    print("🔍 Starting age prediction model evaluation...")
    
    # Load the dataset with the backup to get original age data
    print("📚 Loading original dataset...")
    df_backup = pd.read_csv('../data/raw/books_final_complete_backup.csv')
    
    # Filter to books that originally had age data
    books_with_original_age = df_backup[
        df_backup['age_range_llm'].notna() & 
        (df_backup['age_range_llm'] != '') & 
        (df_backup['age_range_llm'].str.strip() != '')
    ].copy()
    
    print(f"✓ Found {len(books_with_original_age)} books with original age data")
    
    if len(books_with_original_age) < 50:
        print("⚠️  Not enough books with original age data for meaningful evaluation")
        return
    
    # Load the age prediction model
    model = load_age_model()
    if model is None:
        print("❌ Cannot evaluate without model")
        return
    
    # Prepare features
    print("🔧 Preparing features...")
    feature_texts = prepare_features(books_with_original_age)
    
    # Get actual age ranges
    actual_ages = books_with_original_age['age_range_llm'].tolist()
    
    # Make predictions
    print("🎯 Making predictions...")
    try:
        if hasattr(model, 'predict'):
            predictions = model.predict(feature_texts)
        else:
            print("Model structure not recognized")
            return
    except Exception as e:
        print(f"❌ Error making predictions: {e}")
        return
    
    # Normalize age ranges for comparison
    actual_normalized = normalize_age_ranges(actual_ages)
    pred_normalized = normalize_age_ranges(predictions)
    
    # Calculate accuracy
    accuracy = accuracy_score(actual_normalized, pred_normalized)
    print(f"\n📊 Model Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Show detailed classification report
    print("\n📋 Detailed Classification Report:")
    print(classification_report(actual_normalized, pred_normalized))
    
    # Show confusion matrix
    print("\n🔀 Confusion Matrix:")
    unique_labels = sorted(list(set(actual_normalized + pred_normalized)))
    cm = confusion_matrix(actual_normalized, pred_normalized, labels=unique_labels)
    
    # Create a readable confusion matrix
    cm_df = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)
    print(cm_df)
    
    # Show some specific examples
    print("\n📖 Sample Predictions vs Actual:")
    sample_size = min(20, len(books_with_original_age))
    sample_indices = np.random.choice(len(books_with_original_age), sample_size, replace=False)
    
    correct_predictions = 0
    total_predictions = 0
    
    for i in sample_indices:
        title = books_with_original_age.iloc[i]['title']
        author = books_with_original_age.iloc[i]['author']
        actual = actual_normalized[i]
        predicted = pred_normalized[i]
        match = "✓" if actual == predicted else "✗"
        
        if actual == predicted:
            correct_predictions += 1
        total_predictions += 1
        
        print(f"  {match} \"{title}\" by {author}")
        print(f"    Actual: {actual}, Predicted: {predicted}")
    
    sample_accuracy = correct_predictions / total_predictions
    print(f"\n📊 Sample Accuracy: {sample_accuracy:.3f} ({sample_accuracy*100:.1f}%)")
    
    # Analyze by age group
    print("\n📈 Performance by Age Group:")
    age_group_performance = {}
    for actual, predicted in zip(actual_normalized, pred_normalized):
        if actual not in age_group_performance:
            age_group_performance[actual] = {'correct': 0, 'total': 0}
        age_group_performance[actual]['total'] += 1
        if actual == predicted:
            age_group_performance[actual]['correct'] += 1
    
    for age_group in sorted(age_group_performance.keys()):
        stats = age_group_performance[age_group]
        group_accuracy = stats['correct'] / stats['total']
        print(f"  {age_group}: {group_accuracy:.3f} ({group_accuracy*100:.1f}%) - {stats['correct']}/{stats['total']} correct")
    
    # Identify common misclassifications
    print("\n🔍 Common Misclassifications:")
    misclassifications = {}
    for actual, predicted in zip(actual_normalized, pred_normalized):
        if actual != predicted:
            key = f"{actual} → {predicted}"
            misclassifications[key] = misclassifications.get(key, 0) + 1
    
    # Sort by frequency
    for misclass, count in sorted(misclassifications.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {misclass}: {count} times")
    
    # Overall assessment
    print(f"\n🎯 EVALUATION SUMMARY:")
    print(f"Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    if accuracy >= 0.8:
        print("🎉 EXCELLENT: Model performs very well!")
    elif accuracy >= 0.6:
        print("👍 GOOD: Model performs reasonably well")
    elif accuracy >= 0.4:
        print("⚠️  FAIR: Model has moderate accuracy, could use improvement")
    else:
        print("❌ POOR: Model accuracy is low, needs significant improvement")

if __name__ == "__main__":
    evaluate_model()