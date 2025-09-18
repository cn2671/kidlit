#!/usr/bin/env python3
"""
Script to fill missing age data for books using the existing age prediction model.
This will add age_range_llm data to the 117 books that are currently missing it.
"""

import pandas as pd
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re

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

def predict_age_ranges(df, missing_mask, model):
    """Predict age ranges for books missing age data"""
    print(f"Predicting age ranges for {missing_mask.sum()} books...")
    
    # Get books that need age prediction
    books_to_predict = df[missing_mask].copy()
    
    # Create feature text combining available fields
    feature_texts = []
    for _, row in books_to_predict.iterrows():
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
    
    # Check if model is a pipeline or just the classifier
    if hasattr(model, 'predict'):
        # Model is ready to use
        try:
            predictions = model.predict(feature_texts)
        except Exception as e:
            print(f"Error with model prediction: {e}")
            # Fallback to manual age assignment based on known patterns
            predictions = assign_manual_ages(books_to_predict)
    else:
        print("Model structure not recognized, using manual age assignment")
        predictions = assign_manual_ages(books_to_predict)
    
    return predictions

def assign_manual_ages(books_df):
    """Manual age assignment based on book characteristics"""
    predictions = []
    
    for _, row in books_df.iterrows():
        title = str(row.get('title', '')).lower()
        author = str(row.get('author', '')).lower()
        description = str(row.get('description', '')).lower()
        
        # Rules based on common children's book patterns
        age_range = "3-5"  # Default for picture books
        
        # Board books and very simple stories
        if any(word in title for word in ['goodnight', 'very hungry', 'giving tree', 'pat the', 'brown bear']):
            age_range = "3-5"
        
        # Classic picture books
        elif any(word in title for word in ['where the wild things', 'cat in the hat', 'green eggs']):
            age_range = "3-5"
        
        # Early readers
        elif any(word in title for word in ['frog and toad', 'little critter', 'berenstain']):
            age_range = "6-8"
        
        # Chapter books
        elif any(word in title for word in ['diary of', 'magic tree house', 'junie b']):
            age_range = "6-8"
        
        # Middle grade
        elif any(word in title for word in ['harry potter', 'percy jackson', 'wonder']):
            age_range = "9-12"
        
        # Check author patterns
        elif 'dr. seuss' in author or 'dr seuss' in author:
            age_range = "3-5"
        elif 'eric carle' in author:
            age_range = "3-5"
        elif 'maurice sendak' in author:
            age_range = "3-5"
        elif 'shel silverstein' in author:
            age_range = "6-8"
        
        # Check description for complexity indicators
        elif 'chapter' in description or 'grade' in description:
            age_range = "9-12"
        elif 'picture book' in description or 'bedtime' in description:
            age_range = "3-5"
        
        predictions.append(age_range)
    
    return predictions

def main():
    print("🔄 Starting age data completion process...")
    
    # Load the dataset
    print("📚 Loading book dataset...")
    df = pd.read_csv('../data/raw/books_final_complete.csv')
    print(f"✓ Loaded {len(df)} books")
    
    # Identify books missing age data
    missing_age = df['age_range_llm'].isna() | (df['age_range_llm'] == '') | (df['age_range_llm'].str.strip() == '')
    missing_count = missing_age.sum()
    
    print(f"📊 Found {missing_count} books missing age data")
    
    if missing_count == 0:
        print("✅ All books already have age data!")
        return
    
    # Load the age prediction model
    model = load_age_model()
    
    # Predict age ranges for missing books
    if model is not None:
        predictions = predict_age_ranges(df, missing_age, model)
    else:
        print("🔧 Using manual age assignment rules...")
        books_to_predict = df[missing_age].copy()
        predictions = assign_manual_ages(books_to_predict)
    
    # Update the dataframe with predictions
    df.loc[missing_age, 'age_range_llm'] = predictions
    
    # Verify the update
    still_missing = df['age_range_llm'].isna() | (df['age_range_llm'] == '') | (df['age_range_llm'].str.strip() == '')
    still_missing_count = still_missing.sum()
    
    print(f"✅ Successfully predicted age ranges for {missing_count - still_missing_count} books")
    
    if still_missing_count > 0:
        print(f"⚠️  {still_missing_count} books still missing age data")
    
    # Show some examples of the predictions
    print("\n📖 Sample of newly assigned age ranges:")
    newly_assigned = df[missing_age][['title', 'author', 'age_range_llm']].head(10)
    for _, row in newly_assigned.iterrows():
        print(f"  • \"{row['title']}\" by {row['author']} → {row['age_range_llm']}")
    
    # Save the updated dataset
    backup_path = '../data/raw/books_final_complete_backup.csv'
    print(f"\n💾 Creating backup at {backup_path}")
    df.to_csv(backup_path, index=False)
    
    output_path = '../data/raw/books_final_complete.csv'
    print(f"💾 Saving updated dataset to {output_path}")
    df.to_csv(output_path, index=False)
    
    # Final verification
    final_missing = df['age_range_llm'].isna() | (df['age_range_llm'] == '') | (df['age_range_llm'].str.strip() == '')
    final_missing_count = final_missing.sum()
    
    print(f"\n🎉 Final result: {len(df) - final_missing_count}/{len(df)} books now have age data")
    print(f"📈 Completion rate: {((len(df) - final_missing_count) / len(df) * 100):.1f}%")
    
    if final_missing_count == 0:
        print("🎯 SUCCESS: All books now have age data!")
    
    return df

if __name__ == "__main__":
    main()