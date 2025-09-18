#!/usr/bin/env python3
"""
Test script for reading level predictions on new books
Usage: python3 scripts/test_predictions.py
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import json

class ReadingLevelTester:
    """Test the trained reading level prediction system"""
    
    def __init__(self, model_timestamp="20250831_182131"):
        self.model_timestamp = model_timestamp
        self.load_trained_models()
        
    def load_trained_models(self):
        """Load the trained models and metadata"""
        model_dir = "data/models/"
        
        try:
            # Load ensemble model (this contains the best lexile and category models)
            ensemble_path = f"{model_dir}ensemble_{self.model_timestamp}.joblib"
            self.ensemble = joblib.load(ensemble_path)
            print(f"✓ Loaded ensemble model from {ensemble_path}")
            
            # Load metadata (contains feature columns and label encoders)
            metadata_path = f"{model_dir}model_metadata_{self.model_timestamp}.joblib"
            self.metadata = joblib.load(metadata_path)
            self.feature_columns = self.metadata['feature_columns']
            self.label_encoders = self.metadata['label_encoders']
            print(f"✓ Loaded metadata from {metadata_path}")
            
            print(f"✓ Models loaded successfully!")
            print(f"  - Lexile model: {self.ensemble['lexile_model_name']}")
            print(f"  - Category model: {self.ensemble['category_model_name']}")
            
        except FileNotFoundError as e:
            print(f"❌ Error loading models: {e}")
            print("Make sure you've run the training script first!")
            raise
    
    def create_test_books(self):
        """Create sample test books to demonstrate predictions"""
        test_books = [
            {
                'title': 'The Very Hungry Caterpillar',
                'author': 'Eric Carle',
                'author_clean': 'Eric Carle',
                'themes': 'nature, growth, transformation',
                'description': 'A classic picture book about a caterpillar who eats his way through various foods before becoming a beautiful butterfly.',
                'reading_confidence_llm': 0.9,
                'lexile_confidence': 0.8
            },
            {
                'title': 'Harry Potter and the Sorcerer\'s Stone',
                'author': 'J.K. Rowling',
                'author_clean': 'J.K. Rowling',
                'themes': 'magic, friendship, adventure, courage',
                'description': 'Harry Potter learns he is a wizard on his 11th birthday and begins his magical education at Hogwarts School of Witchcraft and Wizardry.',
                'reading_confidence_llm': 0.7,
                'lexile_confidence': 0.7
            },
            {
                'title': 'Green Eggs and Ham',
                'author': 'Dr. Seuss',
                'author_clean': 'Dr. Seuss',
                'themes': 'humor, friendship, trying new things',
                'description': 'Sam-I-Am tries to convince another character to try green eggs and ham.',
                'reading_confidence_llm': 0.95,
                'lexile_confidence': 0.9
            },
            {
                'title': 'Wonder',
                'author': 'R.J. Palacio',
                'author_clean': 'R.J. Palacio',
                'themes': 'acceptance, identity, courage, friendship',
                'description': 'A story about August Pullman, a boy with facial differences who enters mainstream school for the first time.',
                'reading_confidence_llm': 0.8,
                'lexile_confidence': 0.75
            },
            {
                'title': 'Unknown New Book',
                'author': 'New Author',
                'author_clean': 'New Author',
                'themes': 'adventure, mystery',
                'description': 'A short description of a mysterious new adventure.',
                'reading_confidence_llm': 0.5,
                'lexile_confidence': 0.4
            }
        ]
        
        return pd.DataFrame(test_books)
    
    def engineer_features_for_prediction(self, books_df):
        """Create the same features used in training"""
        print("Engineering features for prediction...")
        
        # Initialize feature DataFrame
        features_df = pd.DataFrame(index=books_df.index)
        
        # Calculate author statistics from training data (simplified - in production you'd use the full author stats)
        author_stats = {
            'Dr. Seuss': {'book_count': 32, 'avg_lexile': 263, 'std_lexile': 34, 'primary_age_category': 'Early'},
            'Eric Carle': {'book_count': 5, 'avg_lexile': 200, 'std_lexile': 25, 'primary_age_category': 'Early'},
            'J.K. Rowling': {'book_count': 7, 'avg_lexile': 880, 'std_lexile': 50, 'primary_age_category': 'Advanced'},
            'R.J. Palacio': {'book_count': 3, 'avg_lexile': 720, 'std_lexile': 30, 'primary_age_category': 'Intermediate'}
        }
        
        for idx, book in books_df.iterrows():
            book_features = self._create_book_features(book, author_stats)
            for feature, value in book_features.items():
                features_df.loc[idx, feature] = value
        
        # Ensure all required columns are present with default values
        for col in self.feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0
        
        # Select only the features used in training and in the right order
        features_df = features_df[self.feature_columns]
        
        # Fill any remaining NaN values
        features_df = features_df.fillna(0)
        
        print(f"✓ Created {len(features_df)} feature vectors with {len(self.feature_columns)} features each")
        return features_df
    
    def _create_book_features(self, book, author_stats):
        """Create features for a single book"""
        features = {}
        
        # Author features
        author = book.get('author_clean', '')
        if author in author_stats:
            stats = author_stats[author]
            features['author_book_count'] = stats['book_count']
            features['author_avg_lexile'] = stats['avg_lexile']
            features['author_lexile_consistency'] = stats['std_lexile']
            features['is_prolific_author'] = 1 if stats['book_count'] >= 5 else 0
            features['is_specialist_author'] = 1 if stats['std_lexile'] < 50 else 0
            
            # Encode author primary category (simplified mapping)
            category_encoding = {'Early': 0, 'Beginning': 1, 'Intermediate': 2, 'Advanced': 3, 'Unknown': 4}
            features['author_primary_category_encoded'] = category_encoding.get(stats['primary_age_category'], 4)
        else:
            # Unknown author defaults
            features['author_book_count'] = 1
            features['author_avg_lexile'] = 0  # Will be filled as 0
            features['author_lexile_consistency'] = 0
            features['is_prolific_author'] = 0
            features['is_specialist_author'] = 0
            features['author_primary_category_encoded'] = 4
        
        # Text features
        title = book.get('title', '')
        features['title_length'] = len(title)
        features['title_word_count'] = len(title.split()) if title else 0
        features['title_has_numbers'] = 1 if any(c.isdigit() for c in title) else 0
        
        description = book.get('description', '')
        features['description_length'] = len(description)
        features['description_word_count'] = len(description.split()) if description else 0
        features['has_substantial_description'] = 1 if len(description) > 100 else 0
        
        # Theme features
        themes_str = book.get('themes', '')
        themes = [t.strip().lower() for t in themes_str.split(',') if t.strip()] if themes_str else []
        features['theme_count'] = len(themes)
        
        # Specific theme indicators
        for theme in ['friendship', 'adventure', 'family', 'imagination', 'magic', 'humor']:
            features[f'has_{theme}_theme'] = 1 if theme in themes else 0
        
        # Complex themes
        complex_themes = ['identity', 'courage', 'loyalty', 'acceptance']
        features['has_complex_themes'] = 1 if any(t in complex_themes for t in themes) else 0
        
        # Confidence features
        features['reading_confidence_llm'] = book.get('reading_confidence_llm', 0.5)
        features['lexile_confidence'] = book.get('lexile_confidence', 0.5)
        features['avg_confidence'] = (features['reading_confidence_llm'] + features['lexile_confidence']) / 2
        features['high_confidence'] = 1 if features['avg_confidence'] >= 0.8 else 0
        
        # Data quality features
        features['data_completeness_score'] = (
            (1 if len(description) > 50 else 0) +
            (1 if themes_str else 0) +
            (1 if book.get('summary_gpt', '') else 0)
        ) / 3
        
        return features
    
    def predict_reading_levels(self, books_df, use_tiered_system=True):
        """Make predictions for the books"""
        print(f"\nMaking predictions for {len(books_df)} books...")
        
        # Engineer features
        X = self.engineer_features_for_prediction(books_df)
        
        results = []
        
        for idx in range(len(X)):
            sample = X.iloc[idx:idx+1]
            book_title = books_df.iloc[idx]['title']
            
            # Get confidence score
            confidence = sample['avg_confidence'].iloc[0]
            
            # Tiered assignment logic
            if use_tiered_system:
                if confidence >= 0.8:
                    tier = "Tier 1: High Confidence"
                elif confidence >= 0.6:
                    tier = "Tier 2: Medium Confidence"
                else:
                    tier = "Tier 3: Conservative Estimate"
            else:
                tier = "Direct Prediction"
            
            # Make predictions
            if tier == "Tier 3: Conservative Estimate":
                lexile_pred = 400  # Conservative middle value
                category_pred_encoded = 1  # Beginning readers
            else:
                lexile_pred = self.ensemble['lexile_model'].predict(sample)[0]
                category_pred_encoded = self.ensemble['category_model'].predict(sample)[0]
            
            # Convert category prediction back to label
            category_label = self.label_encoders['category'].classes_[category_pred_encoded]
            
            # Map to age ranges
            age_range_mapping = {
                'Early': '3-5', 'Beginning': '6-8', 
                'Intermediate': '9-12', 'Advanced': '13+'
            }
            age_range = age_range_mapping.get(category_label, '6-8')
            
            result = {
                'book_title': book_title,
                'lexile_score': round(lexile_pred),
                'age_category': category_label,
                'age_range': age_range,
                'confidence_score': round(confidence, 3),
                'assignment_tier': tier
            }
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def run_test(self):
        """Run the complete test"""
        print("=== TESTING READING LEVEL PREDICTIONS ===\n")
        
        # Create test books
        test_books = self.create_test_books()
        print(f"Created {len(test_books)} test books:")
        for idx, book in test_books.iterrows():
            print(f"  {idx+1}. {book['title']} by {book['author']}")
        
        # Make predictions
        predictions = self.predict_reading_levels(test_books)
        
        # Display results
        print(f"\n=== PREDICTION RESULTS ===")
        for idx, result in predictions.iterrows():
            print(f"\n{idx+1}. {result['book_title']}")
            print(f"   📊 Lexile Score: {result['lexile_score']}")
            print(f"   🎯 Age Range: {result['age_range']} years")
            print(f"   📚 Category: {result['age_category']} Readers")
            print(f"   🔍 Confidence: {result['confidence_score']}")
            print(f"   ⚡ Tier: {result['assignment_tier']}")
        
        # Summary statistics
        print(f"\n=== SUMMARY STATISTICS ===")
        print(f"Average Lexile Score: {predictions['lexile_score'].mean():.0f}")
        print(f"Lexile Range: {predictions['lexile_score'].min()} - {predictions['lexile_score'].max()}")
        print("\nCategory Distribution:")
        print(predictions['age_category'].value_counts())
        print("\nTier Distribution:")
        print(predictions['assignment_tier'].value_counts())
        
        # Save results
        output_file = f"data/test_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        combined_results = pd.concat([test_books, predictions.drop('book_title', axis=1)], axis=1)
        combined_results.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to {output_file}")
        
        return predictions

def test_with_custom_books():
    """Example of testing with your own books"""
    tester = ReadingLevelTester()
    
    # Add your own books here!
    custom_books = pd.DataFrame([
        {
            'title': 'Your Book Title Here',
            'author': 'Author Name',
            'author_clean': 'Author Name',
            'themes': 'theme1, theme2, theme3',
            'description': 'Your book description here...',
            'reading_confidence_llm': 0.8,
            'lexile_confidence': 0.7
        }
    ])
    
    predictions = tester.predict_reading_levels(custom_books)
    print("Custom book predictions:")
    print(predictions)
    
    return predictions

if __name__ == "__main__":
    # Run the test
    tester = ReadingLevelTester()
    results = tester.run_test()
    
    print("\n" + "="*50)
    print("🎉 Testing complete! Your models are working!")
    print("="*50)
    
    # Uncomment to test custom books:
    # custom_results = test_with_custom_books()