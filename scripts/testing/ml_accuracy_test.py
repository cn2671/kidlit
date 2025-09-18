#!/usr/bin/env python3
"""
ML Accuracy Test - Compare ML predictions against known Lexile scores
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.core.enriched_predictor import EnrichedLexilePredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ml_accuracy():
    """Test ML accuracy against books with known Lexile scores"""
    print("🎯 Testing ML Accuracy Against Known Lexile Scores")
    print("=" * 60)
    
    predictor = EnrichedLexilePredictor()
    
    # Load catalog
    catalog_path = ROOT / "data" / "raw" / "books_final_complete.csv"
    catalog = pd.read_csv(catalog_path)
    
    # Find books with known Lexile scores that are NOT in enriched database
    books_with_lexile = catalog[catalog['lexile_score'].notna()].copy()
    print(f"📚 Found {len(books_with_lexile)} books with known Lexile scores")
    
    # Filter out books that are in enriched database
    non_enriched_with_lexile = []
    for _, row in books_with_lexile.iterrows():
        book_key = predictor._normalize_book_key(row['title'], row.get('author', ''))
        if book_key not in predictor.enriched_scores:
            non_enriched_with_lexile.append({
                'title': row['title'],
                'author': row.get('author', ''),
                'actual_lexile': row['lexile_score'],
                'age_min': row.get('age_min'),
                'age_max': row.get('age_max')
            })
    
    print(f"🔍 Found {len(non_enriched_with_lexile)} non-enriched books with known Lexile scores")
    
    if len(non_enriched_with_lexile) == 0:
        print("❌ No books available for ML accuracy testing")
        return
    
    # Test predictions
    results = []
    errors = []
    
    sample_size = min(100, len(non_enriched_with_lexile))
    test_books = np.random.choice(non_enriched_with_lexile, size=sample_size, replace=False)
    
    print(f"🧪 Testing {sample_size} books...")
    
    for book in test_books:
        try:
            prediction = predictor.predict(book['title'], book['author'])
            
            if prediction['lexile_score'] is not None:
                actual = book['actual_lexile']
                predicted = prediction['lexile_score']
                error = abs(predicted - actual)
                
                result = {
                    'title': book['title'],
                    'author': book['author'],
                    'actual_lexile': actual,
                    'predicted_lexile': predicted,
                    'error': error,
                    'relative_error': (error / actual) * 100 if actual > 0 else 0,
                    'source': prediction['source']
                }
                
                results.append(result)
                errors.append(error)
                
        except Exception as e:
            logger.warning(f"⚠️ Prediction failed for {book['title']}: {e}")
    
    if not results:
        print("❌ No successful predictions to analyze")
        return
    
    # Calculate statistics
    print(f"\n📊 ML Accuracy Analysis ({len(results)} successful predictions)")
    print("-" * 60)
    
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    std_error = np.std(errors)
    
    print(f"📈 Error Statistics:")
    print(f"   Mean Absolute Error: {mean_error:.1f}L")
    print(f"   Median Absolute Error: {median_error:.1f}L")
    print(f"   Standard Deviation: {std_error:.1f}L")
    print(f"   Min Error: {np.min(errors):.1f}L")
    print(f"   Max Error: {np.max(errors):.1f}L")
    
    # Accuracy thresholds
    within_50 = len([e for e in errors if e <= 50]) / len(errors) * 100
    within_100 = len([e for e in errors if e <= 100]) / len(errors) * 100
    within_200 = len([e for e in errors if e <= 200]) / len(errors) * 100
    
    print(f"\n🎯 Accuracy Thresholds:")
    print(f"   Within ±50L: {within_50:.1f}%")
    print(f"   Within ±100L: {within_100:.1f}%")
    print(f"   Within ±200L: {within_200:.1f}%")
    
    # Show some examples
    print(f"\n📖 Sample Predictions:")
    print("-" * 60)
    
    # Sort by error for analysis
    results_sorted = sorted(results, key=lambda x: x['error'])
    
    print("🎯 Best Predictions (lowest error):")
    for i, result in enumerate(results_sorted[:5]):
        print(f"   {i+1}. {result['title']}: {result['actual_lexile']:.0f}L → {result['predicted_lexile']:.0f}L (error: {result['error']:.0f}L)")
    
    print("\n😰 Worst Predictions (highest error):")
    for i, result in enumerate(results_sorted[-5:]):
        print(f"   {i+1}. {result['title']}: {result['actual_lexile']:.0f}L → {result['predicted_lexile']:.0f}L (error: {result['error']:.0f}L)")
    
    # Analyze by reading level
    print(f"\n📚 Accuracy by Reading Level:")
    print("-" * 60)
    
    levels = {
        'Early Readers (0-300L)': [r for r in results if r['actual_lexile'] <= 300],
        'Elementary (300-600L)': [r for r in results if 300 < r['actual_lexile'] <= 600],
        'Middle Grade (600-900L)': [r for r in results if 600 < r['actual_lexile'] <= 900],
        'Advanced (900L+)': [r for r in results if r['actual_lexile'] > 900]
    }
    
    for level_name, level_results in levels.items():
        if level_results:
            level_errors = [r['error'] for r in level_results]
            level_within_100 = len([e for e in level_errors if e <= 100]) / len(level_errors) * 100
            print(f"   {level_name}: {np.mean(level_errors):.1f}L avg error, {level_within_100:.1f}% within ±100L ({len(level_results)} books)")
    
    # Quality assessment
    print(f"\n🔍 ML Quality Assessment:")
    print("-" * 60)
    
    if mean_error <= 100:
        quality = "🟢 GOOD"
    elif mean_error <= 150:
        quality = "🟡 FAIR"
    else:
        quality = "🔴 POOR"
    
    print(f"Overall ML Quality: {quality}")
    print(f"Mean Error: {mean_error:.1f}L (Industry standard: <100L)")
    print(f"Reliability: {within_100:.1f}% within ±100L (Target: >70%)")
    
    return results, errors

if __name__ == "__main__":
    test_ml_accuracy()