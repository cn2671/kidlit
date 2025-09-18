#!/usr/bin/env python3
"""
ML Quality Assessment - Test ML predictions against benchmark books
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

def test_ml_quality():
    """Test ML quality using benchmark books with known characteristics"""
    print("🧪 ML Quality Assessment - Testing Against Benchmark Books")
    print("=" * 70)
    
    predictor = EnrichedLexilePredictor()
    
    # Benchmark books with expected characteristics
    # These are books NOT in our enriched database but with well-known reading levels
    benchmark_books = [
        # Early Readers (Expected: 0-300L)
        {"title": "Go, Dog. Go!", "author": "P.D. Eastman", "expected_range": (50, 250), "category": "Early Reader"},
        {"title": "Are You My Mother?", "author": "P.D. Eastman", "expected_range": (100, 300), "category": "Early Reader"},
        {"title": "Put Me in the Zoo", "author": "Robert Lopshire", "expected_range": (150, 350), "category": "Early Reader"},
        
        # Elementary (Expected: 300-600L)
        {"title": "Frog and Toad Are Friends", "author": "Arnold Lobel", "expected_range": (300, 500), "category": "Elementary"},
        {"title": "Henry and Mudge", "author": "Cynthia Rylant", "expected_range": (350, 550), "category": "Elementary"},
        {"title": "The Magic Tree House", "author": "Mary Pope Osborne", "expected_range": (300, 500), "category": "Elementary"},
        
        # Middle Grade (Expected: 500-800L)
        {"title": "Judy Moody", "author": "Megan McDonald", "expected_range": (500, 700), "category": "Middle Grade"},
        {"title": "Junie B. Jones", "author": "Barbara Park", "expected_range": (400, 600), "category": "Middle Grade"},
        {"title": "Ramona the Pest", "author": "Beverly Cleary", "expected_range": (600, 800), "category": "Middle Grade"},
        
        # Advanced (Expected: 700L+)
        {"title": "Bridge to Terabithia", "author": "Katherine Paterson", "expected_range": (700, 900), "category": "Advanced"},
        {"title": "Where the Red Fern Grows", "author": "Wilson Rawls", "expected_range": (800, 1000), "category": "Advanced"},
        {"title": "Island of the Blue Dolphins", "author": "Scott O'Dell", "expected_range": (750, 950), "category": "Advanced"},
    ]
    
    print(f"📚 Testing {len(benchmark_books)} benchmark books across reading levels")
    print("-" * 70)
    
    results = []
    category_performance = {}
    
    for book in benchmark_books:
        try:
            prediction = predictor.predict(book['title'], book['author'])
            
            if prediction['lexile_score'] is not None:
                predicted_lexile = prediction['lexile_score']
                min_expected, max_expected = book['expected_range']
                within_range = min_expected <= predicted_lexile <= max_expected
                
                # Calculate how far off from expected range
                if predicted_lexile < min_expected:
                    range_error = min_expected - predicted_lexile
                elif predicted_lexile > max_expected:
                    range_error = predicted_lexile - max_expected
                else:
                    range_error = 0
                
                result = {
                    'title': book['title'],
                    'author': book['author'],
                    'predicted_lexile': predicted_lexile,
                    'expected_range': book['expected_range'],
                    'category': book['category'],
                    'within_range': within_range,
                    'range_error': range_error,
                    'source': prediction['source'],
                    'confidence': prediction['confidence']
                }
                
                results.append(result)
                
                # Track by category
                if book['category'] not in category_performance:
                    category_performance[book['category']] = []
                category_performance[book['category']].append(result)
                
                status = "✅" if within_range else "❌"
                print(f"{status} {book['title']}: {predicted_lexile:.0f}L (expected: {min_expected}-{max_expected}L)")
                
            else:
                print(f"❌ {book['title']}: Prediction failed")
                
        except Exception as e:
            print(f"❌ {book['title']}: Error - {e}")
    
    if not results:
        print("❌ No successful predictions to analyze")
        return
    
    # Overall Analysis
    print(f"\n📊 Overall ML Quality Analysis")
    print("-" * 70)
    
    successful_predictions = len(results)
    within_range_count = len([r for r in results if r['within_range']])
    accuracy_rate = (within_range_count / successful_predictions) * 100
    
    print(f"📈 Success Metrics:")
    print(f"   Successful Predictions: {successful_predictions}/{len(benchmark_books)} ({successful_predictions/len(benchmark_books)*100:.1f}%)")
    print(f"   Within Expected Range: {within_range_count}/{successful_predictions} ({accuracy_rate:.1f}%)")
    
    # Range errors
    range_errors = [r['range_error'] for r in results]
    mean_range_error = np.mean(range_errors)
    
    print(f"   Average Range Error: {mean_range_error:.1f}L")
    print(f"   Perfect Predictions: {len([e for e in range_errors if e == 0])}")
    
    # Category Analysis
    print(f"\n📚 Performance by Reading Level:")
    print("-" * 70)
    
    for category, cat_results in category_performance.items():
        cat_within_range = len([r for r in cat_results if r['within_range']])
        cat_accuracy = (cat_within_range / len(cat_results)) * 100
        cat_predictions = [r['predicted_lexile'] for r in cat_results]
        cat_mean = np.mean(cat_predictions)
        
        print(f"📖 {category}:")
        print(f"   Accuracy: {cat_within_range}/{len(cat_results)} ({cat_accuracy:.1f}%)")
        print(f"   Average Prediction: {cat_mean:.0f}L")
        
        # Show individual results
        for result in cat_results:
            status = "✅" if result['within_range'] else "❌"
            min_exp, max_exp = result['expected_range']
            print(f"     {status} {result['title']}: {result['predicted_lexile']:.0f}L (expected: {min_exp}-{max_exp}L)")
        print()
    
    # Quality Assessment
    print(f"🔍 ML Quality Assessment:")
    print("-" * 70)
    
    if accuracy_rate >= 80:
        quality_rating = "🟢 EXCELLENT"
        assessment = "ML predictions are highly reliable for educational use"
    elif accuracy_rate >= 60:
        quality_rating = "🟡 GOOD"
        assessment = "ML predictions are reliable with minor adjustments needed"
    elif accuracy_rate >= 40:
        quality_rating = "🟠 FAIR"
        assessment = "ML predictions provide useful estimates but need improvement"
    else:
        quality_rating = "🔴 POOR"
        assessment = "ML predictions are not reliable for educational use"
    
    print(f"Overall Quality: {quality_rating}")
    print(f"Accuracy Rate: {accuracy_rate:.1f}% (Target: >60%)")
    print(f"Assessment: {assessment}")
    
    # Prediction distribution
    all_predictions = [r['predicted_lexile'] for r in results]
    print(f"\n📊 Prediction Distribution:")
    print(f"   Range: {np.min(all_predictions):.0f}L - {np.max(all_predictions):.0f}L")
    print(f"   Mean: {np.mean(all_predictions):.0f}L")
    print(f"   Median: {np.median(all_predictions):.0f}L")
    print(f"   Std Dev: {np.std(all_predictions):.0f}L")
    
    return results, category_performance, accuracy_rate

def test_catalog_sample():
    """Test a random sample of catalog books for consistency"""
    print(f"\n🎲 Testing Random Catalog Sample for Consistency")
    print("-" * 70)
    
    predictor = EnrichedLexilePredictor()
    
    # Load catalog
    catalog_path = ROOT / "data" / "raw" / "books_final_complete.csv"
    catalog = pd.read_csv(catalog_path)
    
    # Filter out enriched books
    non_enriched_books = []
    for _, row in catalog.iterrows():
        book_key = predictor._normalize_book_key(row['title'], row.get('author', ''))
        if book_key not in predictor.enriched_scores:
            non_enriched_books.append({
                'title': row['title'],
                'author': row.get('author', ''),
                'age_range_llm': row.get('age_range_llm', ''),
                'reading_level_llm': row.get('reading_level_llm', '')
            })
    
    # Sample random books
    sample_size = min(50, len(non_enriched_books))
    sample_books = np.random.choice(non_enriched_books, size=sample_size, replace=False)
    
    predictions = []
    sources = []
    
    print(f"🧪 Testing {sample_size} random catalog books...")
    
    for book in sample_books:
        try:
            prediction = predictor.predict(book['title'], book['author'])
            if prediction['lexile_score'] is not None:
                predictions.append(prediction['lexile_score'])
                sources.append(prediction['source'])
        except:
            continue
    
    if predictions:
        print(f"📊 Sample Statistics:")
        print(f"   Successful Predictions: {len(predictions)}/{sample_size}")
        print(f"   Range: {np.min(predictions):.0f}L - {np.max(predictions):.0f}L")
        print(f"   Mean: {np.mean(predictions):.0f}L")
        print(f"   Median: {np.median(predictions):.0f}L")
        print(f"   Std Dev: {np.std(predictions):.0f}L")
        
        # Source breakdown
        source_counts = {source: sources.count(source) for source in set(sources)}
        print(f"   Sources: {source_counts}")
        
        # Distribution
        early = len([p for p in predictions if p <= 300])
        elementary = len([p for p in predictions if 300 < p <= 600])
        middle = len([p for p in predictions if 600 < p <= 900])
        advanced = len([p for p in predictions if p > 900])
        
        print(f"   Distribution:")
        print(f"     Early (≤300L): {early} ({early/len(predictions)*100:.1f}%)")
        print(f"     Elementary (300-600L): {elementary} ({elementary/len(predictions)*100:.1f}%)")
        print(f"     Middle (600-900L): {middle} ({middle/len(predictions)*100:.1f}%)")
        print(f"     Advanced (>900L): {advanced} ({advanced/len(predictions)*100:.1f}%)")

if __name__ == "__main__":
    results, category_performance, accuracy_rate = test_ml_quality()
    test_catalog_sample()