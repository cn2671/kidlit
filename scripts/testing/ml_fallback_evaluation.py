#!/usr/bin/env python3
"""
ML Fallback Quality Evaluation Script
Tests the quality of ML predictions for books not in enriched database
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.core.enriched_predictor import EnrichedLexilePredictor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLFallbackEvaluator:
    """Comprehensive evaluation of ML fallback prediction quality"""
    
    def __init__(self):
        self.predictor = EnrichedLexilePredictor()
        self.catalog = None
        self.enriched_books = set()
        self.non_enriched_books = []
        self.results = {}
        
    def load_catalog(self, catalog_path: str = None):
        """Load the book catalog"""
        if catalog_path is None:
            catalog_path = ROOT / "data" / "raw" / "books_final_complete.csv"
        
        try:
            self.catalog = pd.read_csv(catalog_path)
            logger.info(f"📚 Loaded catalog with {len(self.catalog)} books")
            return True
        except Exception as e:
            logger.error(f"❌ Error loading catalog: {e}")
            return False
    
    def identify_enriched_vs_non_enriched(self):
        """Separate books into enriched vs non-enriched categories"""
        enriched_count = 0
        non_enriched_count = 0
        
        for _, row in self.catalog.iterrows():
            book_key = self.predictor._normalize_book_key(row['title'], row.get('author', ''))
            
            if book_key in self.predictor.enriched_scores:
                self.enriched_books.add(book_key)
                enriched_count += 1
            else:
                self.non_enriched_books.append({
                    'title': row['title'],
                    'author': row.get('author', ''),
                    'original_lexile': row.get('lexile_score', None),
                    'age_min': row.get('age_min', None),
                    'age_max': row.get('age_max', None),
                    'genres': row.get('genres', ''),
                    'book_key': book_key
                })
                non_enriched_count += 1
        
        logger.info(f"📊 Found {enriched_count} enriched books and {non_enriched_count} non-enriched books")
        return enriched_count, non_enriched_count
    
    def evaluate_ml_predictions(self, sample_size: int = 100):
        """Evaluate ML prediction quality on a sample of non-enriched books"""
        logger.info(f"🧪 Evaluating ML predictions on {sample_size} non-enriched books")
        
        # Sample books for evaluation
        if sample_size > len(self.non_enriched_books):
            sample_size = len(self.non_enriched_books)
        
        sample_books = np.random.choice(self.non_enriched_books, size=sample_size, replace=False)
        
        predictions = []
        original_scores = []
        errors = []
        
        for book in sample_books:
            try:
                # Get ML prediction
                prediction = self.predictor.predict(book['title'], book['author'])
                
                pred_result = {
                    'title': book['title'],
                    'author': book['author'],
                    'predicted_lexile': prediction['lexile_score'],
                    'confidence': prediction['confidence'],
                    'source': prediction['source'],
                    'method': prediction['method'],
                    'original_lexile': book['original_lexile']
                }
                
                predictions.append(pred_result)
                
                # Calculate error if original Lexile available
                if book['original_lexile'] and not pd.isna(book['original_lexile']):
                    error = abs(prediction['lexile_score'] - book['original_lexile'])
                    errors.append(error)
                    original_scores.append(book['original_lexile'])
                
            except Exception as e:
                logger.warning(f"⚠️ Prediction failed for {book['title']}: {e}")
        
        return predictions, errors, original_scores
    
    def analyze_ml_quality(self, predictions: List[Dict], errors: List[float], original_scores: List[float]):
        """Analyze the quality of ML predictions"""
        analysis = {
            'total_predictions': len(predictions),
            'successful_predictions': len([p for p in predictions if p['predicted_lexile'] is not None]),
            'prediction_sources': {},
            'lexile_distribution': {},
            'error_statistics': {}
        }
        
        # Source distribution
        sources = [p['source'] for p in predictions]
        analysis['prediction_sources'] = {source: sources.count(source) for source in set(sources)}
        
        # Lexile score distribution
        valid_predictions = [p['predicted_lexile'] for p in predictions if p['predicted_lexile'] is not None]
        if valid_predictions:
            analysis['lexile_distribution'] = {
                'mean': np.mean(valid_predictions),
                'median': np.median(valid_predictions),
                'std': np.std(valid_predictions),
                'min': np.min(valid_predictions),
                'max': np.max(valid_predictions),
                'range_0_200': len([p for p in valid_predictions if 0 <= p <= 200]),
                'range_200_500': len([p for p in valid_predictions if 200 < p <= 500]),
                'range_500_800': len([p for p in valid_predictions if 500 < p <= 800]),
                'range_800_plus': len([p for p in valid_predictions if p > 800])
            }
        
        # Error statistics (when original Lexile available)
        if errors:
            analysis['error_statistics'] = {
                'mean_absolute_error': np.mean(errors),
                'median_absolute_error': np.median(errors),
                'std_error': np.std(errors),
                'min_error': np.min(errors),
                'max_error': np.max(errors),
                'error_under_50': len([e for e in errors if e <= 50]) / len(errors) * 100,
                'error_under_100': len([e for e in errors if e <= 100]) / len(errors) * 100,
                'error_under_200': len([e for e in errors if e <= 200]) / len(errors) * 100,
                'samples_with_original': len(errors)
            }
        
        return analysis
    
    def test_specific_genres(self):
        """Test ML predictions on specific genres to identify strengths/weaknesses"""
        genre_performance = {}
        
        # Group books by primary genre
        genre_books = {}
        for book in self.non_enriched_books:
            genres = str(book.get('genres', '')).lower()
            primary_genre = 'unknown'
            
            if 'picture' in genres:
                primary_genre = 'picture_books'
            elif 'fantasy' in genres:
                primary_genre = 'fantasy'
            elif 'mystery' in genres:
                primary_genre = 'mystery'
            elif 'adventure' in genres:
                primary_genre = 'adventure'
            elif 'realistic' in genres:
                primary_genre = 'realistic_fiction'
            elif 'biography' in genres:
                primary_genre = 'biography'
            
            if primary_genre not in genre_books:
                genre_books[primary_genre] = []
            genre_books[primary_genre].append(book)
        
        logger.info(f"📊 Testing ML performance by genre: {list(genre_books.keys())}")
        
        for genre, books in genre_books.items():
            if len(books) >= 5:  # Only test genres with sufficient samples
                sample_size = min(20, len(books))
                sample_books = np.random.choice(books, size=sample_size, replace=False)
                
                predictions = []
                for book in sample_books:
                    try:
                        prediction = self.predictor.predict(book['title'], book['author'])
                        predictions.append(prediction['lexile_score'])
                    except:
                        continue
                
                if predictions:
                    genre_performance[genre] = {
                        'count': len(predictions),
                        'mean_lexile': np.mean(predictions),
                        'std_lexile': np.std(predictions),
                        'range': [np.min(predictions), np.max(predictions)]
                    }
        
        return genre_performance
    
    def test_age_range_accuracy(self):
        """Test if ML predictions align with expected reading age ranges"""
        age_accuracy = {}
        
        age_groups = {
            'early_readers': (3, 6),
            'elementary': (6, 9),
            'middle_grade': (9, 12),
            'young_adult': (12, 18)
        }
        
        for age_group, (min_age, max_age) in age_groups.items():
            matching_books = [
                book for book in self.non_enriched_books 
                if book.get('age_min') and book.get('age_max') and
                book['age_min'] >= min_age and book['age_max'] <= max_age
            ]
            
            if len(matching_books) >= 10:
                sample_size = min(50, len(matching_books))
                sample_books = np.random.choice(matching_books, size=sample_size, replace=False)
                
                predictions = []
                for book in sample_books:
                    try:
                        prediction = self.predictor.predict(book['title'], book['author'])
                        if prediction['lexile_score']:
                            predictions.append(prediction['lexile_score'])
                    except:
                        continue
                
                if predictions:
                    age_accuracy[age_group] = {
                        'count': len(predictions),
                        'mean_lexile': np.mean(predictions),
                        'expected_range': self._get_expected_lexile_range(age_group),
                        'within_expected': self._count_within_expected_range(predictions, age_group)
                    }
        
        return age_accuracy
    
    def _get_expected_lexile_range(self, age_group: str) -> Tuple[int, int]:
        """Get expected Lexile range for age group"""
        ranges = {
            'early_readers': (0, 300),
            'elementary': (200, 600),
            'middle_grade': (500, 900),
            'young_adult': (700, 1200)
        }
        return ranges.get(age_group, (0, 1000))
    
    def _count_within_expected_range(self, predictions: List[float], age_group: str) -> float:
        """Count percentage of predictions within expected range"""
        min_expected, max_expected = self._get_expected_lexile_range(age_group)
        within_range = len([p for p in predictions if min_expected <= p <= max_expected])
        return (within_range / len(predictions)) * 100 if predictions else 0
    
    def generate_report(self, analysis: Dict, genre_performance: Dict, age_accuracy: Dict):
        """Generate comprehensive evaluation report"""
        report = f"""
# 🧪 ML Fallback Quality Evaluation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Overall ML Prediction Quality

### Prediction Success Rate
- **Total Predictions**: {analysis['total_predictions']}
- **Successful Predictions**: {analysis['successful_predictions']}
- **Success Rate**: {(analysis['successful_predictions']/analysis['total_predictions']*100):.1f}%

### Prediction Sources
"""
        for source, count in analysis['prediction_sources'].items():
            percentage = (count / analysis['total_predictions']) * 100
            report += f"- **{source}**: {count} ({percentage:.1f}%)\n"
        
        if 'lexile_distribution' in analysis:
            dist = analysis['lexile_distribution']
            report += f"""
### Lexile Score Distribution
- **Mean**: {dist['mean']:.0f}L
- **Median**: {dist['median']:.0f}L
- **Standard Deviation**: {dist['std']:.0f}L
- **Range**: {dist['min']:.0f}L - {dist['max']:.0f}L

### Reading Level Distribution
- **Early Readers (0-200L)**: {dist['range_0_200']} books
- **Elementary (200-500L)**: {dist['range_200_500']} books
- **Middle Grade (500-800L)**: {dist['range_500_800']} books
- **Advanced (800L+)**: {dist['range_800_plus']} books
"""
        
        if 'error_statistics' in analysis and analysis['error_statistics']:
            err = analysis['error_statistics']
            report += f"""
## 🎯 Accuracy Analysis (vs Original Lexile Scores)
*Based on {err['samples_with_original']} books with known Lexile scores*

### Error Statistics
- **Mean Absolute Error**: {err['mean_absolute_error']:.0f}L
- **Median Absolute Error**: {err['median_absolute_error']:.0f}L
- **Standard Deviation**: {err['std_error']:.0f}L
- **Error Range**: {err['min_error']:.0f}L - {err['max_error']:.0f}L

### Accuracy Thresholds
- **Within ±50L**: {err['error_under_50']:.1f}%
- **Within ±100L**: {err['error_under_100']:.1f}%
- **Within ±200L**: {err['error_under_200']:.1f}%
"""
        
        if genre_performance:
            report += "\n## 📚 Performance by Genre\n"
            for genre, perf in genre_performance.items():
                report += f"- **{genre.replace('_', ' ').title()}**: {perf['mean_lexile']:.0f}L ±{perf['std_lexile']:.0f} ({perf['count']} books)\n"
        
        if age_accuracy:
            report += "\n## 👶 Age Range Accuracy\n"
            for age_group, acc in age_accuracy.items():
                expected_min, expected_max = acc['expected_range']
                report += f"- **{age_group.replace('_', ' ').title()}**: {acc['mean_lexile']:.0f}L (expected {expected_min}-{expected_max}L) - {acc['within_expected']:.1f}% within range\n"
        
        return report
    
    def run_comprehensive_evaluation(self, sample_size: int = 200):
        """Run complete ML fallback evaluation"""
        logger.info("🚀 Starting comprehensive ML fallback evaluation")
        
        # Load catalog and identify book categories
        if not self.load_catalog():
            return None
        
        enriched_count, non_enriched_count = self.identify_enriched_vs_non_enriched()
        
        # Evaluate ML predictions
        predictions, errors, original_scores = self.evaluate_ml_predictions(sample_size)
        
        # Analyze quality
        analysis = self.analyze_ml_quality(predictions, errors, original_scores)
        
        # Test specific areas
        genre_performance = self.test_specific_genres()
        age_accuracy = self.test_age_range_accuracy()
        
        # Generate report
        report = self.generate_report(analysis, genre_performance, age_accuracy)
        
        # Save results
        self.results = {
            'analysis': analysis,
            'genre_performance': genre_performance,
            'age_accuracy': age_accuracy,
            'sample_predictions': predictions[:20],  # Save first 20 as examples
            'enriched_count': enriched_count,
            'non_enriched_count': non_enriched_count
        }
        
        return report

def main():
    """Run ML fallback evaluation"""
    print("🧪 ML Fallback Quality Evaluation")
    print("=" * 50)
    
    evaluator = MLFallbackEvaluator()
    report = evaluator.run_comprehensive_evaluation(sample_size=200)
    
    if report:
        print(report)
        
        # Save report to file
        report_path = ROOT / "ML_FALLBACK_EVALUATION_REPORT.md"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\n📄 Report saved to: {report_path}")
        
        # Save detailed results
        results_path = ROOT / "ml_fallback_results.json"
        with open(results_path, 'w') as f:
            json.dump(evaluator.results, f, indent=2, default=str)
        print(f"📊 Detailed results saved to: {results_path}")
    
    else:
        print("❌ Evaluation failed")

if __name__ == "__main__":
    main()