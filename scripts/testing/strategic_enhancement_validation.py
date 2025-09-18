#!/usr/bin/env python3
"""
Strategic Enhancement Validation - Test ML improvements after strategic coverage
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

def test_strategic_enhancement_impact():
    """Test the impact of strategic enhancement on ML fallback quality"""
    print("🎯 Testing Strategic Enhancement Impact on ML Fallback")
    print("=" * 70)
    
    predictor = EnrichedLexilePredictor()
    
    # Books from our strategic enhancement that should now have enriched scores
    strategic_books = [
        # Early Readers - should now be enriched (perfect accuracy)
        {"title": "Go Dog Go", "author": "P.D. Eastman", "expected_enriched": True, "expected_lexile": 160, "category": "Early Reader"},
        {"title": "Are You My Mother", "author": "P.D. Eastman", "expected_enriched": True, "expected_lexile": 190, "category": "Early Reader"},
        {"title": "Put Me in the Zoo", "author": "Robert Lopshire", "expected_enriched": True, "expected_lexile": 220, "category": "Early Reader"},
        {"title": "Frog and Toad Are Friends", "author": "Arnold Lobel", "expected_enriched": True, "expected_lexile": 400, "category": "Early Reader Bridge"},
        {"title": "I Really Like Slop", "author": "Mo Willems", "expected_enriched": True, "expected_lexile": 150, "category": "Early Reader"},
        
        # Advanced Books - should now be enriched (perfect accuracy)
        {"title": "Bridge to Terabithia", "author": "Katherine Paterson", "expected_enriched": True, "expected_lexile": 810, "category": "Advanced"},
        {"title": "Wonder", "author": "R.J. Palacio", "expected_enriched": True, "expected_lexile": 790, "category": "Advanced"},
        {"title": "Hatchet", "author": "Gary Paulsen", "expected_enriched": True, "expected_lexile": 1020, "category": "Advanced"},
        {"title": "The Giver", "author": "Lois Lowry", "expected_enriched": True, "expected_lexile": 760, "category": "Advanced"},
        {"title": "Holes", "author": "Louis Sachar", "expected_enriched": True, "expected_lexile": 660, "category": "Advanced"},
    ]
    
    # Test similar books that should still use ML fallback
    ml_fallback_books = [
        # Early readers not in our strategic enhancement
        {"title": "Frog and Toad Together", "author": "Arnold Lobel", "expected_enriched": False, "category": "Early Reader", "expected_range": (300, 500)},
        {"title": "Little Bear", "author": "Else Holmelund Minarik", "expected_enriched": False, "category": "Early Reader", "expected_range": (200, 400)},
        {"title": "Amelia Bedelia Goes Camping", "author": "Peggy Parish", "expected_enriched": False, "category": "Early Reader", "expected_range": (400, 600)},
        
        # Advanced books not in our strategic enhancement
        {"title": "Tuck Everlasting", "author": "Natalie Babbitt", "expected_enriched": False, "category": "Advanced", "expected_range": (700, 900)},
        {"title": "The Outsiders", "author": "S.E. Hinton", "expected_enriched": False, "category": "Advanced", "expected_range": (700, 900)},
        {"title": "Freak the Mighty", "author": "Rodman Philbrick", "expected_enriched": False, "category": "Advanced", "expected_range": (600, 800)},
    ]
    
    print("🧪 Testing Strategic Enhancement Books (Should Use Enriched Scores)")
    print("-" * 70)
    
    enriched_results = []
    enriched_perfect = 0
    
    for book in strategic_books:
        try:
            prediction = predictor.predict(book['title'], book['author'])
            
            is_enriched = prediction['source'] == 'enriched'
            is_perfect = prediction['lexile_score'] == book['expected_lexile']
            
            if is_enriched and is_perfect:
                enriched_perfect += 1
                status = "✅ PERFECT"
            elif is_enriched:
                status = "✅ ENRICHED"
            else:
                status = "❌ ML FALLBACK"
            
            print(f"{status} {book['title']}: {prediction['lexile_score']:.0f}L (expected: {book['expected_lexile']}L)")
            print(f"   Source: {prediction['source']}, Confidence: {prediction['confidence']:.2f}")
            
            enriched_results.append({
                'book': book,
                'prediction': prediction,
                'is_enriched': is_enriched,
                'is_perfect': is_perfect
            })
            
        except Exception as e:
            print(f"❌ ERROR {book['title']}: {e}")
    
    print(f"\n📊 Strategic Enhancement Results:")
    print(f"   Perfect Enriched Predictions: {enriched_perfect}/{len(strategic_books)} ({enriched_perfect/len(strategic_books)*100:.1f}%)")
    
    print(f"\n🤖 Testing ML Fallback Books (Should Show Improved Accuracy)")
    print("-" * 70)
    
    ml_results = []
    ml_within_range = 0
    
    for book in ml_fallback_books:
        try:
            prediction = predictor.predict(book['title'], book['author'])
            
            is_ml = prediction['source'] == 'ml_model'
            min_expected, max_expected = book['expected_range']
            within_range = min_expected <= prediction['lexile_score'] <= max_expected
            
            if within_range:
                ml_within_range += 1
                status = "✅ GOOD ML"
            elif is_ml:
                status = "⚠️ ML OUT OF RANGE"
            else:
                status = "🎯 ENRICHED (BONUS)"
            
            print(f"{status} {book['title']}: {prediction['lexile_score']:.0f}L (expected: {min_expected}-{max_expected}L)")
            print(f"   Source: {prediction['source']}, Confidence: {prediction['confidence']:.2f}")
            
            ml_results.append({
                'book': book,
                'prediction': prediction,
                'is_ml': is_ml,
                'within_range': within_range
            })
            
        except Exception as e:
            print(f"❌ ERROR {book['title']}: {e}")
    
    print(f"\n📊 ML Fallback Results:")
    print(f"   ML Predictions Within Range: {ml_within_range}/{len(ml_fallback_books)} ({ml_within_range/len(ml_fallback_books)*100:.1f}%)")
    
    # Overall system improvement analysis
    print(f"\n🔍 Overall System Quality Analysis:")
    print("-" * 70)
    
    enriched_rate = len([r for r in enriched_results if r['is_enriched']]) / len(enriched_results) * 100
    ml_accuracy_rate = ml_within_range / len(ml_fallback_books) * 100
    
    print(f"📈 Strategic Books Using Enriched Scores: {enriched_rate:.1f}%")
    print(f"🤖 ML Fallback Accuracy for Remaining Books: {ml_accuracy_rate:.1f}%")
    
    if enriched_rate >= 90 and ml_accuracy_rate >= 50:
        quality_rating = "🟢 EXCELLENT"
        assessment = "Strategic enhancement successfully addresses ML weaknesses"
    elif enriched_rate >= 80 and ml_accuracy_rate >= 40:
        quality_rating = "🟡 GOOD"
        assessment = "Significant improvement with minor optimization needed"
    else:
        quality_rating = "🟠 FAIR"
        assessment = "Some improvement but further enhancement recommended"
    
    print(f"\nOverall System Quality: {quality_rating}")
    print(f"Assessment: {assessment}")
    
    # Test some edge cases
    print(f"\n🧪 Testing Edge Case Improvements:")
    print("-" * 70)
    
    edge_cases = [
        {"title": "Green Eggs and Ham", "author": "Dr. Seuss", "note": "Should be enriched from previous system"},
        {"title": "Where the Wild Things Are", "author": "Maurice Sendak", "note": "Should be enriched from previous system"},
        {"title": "The Magic Tree House Dinosaurs Before Dark", "author": "Mary Pope Osborne", "note": "Should use ML fallback"},
        {"title": "Captain Underpants", "author": "Dav Pilkey", "note": "Should use ML fallback"},
    ]
    
    for case in edge_cases:
        try:
            prediction = predictor.predict(case['title'], case['author'])
            source_icon = "🎯" if prediction['source'] == 'enriched' else "🤖"
            print(f"{source_icon} {case['title']}: {prediction['lexile_score']:.0f}L ({prediction['source']})")
            print(f"   {case['note']}")
        except Exception as e:
            print(f"❌ {case['title']}: {e}")
    
    return enriched_results, ml_results, enriched_rate, ml_accuracy_rate

def test_coverage_distribution():
    """Test the distribution of coverage across reading levels"""
    print(f"\n📊 Coverage Distribution Analysis:")
    print("-" * 70)
    
    predictor = EnrichedLexilePredictor()
    
    # Sample books across different reading levels
    test_books = [
        # Early readers (0-300L)
        ("Hop on Pop", "Dr. Seuss"),
        ("Go Dog Go", "P.D. Eastman"),
        ("Green Eggs and Ham", "Dr. Seuss"),
        
        # Elementary (300-600L)
        ("Frog and Toad Are Friends", "Arnold Lobel"),
        ("Junie B. Jones", "Barbara Park"),
        ("Magic Tree House", "Mary Pope Osborne"),
        
        # Middle Grade (600-900L)
        ("Wonder", "R.J. Palacio"),
        ("Holes", "Louis Sachar"),
        ("Bridge to Terabithia", "Katherine Paterson"),
        
        # Advanced (900L+)
        ("Hatchet", "Gary Paulsen"),
        ("The Giver", "Lois Lowry"),
        ("Where the Red Fern Grows", "Wilson Rawls"),
    ]
    
    enriched_count = 0
    by_level = {
        'early': {'enriched': 0, 'total': 0},
        'elementary': {'enriched': 0, 'total': 0},
        'middle': {'enriched': 0, 'total': 0},
        'advanced': {'enriched': 0, 'total': 0}
    }
    
    for title, author in test_books:
        try:
            prediction = predictor.predict(title, author)
            is_enriched = prediction['source'] == 'enriched'
            lexile = prediction['lexile_score']
            
            if is_enriched:
                enriched_count += 1
            
            # Categorize by level
            if lexile <= 300:
                level = 'early'
            elif lexile <= 600:
                level = 'elementary'
            elif lexile <= 900:
                level = 'middle'
            else:
                level = 'advanced'
            
            by_level[level]['total'] += 1
            if is_enriched:
                by_level[level]['enriched'] += 1
            
            source_icon = "🎯" if is_enriched else "🤖"
            print(f"{source_icon} {title}: {lexile:.0f}L ({level})")
            
        except Exception as e:
            print(f"❌ {title}: {e}")
    
    print(f"\n📈 Coverage by Reading Level:")
    for level, stats in by_level.items():
        if stats['total'] > 0:
            coverage = (stats['enriched'] / stats['total']) * 100
            print(f"   {level.title()}: {stats['enriched']}/{stats['total']} ({coverage:.1f}% enriched)")
    
    overall_coverage = (enriched_count / len(test_books)) * 100
    print(f"\nOverall Sample Coverage: {enriched_count}/{len(test_books)} ({overall_coverage:.1f}% enriched)")

if __name__ == "__main__":
    enriched_results, ml_results, enriched_rate, ml_accuracy_rate = test_strategic_enhancement_impact()
    test_coverage_distribution()