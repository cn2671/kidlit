#!/usr/bin/env python3
"""Simple test of the integrated Lexile predictor"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from production_lexile_predictor import ProductionLexilePredictor
    
    print("🧪 TESTING PRODUCTION LEXILE PREDICTOR")
    print("=" * 50)
    
    # Initialize predictor
    predictor = ProductionLexilePredictor()
    print("✅ Predictor initialized successfully")
    
    # Test case 1: Modern book
    print("\n📖 Testing: The Very Hungry Caterpillar")
    result1 = predictor.predict_lexile(
        title="The Very Hungry Caterpillar",
        author="Eric Carle",
        age_min=3,
        age_max=7,
        book_type="Adult_Directed",
        notes="Classic picture book"
    )
    print(f"   Predicted: {result1['predicted_lexile']}L")
    print(f"   Confidence: {result1['confidence_level']}")
    print(f"   Edge Case: {result1['is_edge_case']}")
    
    # Test case 2: Vintage classic
    print("\n📖 Testing: The Poky Little Puppy")
    result2 = predictor.predict_lexile(
        title="The Poky Little Puppy",
        author="Janette Sebring Lowrey",
        age_min=2,
        age_max=5,
        book_type="Standard_Lexile",
        notes="1942 Little Golden Book classic"
    )
    print(f"   Predicted: {result2['predicted_lexile']}L")
    print(f"   Confidence: {result2['confidence_level']}")
    print(f"   Edge Case: {result2['is_edge_case']}")
    if result2.get('warning'):
        print(f"   Warning: {result2['warning']}")
    if result2.get('prediction_range'):
        print(f"   Range: {result2['prediction_range']}")
    
    print("\n✅ Integration test successful!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()