#!/usr/bin/env python3
"""
Test script for the new Lexile prediction API endpoints
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:5001"

def test_lexile_api():
    """Test the Lexile prediction API endpoints"""
    
    print("🧪 TESTING LEXILE PREDICTION API")
    print("=" * 50)
    
    # Test 1: Model Info
    print("\n1️⃣  Testing Model Info Endpoint")
    try:
        response = requests.get(f"{BASE_URL}/api/lexile-model-info", timeout=5)
        if response.status_code == 200:
            info = response.json()
            print(f"   ✅ Model Available: {info.get('model_available')}")
            print(f"   📊 Version: {info.get('model_version')}")
            print(f"   🎯 Overall MAE: {info.get('performance_metrics', {}).get('overall_mae')}")
            print(f"   📚 Models Loaded: {info.get('models_loaded')}")
        else:
            print(f"   ❌ Failed: Status {response.status_code}")
    except Exception as e:
        print(f"   ⚠️  Connection error: {e}")
    
    # Test 2: Single Prediction
    print("\n2️⃣  Testing Single Book Prediction")
    test_book = {
        "title": "The Very Hungry Caterpillar",
        "author": "Eric Carle",
        "age_min": 3,
        "age_max": 7,
        "book_type": "Adult_Directed",
        "notes": "Classic picture book about caterpillar transformation"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/predict-lexile", 
                               json=test_book, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"   📖 Title: {result.get('title')}")
            print(f"   🔮 Predicted Lexile: {result.get('predicted_lexile')}L")
            print(f"   📈 Confidence: {result.get('confidence_level')}")
            print(f"   ⚠️  Edge Case: {result.get('is_edge_case', False)}")
            if result.get('warning'):
                print(f"   ⚠️  Warning: {result.get('warning')}")
        else:
            print(f"   ❌ Failed: Status {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ⚠️  Connection error: {e}")
    
    # Test 3: Edge Case (Vintage Classic)
    print("\n3️⃣  Testing Edge Case - Vintage Classic")
    vintage_book = {
        "title": "The Poky Little Puppy",
        "author": "Janette Sebring Lowrey",
        "age_min": 2,
        "age_max": 5,
        "book_type": "Standard_Lexile",
        "notes": "Classic 1942 Little Golden Book"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/predict-lexile", 
                               json=vintage_book, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"   📖 Title: {result.get('title')}")
            print(f"   🔮 Predicted Lexile: {result.get('predicted_lexile')}L")
            print(f"   📈 Confidence: {result.get('confidence_level')}")
            print(f"   🌟 Edge Case: {result.get('is_edge_case', False)}")
            if result.get('warning'):
                print(f"   ⚠️  Warning: {result.get('warning')}")
            if result.get('prediction_range'):
                print(f"   📊 Likely Range: {result.get('prediction_range')}")
        else:
            print(f"   ❌ Failed: Status {response.status_code}")
    except Exception as e:
        print(f"   ⚠️  Connection error: {e}")
    
    # Test 4: Batch Prediction
    print("\n4️⃣  Testing Batch Prediction")
    batch_books = {
        "books": [
            {
                "title": "Green Eggs and Ham",
                "author": "Dr. Seuss",
                "age_min": 3,
                "age_max": 7,
                "book_type": "Standard_Lexile"
            },
            {
                "title": "Charlotte's Web", 
                "author": "E.B. White",
                "age_min": 8,
                "age_max": 12,
                "book_type": "Standard_Lexile"
            }
        ]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/batch-predict-lexile", 
                               json=batch_books, timeout=15)
        if response.status_code == 200:
            result = response.json()
            summary = result.get('summary', {})
            print(f"   📊 Total Books: {summary.get('total_books')}")
            print(f"   ✅ Successful: {summary.get('successful_predictions')}")
            print(f"   ❌ Failed: {summary.get('failed_predictions')}")
            
            for i, book_result in enumerate(result.get('results', [])[:2]):
                print(f"   📖 Book {i+1}: {book_result.get('title')} → {book_result.get('predicted_lexile')}L")
        else:
            print(f"   ❌ Failed: Status {response.status_code}")
    except Exception as e:
        print(f"   ⚠️  Connection error: {e}")
    
    print(f"\n✅ API testing complete!")

if __name__ == "__main__":
    print("⏳ Waiting 3 seconds for server to be ready...")
    time.sleep(3)
    test_lexile_api()