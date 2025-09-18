#!/usr/bin/env python3
"""
Test script for LLM/Web Lexile enrichment systems
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def test_web_enrichment():
    """Test the web enrichment system with a small sample"""
    print("🧪 Testing Web Lexile Enrichment System")
    print("=" * 50)
    
    # Create test data
    test_books = [
        {"title": "Charlotte's Web", "author": "E.B. White", "lexile_score": None},
        {"title": "The Very Hungry Caterpillar", "author": "Eric Carle", "lexile_score": None},
        {"title": "Where the Wild Things Are", "author": "Maurice Sendak", "lexile_score": None},
        {"title": "Green Eggs and Ham", "author": "Dr. Seuss", "lexile_score": None},
        {"title": "Wonder", "author": "R.J. Palacio", "lexile_score": None}
    ]
    
    # Save test catalog
    test_catalog_path = ROOT / "data" / "test_catalog.csv"
    test_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    
    test_df = pd.DataFrame(test_books)
    test_df.to_csv(test_catalog_path, index=False)
    print(f"📚 Created test catalog: {test_catalog_path}")
    
    # Import and run web enrichment
    try:
        from web_lexile_enrichment import WebLexileEnricher
        
        enricher = WebLexileEnricher(output_dir="data/test_output")
        
        print("\n🔍 Starting web search for Lexile scores...")
        enriched_df = enricher.enrich_catalog(
            catalog_path=str(test_catalog_path),
            sample_size=5
        )
        
        print("\n📊 Results:")
        for idx, book in enriched_df.iterrows():
            title = book['title']
            author = book['author']
            web_score = book.get('web_lexile_score')
            confidence = book.get('web_lexile_confidence', 'unknown')
            source = book.get('web_lexile_source', 'unknown')
            
            if web_score:
                print(f"✅ {title} by {author}: {web_score}L ({confidence} confidence from {source})")
            else:
                print(f"❌ {title} by {author}: No score found")
        
        print(f"\n💾 Full results saved to: {enricher.results_file}")
        
    except ImportError as e:
        print(f"❌ Could not import web enrichment: {e}")
    except Exception as e:
        print(f"❌ Error during enrichment: {e}")

def test_google_books_api():
    """Test just the Google Books API search"""
    print("\n🧪 Testing Google Books API Search")
    print("=" * 40)
    
    try:
        from web_lexile_enrichment import WebLexileEnricher
        
        enricher = WebLexileEnricher()
        
        test_books = [
            ("Charlotte's Web", "E.B. White"),
            ("The Very Hungry Caterpillar", "Eric Carle"),
            ("Wonder", "R.J. Palacio")
        ]
        
        for title, author in test_books:
            print(f"\n🔍 Searching: '{title}' by {author}")
            result = enricher.search_google_books_api(title, author)
            
            if result.get('found'):
                score = result.get('lexile_score')
                print(f"✅ Found: {score}L (confidence: {result.get('confidence')})")
                if 'description_snippet' in result:
                    print(f"   Source: {result['description_snippet'][:100]}...")
            else:
                print(f"❌ Not found in Google Books")
    
    except Exception as e:
        print(f"❌ Error testing Google Books API: {e}")

def create_sample_catalog():
    """Create a larger sample catalog for testing"""
    print("\n📚 Creating Sample Catalog for Testing")
    print("=" * 40)
    
    sample_books = [
        {"title": "Charlotte's Web", "author": "E.B. White"},
        {"title": "The Very Hungry Caterpillar", "author": "Eric Carle"},
        {"title": "Where the Wild Things Are", "author": "Maurice Sendak"},
        {"title": "Green Eggs and Ham", "author": "Dr. Seuss"},
        {"title": "Wonder", "author": "R.J. Palacio"},
        {"title": "The Cat in the Hat", "author": "Dr. Seuss"},
        {"title": "Goodnight Moon", "author": "Margaret Wise Brown"},
        {"title": "The Giving Tree", "author": "Shel Silverstein"},
        {"title": "Matilda", "author": "Roald Dahl"},
        {"title": "Diary of a Wimpy Kid", "author": "Jeff Kinney"},
        {"title": "Harry Potter and the Sorcerer's Stone", "author": "J.K. Rowling"},
        {"title": "The Lion, the Witch and the Wardrobe", "author": "C.S. Lewis"},
        {"title": "A Wrinkle in Time", "author": "Madeleine L'Engle"},
        {"title": "Bridge to Terabithia", "author": "Katherine Paterson"},
        {"title": "The Secret Garden", "author": "Frances Hodgson Burnett"}
    ]
    
    # Add some mock data
    for book in sample_books:
        book.update({
            "age_range": "6-8",  # Default
            "themes": "friendship, adventure",
            "tone": "gentle",
            "lexile_score": None,  # We'll enrich these
            "description": f"A wonderful children's book about {book['title'].lower()}."
        })
    
    # Save sample catalog
    sample_path = ROOT / "data" / "sample_catalog_for_enrichment.csv"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    
    sample_df = pd.DataFrame(sample_books)
    sample_df.to_csv(sample_path, index=False)
    
    print(f"✅ Created sample catalog with {len(sample_books)} books")
    print(f"📍 Location: {sample_path}")
    print(f"📊 Columns: {list(sample_df.columns)}")
    
    return sample_path

def show_usage_examples():
    """Show usage examples for the enrichment systems"""
    print("\n📖 Usage Examples")
    print("=" * 40)
    
    print("1. Test with small sample:")
    print("   python test_enrichment.py")
    
    print("\n2. Run web enrichment on sample catalog:")
    print("   python web_lexile_enrichment.py --sample 10")
    
    print("\n3. Run on full catalog:")
    print("   python web_lexile_enrichment.py --catalog data/books_catalog.csv")
    
    print("\n4. Run LLM enrichment (requires LLM API setup):")
    print("   python llm_lexile_enrichment.py --sample 5")
    
    print("\n📁 Output files:")
    print("   - data/processed/web_enriched_lexile_scores.csv")
    print("   - data/processed/enriched_lexile_scores.csv")
    print("   - data/processed/web_enrichment_report.txt")

def main():
    """Main test function"""
    print("🚀 Lexile Enrichment System Test Suite")
    print("=" * 50)
    
    # Create sample catalog
    sample_path = create_sample_catalog()
    
    # Test Google Books API (most reliable)
    test_google_books_api()
    
    # Test full web enrichment
    test_web_enrichment()
    
    # Show usage examples
    show_usage_examples()
    
    print("\n🎉 Test suite completed!")
    print("\nNext steps:")
    print("1. Review the test results above")
    print("2. Run enrichment on your full catalog")
    print("3. Integrate enriched scores into your ML models")
    print("4. Set up automated enrichment for new books")

if __name__ == "__main__":
    main()