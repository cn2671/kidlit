import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

def add_missing_test_books():
    """Add missing test books to our tracking dataset"""
    
    # Load current tracking data
    tracking_path = DATA_DIR / "lexile_collection_tracking.csv"
    df = pd.read_csv(tracking_path)
    
    # Define missing test books with their age data
    missing_books = [
        {
            'title': "Tom's Midnight Garden",
            'author': 'Philippa Pearce',
            'current_ml_estimate': None,
            'ml_confidence': None,
            'search_query': None,
            'google_search_url': None,
            'verified_lexile': None,
            'lexile_source': None,
            'collection_date': None,
            'notes': "Ages 8-12, Grades 3-7, 1958 Carnegie Medal winner, classic children's literature",
            'status': 'Test_Book',
            'lexile_prefix': None,
            'lexile_numeric': None,
            'book_type': None,
            'predicted_lexile_age_model': None,
            'age_model_confidence': None,
            'prediction_method': None
        },
        {
            'title': 'The Invention of Hugo Cabret',
            'author': 'Brian Selznick',
            'current_ml_estimate': None,
            'ml_confidence': None,
            'search_query': None,
            'google_search_url': None,
            'verified_lexile': None,
            'lexile_source': None,
            'collection_date': None,
            'notes': "Ages 9-12, Grades 3-8, AR 5.1, 2008 Caldecott Medal winner, hybrid novel/graphic format",
            'status': 'Test_Book',
            'lexile_prefix': None,
            'lexile_numeric': None,
            'book_type': None,
            'predicted_lexile_age_model': None,
            'age_model_confidence': None,
            'prediction_method': None
        },
        {
            'title': 'Blueberries for Sal',
            'author': 'Robert McCloskey',
            'current_ml_estimate': None,
            'ml_confidence': None,
            'search_query': None,
            'google_search_url': None,
            'verified_lexile': None,
            'lexile_source': None,
            'collection_date': None,
            'notes': "Ages 2-7, PreK-Grade 2, Caldecott Honor Book, excellent read-aloud",
            'status': 'Test_Book',
            'lexile_prefix': None,
            'lexile_numeric': None,
            'book_type': None,
            'predicted_lexile_age_model': None,
            'age_model_confidence': None,
            'prediction_method': None
        },
        {
            'title': 'The Rainbow Fish',
            'author': 'Marcus Pfister',
            'current_ml_estimate': None,
            'ml_confidence': None,
            'search_query': None,
            'google_search_url': None,
            'verified_lexile': None,
            'lexile_source': None,
            'collection_date': None,
            'notes': "Ages 4-8, PreK-Grade 2, AR 3.3, picture book with moral themes",
            'status': 'Test_Book',
            'lexile_prefix': None,
            'lexile_numeric': None,
            'book_type': None,
            'predicted_lexile_age_model': None,
            'age_model_confidence': None,
            'prediction_method': None
        },
        {
            'title': 'The Very Busy Spider',
            'author': 'Eric Carle',
            'current_ml_estimate': None,
            'ml_confidence': None,
            'search_query': None,
            'google_search_url': None,
            'verified_lexile': None,
            'lexile_source': None,
            'collection_date': None,
            'notes': "Ages 2-5, PreK-K, multi-sensory book with textured illustrations",
            'status': 'Test_Book',
            'lexile_prefix': None,
            'lexile_numeric': None,
            'book_type': None,
            'predicted_lexile_age_model': None,
            'age_model_confidence': None,
            'prediction_method': None
        },
        {
            'title': 'Sideways Stories from Wayside School',
            'author': 'Louis Sachar',
            'current_ml_estimate': None,
            'ml_confidence': None,
            'search_query': None,
            'google_search_url': None,
            'verified_lexile': None,
            'lexile_source': None,
            'collection_date': None,
            'notes': "Ages 7-12, Grades 2-5, perfect for beginning chapter book readers",
            'status': 'Test_Book',
            'lexile_prefix': None,
            'lexile_numeric': None,
            'book_type': None,
            'predicted_lexile_age_model': None,
            'age_model_confidence': None,
            'prediction_method': None
        },
        {
            'title': 'The Indian in the Cupboard',
            'author': 'Lynne Reid Banks',
            'current_ml_estimate': None,
            'ml_confidence': None,
            'search_query': None,
            'google_search_url': None,
            'verified_lexile': None,
            'lexile_source': None,
            'collection_date': None,
            'notes': "Ages 8-12, Grades 3-5, Guided Reading Level R, 227 pages",
            'status': 'Test_Book',
            'lexile_prefix': None,
            'lexile_numeric': None,
            'book_type': None,
            'predicted_lexile_age_model': None,
            'age_model_confidence': None,
            'prediction_method': None
        },
        {
            'title': 'Peter Pan',
            'author': 'J.M. Barrie',
            'current_ml_estimate': None,
            'ml_confidence': None,
            'search_query': None,
            'google_search_url': None,
            'verified_lexile': None,
            'lexile_source': None,
            'collection_date': None,
            'notes': "Ages 8-10, Grades 1-6, AR 4.7, classic fantasy, 18,338 words",
            'status': 'Test_Book',
            'lexile_prefix': None,
            'lexile_numeric': None,
            'book_type': None,
            'predicted_lexile_age_model': None,
            'age_model_confidence': None,
            'prediction_method': None
        },
        {
            'title': 'The Bad Beginning',
            'author': 'Lemony Snicket',
            'current_ml_estimate': None,
            'ml_confidence': None,
            'search_query': None,
            'google_search_url': None,
            'verified_lexile': None,
            'lexile_source': None,
            'collection_date': None,
            'notes': "Ages 10-14, Grades 5-7, AR 6.4, Series of Unfortunate Events #1, sophisticated vocabulary",
            'status': 'Test_Book',
            'lexile_prefix': None,
            'lexile_numeric': None,
            'book_type': None,
            'predicted_lexile_age_model': None,
            'age_model_confidence': None,
            'prediction_method': None
        },
        {
            'title': 'Beezus and Ramona',
            'author': 'Beverly Cleary',
            'current_ml_estimate': None,
            'ml_confidence': None,
            'search_query': None,
            'google_search_url': None,
            'verified_lexile': None,
            'lexile_source': None,
            'collection_date': None,
            'notes': "Ages 8-12, Grades 3-5, AR 4.5, first book in Ramona series, 22,018 words",
            'status': 'Test_Book',
            'lexile_prefix': None,
            'lexile_numeric': None,
            'book_type': None,
            'predicted_lexile_age_model': None,
            'age_model_confidence': None,
            'prediction_method': None
        },
        {
            'title': 'Sadako and the Thousand Paper Cranes',
            'author': 'Eleanor Coerr',
            'current_ml_estimate': None,
            'ml_confidence': None,
            'search_query': None,
            'google_search_url': None,
            'verified_lexile': None,
            'lexile_source': None,
            'collection_date': None,
            'notes': "Ages 8-12, Grades 3-6, historical fiction, sensitive themes",
            'status': 'Test_Book',
            'lexile_prefix': None,
            'lexile_numeric': None,
            'book_type': None,
            'predicted_lexile_age_model': None,
            'age_model_confidence': None,
            'prediction_method': None
        },
        {
            'title': 'The Westing Game',
            'author': 'Ellen Raskin',
            'current_ml_estimate': None,
            'ml_confidence': None,
            'search_query': None,
            'google_search_url': None,
            'verified_lexile': None,
            'lexile_source': None,
            'collection_date': None,
            'notes': "Ages 8-14, Grades 3-7, AR 5.3, 1979 Newbery Medal winner, mystery novel",
            'status': 'Test_Book',
            'lexile_prefix': None,
            'lexile_numeric': None,
            'book_type': None,
            'predicted_lexile_age_model': None,
            'age_model_confidence': None,
            'prediction_method': None
        },
        {
            'title': 'Lyle, Lyle, Crocodile',
            'author': 'Bernard Waber',
            'current_ml_estimate': None,
            'ml_confidence': None,
            'search_query': None,
            'google_search_url': None,
            'verified_lexile': None,
            'lexile_source': None,
            'collection_date': None,
            'notes': "Ages 4-8, K-2, AR 1.8, picture book series, sight word practice",
            'status': 'Test_Book',
            'lexile_prefix': None,
            'lexile_numeric': None,
            'book_type': None,
            'predicted_lexile_age_model': None,
            'age_model_confidence': None,
            'prediction_method': None
        },
        {
            'title': 'Little House in the Big Woods',
            'author': 'Laura Ingalls Wilder',
            'current_ml_estimate': None,
            'ml_confidence': None,
            'search_query': None,
            'google_search_url': None,
            'verified_lexile': None,
            'lexile_source': None,
            'collection_date': None,
            'notes': "Ages 7-11, Grade 2+ literature, AR 5.3, historical fiction, 33,586 words",
            'status': 'Test_Book',
            'lexile_prefix': None,
            'lexile_numeric': None,
            'book_type': None,
            'predicted_lexile_age_model': None,
            'age_model_confidence': None,
            'prediction_method': None
        },
        {
            'title': 'Anne of Ingleside',
            'author': 'L.M. Montgomery',
            'current_ml_estimate': None,
            'ml_confidence': None,
            'search_query': None,
            'google_search_url': None,
            'verified_lexile': None,
            'lexile_source': None,
            'collection_date': None,
            'notes': "Ages 12-16, Grades 6-8, 6th book in Anne series, mature themes",
            'status': 'Test_Book',
            'lexile_prefix': None,
            'lexile_numeric': None,
            'book_type': None,
            'predicted_lexile_age_model': None,
            'age_model_confidence': None,
            'prediction_method': None
        }
    ]
    
    print(f"📚 ADDING MISSING TEST BOOKS TO TRACKING DATASET")
    print(f"{'='*55}")
    print(f"Current dataset size: {len(df)} books")
    print(f"Missing test books to add: {len(missing_books)}")
    
    # Check which books are actually missing
    books_to_add = []
    for book in missing_books:
        title = book['title']
        # Check if book already exists (flexible matching)
        exists = False
        for _, existing_row in df.iterrows():
            if title.lower() in existing_row['title'].lower() or existing_row['title'].lower() in title.lower():
                exists = True
                print(f"  ✅ Already exists: {title} (as {existing_row['title']})")
                break
        
        if not exists:
            books_to_add.append(book)
            print(f"  ➕ Will add: {title}")
    
    if len(books_to_add) > 0:
        # Add missing books to dataset
        new_books_df = pd.DataFrame(books_to_add)
        updated_df = pd.concat([df, new_books_df], ignore_index=True)
        
        # Save updated dataset
        updated_df.to_csv(tracking_path, index=False)
        
        print(f"\n📈 RESULTS:")
        print(f"  Books added: {len(books_to_add)}")
        print(f"  New dataset size: {len(updated_df)} books")
        print(f"💾 Updated tracking dataset saved")
        
        return updated_df, books_to_add
    else:
        print(f"\n✅ All test books already exist in dataset")
        return df, []

def create_comprehensive_test_cases():
    """Create comprehensive test cases from all sources"""
    
    # Load our test cases
    original_test_path = DATA_DIR / "model_test_cases.csv"
    original_test_df = pd.read_csv(original_test_path)
    
    # Load tracking data to get additional verified scores
    tracking_path = DATA_DIR / "lexile_collection_tracking.csv"
    tracking_df = pd.read_csv(tracking_path)
    
    print(f"\n🧪 CREATING COMPREHENSIVE TEST CASES")
    print(f"{'='*45}")
    print(f"Original test cases: {len(original_test_df)}")
    
    # Get all books with verified Lexile scores from tracking
    verified_books = tracking_df[tracking_df['status'] == 'Complete'].copy()
    print(f"Verified books in tracking: {len(verified_books)}")
    
    # Convert verified books to test case format
    additional_test_cases = []
    
    for _, row in verified_books.iterrows():
        lexile_str = str(row['verified_lexile'])
        
        # Parse lexile with prefix
        if pd.notna(row['lexile_prefix']) and row['lexile_prefix'] != '':
            book_type = 'Adult_Directed' if row['lexile_prefix'] == 'AD' else 'Graphic_Novel'
            lexile_numeric = row['lexile_numeric']
            lexile_prefix = row['lexile_prefix']
        else:
            book_type = 'Standard_Lexile'
            lexile_numeric = row['lexile_numeric']
            lexile_prefix = ''
        
        test_case = {
            'title': row['title'],
            'verified_lexile': lexile_str,
            'lexile_prefix': lexile_prefix,
            'lexile_numeric': lexile_numeric,
            'book_type': book_type
        }
        additional_test_cases.append(test_case)
    
    # Combine all test cases (avoiding duplicates)
    additional_test_df = pd.DataFrame(additional_test_cases)
    
    # Remove duplicates by title (keep original test cases)
    original_titles = set(original_test_df['title'].str.lower())
    additional_test_df = additional_test_df[~additional_test_df['title'].str.lower().isin(original_titles)]
    
    # Combine
    comprehensive_test_df = pd.concat([original_test_df, additional_test_df], ignore_index=True)
    
    # Save comprehensive test cases
    comprehensive_test_path = DATA_DIR / "comprehensive_model_test_cases.csv"
    comprehensive_test_df.to_csv(comprehensive_test_path, index=False)
    
    print(f"Additional test cases from verified books: {len(additional_test_df)}")
    print(f"Comprehensive test cases: {len(comprehensive_test_df)}")
    print(f"💾 Comprehensive test cases saved: {comprehensive_test_path}")
    
    return comprehensive_test_df

def main():
    # Add missing test books
    updated_df, added_books = add_missing_test_books()
    
    # Create comprehensive test cases
    comprehensive_test_df = create_comprehensive_test_cases()
    
    print(f"\n🎯 SUMMARY:")
    print(f"  ✅ Dataset now contains {len(updated_df)} books")
    print(f"  ✅ Added {len(added_books)} missing test books")
    print(f"  ✅ Created comprehensive test set with {len(comprehensive_test_df)} books")
    print(f"  ✅ Ready for expanded model testing!")
    
    return updated_df, comprehensive_test_df

if __name__ == "__main__":
    main()