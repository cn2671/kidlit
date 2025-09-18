#!/usr/bin/env python3
"""
Debug script to check why Magic + gentle shows no results
"""

import pandas as pd
import sys
import os

# Add the parent directory to the Python path to import the backend modules
sys.path.append('/Users/chaerinnoh/Desktop/kidlit/app')

def debug_magic_gentle_combination():
    print("🔍 Debugging Magic + gentle combination...")
    
    # Load the catalog
    try:
        df = pd.read_csv('/Users/chaerinnoh/Desktop/kidlit/data/raw/books_final_complete.csv')
        print(f"✓ Loaded catalog with {len(df)} books")
    except Exception as e:
        print(f"❌ Error loading catalog: {e}")
        return
    
    # Check books with magic theme
    magic_mask = df['themes'].str.contains('magic', case=False, na=False, regex=False)
    magic_books = df[magic_mask]
    print(f"📚 Books with 'magic' theme: {len(magic_books)}")
    
    # Check books with gentle tone (in themes or tone columns)
    gentle_theme_mask = df['themes'].str.contains('gentle', case=False, na=False, regex=False)
    gentle_tone_mask = df['tone'].str.contains('gentle', case=False, na=False, regex=False)
    gentle_mask = gentle_theme_mask | gentle_tone_mask
    gentle_books = df[gentle_mask]
    print(f"📚 Books with 'gentle' (in themes or tone): {len(gentle_books)}")
    
    # Check books with valid age data
    valid_age_mask = df['age_range_llm'].notna() & (df['age_range_llm'] != '')
    books_with_age = df[valid_age_mask]
    print(f"📚 Books with valid age data: {len(books_with_age)}")
    
    # Check combination: magic theme + gentle tone + valid age
    combination_mask = magic_mask & gentle_mask & valid_age_mask
    combination_books = df[combination_mask]
    print(f"📚 Books with magic + gentle + valid age: {len(combination_books)}")
    
    if len(combination_books) > 0:
        print("\n📖 Sample books that match both criteria:")
        for _, book in combination_books.head(5).iterrows():
            print(f"  - \"{book['title']}\" by {book['author']}")
            print(f"    Themes: {book['themes']}")
            print(f"    Tone: {book['tone']}")
            print(f"    Age: {book['age_range_llm']}")
            print()
    else:
        print("\n❌ No books found with both magic theme and gentle tone!")
        
        # Let's check what happens step by step
        print("\n🔍 Detailed Analysis:")
        
        # Check magic books with valid age data
        magic_with_age = df[magic_mask & valid_age_mask]
        print(f"  Magic books with valid age: {len(magic_with_age)}")
        
        # Check gentle books with valid age data  
        gentle_with_age = df[gentle_mask & valid_age_mask]
        print(f"  Gentle books with valid age: {len(gentle_with_age)}")
        
        # Show sample magic books
        print(f"\n📖 Sample magic books:")
        for _, book in magic_with_age.head(3).iterrows():
            print(f"  - \"{book['title']}\" - Themes: {book['themes']}")
            print(f"    Tone: {book['tone']}")
            print(f"    Age: {book['age_range_llm']}")
        
        # Show sample gentle books
        print(f"\n📖 Sample gentle books:")
        for _, book in gentle_with_age.head(3).iterrows():
            print(f"  - \"{book['title']}\" - Themes: {book['themes']}")
            print(f"    Tone: {book['tone']}")
            print(f"    Age: {book['age_range_llm']}")

if __name__ == "__main__":
    debug_magic_gentle_combination()