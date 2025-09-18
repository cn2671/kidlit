#!/usr/bin/env python3
"""
Verify all statistics shown in the "How It Works" sections
"""

import pandas as pd

def verify_all_statistics():
    """Check all statistics mentioned in the app"""
    
    try:
        df = pd.read_csv('/Users/chaerinnoh/Desktop/kidlit/data/raw/books_final_complete.csv')
        total_books = len(df)
        print(f"📚 CURRENT DATABASE STATISTICS")
        print("=" * 50)
        print(f"Total Books: {total_books}")
        
        # Authors count
        if 'author' in df.columns:
            unique_authors = df['author'].nunique()
            print(f"Unique Authors: {unique_authors}")
        
        # Lexile analysis
        if 'reading_level_estimate' in df.columns:
            # Count books with enriched/actual lexile scores
            enriched_count = df['reading_level_estimate'].notna().sum()
            enriched_percentage = (enriched_count / total_books) * 100
            print(f"Books with Lexile Estimates: {enriched_count} ({enriched_percentage:.1f}%)")
        
        # ML Analysis coverage
        if 'reading_level_llm' in df.columns:
            ml_coverage = df['reading_level_llm'].notna().sum()
            ml_percentage = (ml_coverage / total_books) * 100
            print(f"ML Analysis Coverage: {ml_coverage} books ({ml_percentage:.1f}%)")
        
        # Check for lexile sources
        if 'reading_level_sources' in df.columns:
            # Count books that have external lexile sources (not just ML predictions)
            external_sources = df[df['reading_level_sources'].notna() & 
                                (df['reading_level_sources'].astype(str) != '') &
                                (df['reading_level_sources'].astype(str) != 'nan')]['reading_level_sources'].count()
            print(f"Books with External Lexile Sources: {external_sources}")
        
        # Enhanced matching (books using enriched vs estimated scores)
        if 'has_level_data' in df.columns:
            has_real_data = df['has_level_data'].sum() if df['has_level_data'].dtype == bool else df[df['has_level_data'] == True].shape[0]
            print(f"Books with Real Lexile Data: {has_real_data}")
            enhanced_books = total_books - has_real_data
            print(f"Books Using ML-Enhanced Scores: {enhanced_books}")
        
        print("\n" + "=" * 50)
        print("🔍 SPECIFIC STATISTICS TO VERIFY:")
        print("=" * 50)
        
        # These are the numbers from the screenshots that we need to verify
        claims = {
            "555 authors": unique_authors if 'unique_authors' in locals() else "Unknown",
            "632 enriched lexile scores": external_sources if 'external_sources' in locals() else "Unknown", 
            "320 books enhanced": enhanced_books if 'enhanced_books' in locals() else "Unknown",
            "1,087 total books": total_books
        }
        
        for claim, actual in claims.items():
            status = "✅ CORRECT" if str(actual) in claim else "❌ NEEDS UPDATE"
            print(f"{claim}: {actual} - {status}")
        
        print("\n" + "=" * 50)
        print("📊 ADDITIONAL ANALYSIS:")
        print("=" * 50)
        
        # Check themes and tones coverage
        if 'themes' in df.columns:
            themes_coverage = df['themes'].notna().sum()
            themes_percentage = (themes_coverage / total_books) * 100
            print(f"Theme Coverage: {themes_coverage} books ({themes_percentage:.1f}%)")
        
        if 'tone' in df.columns:
            tone_coverage = df['tone'].notna().sum()
            tone_percentage = (tone_coverage / total_books) * 100
            print(f"Tone Coverage: {tone_coverage} books ({tone_percentage:.1f}%)")
        
        # Print column names for debugging
        print(f"\nAvailable columns: {list(df.columns)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    verify_all_statistics()