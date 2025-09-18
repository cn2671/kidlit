#!/usr/bin/env python3
"""
Data Completeness Analysis for KidLit Catalog
Analyzes how many books have data for each metadata field
"""

import pandas as pd

def analyze_data_completeness():
    """Analyze completeness of all metadata fields in the catalog"""
    
    # Load the catalog
    try:
        df = pd.read_csv('/Users/chaerinnoh/Desktop/kidlit/data/raw/books_final_complete.csv')
        total_books = len(df)
        print(f"📚 Total Books in Catalog: {total_books}")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error loading catalog: {e}")
        return
    
    # Define fields to analyze (using actual column names)
    fields_to_analyze = {
        'title': 'Title',
        'author': 'Author', 
        'summary_gpt': 'Summary',
        'themes': 'Themes',
        'tone': 'Tones',
        'age_range_llm': 'Age Range',
        'reading_level_llm': 'Reading Level (LLM)',
        'reading_level_estimate': 'Reading Level (Estimate)',
        'description': 'Original Description',
        'goodreads_url': 'Goodreads URL',
        'cover_url': 'Cover Image URL',
        'openlibrary_url': 'OpenLibrary URL',
        'reading_level_sources': 'Reading Level Sources',
        'has_level_data': 'Has Level Data',
        'complexity_analysis': 'Complexity Analysis',
        'reading_confidence_llm': 'Reading Confidence'
    }
    
    results = []
    
    for field, display_name in fields_to_analyze.items():
        if field in df.columns:
            # Count non-null, non-empty values
            non_null = df[field].notna()
            non_empty = df[field].astype(str).str.strip() != ''
            valid_data = non_null & non_empty
            count_with_data = valid_data.sum()
            percentage = (count_with_data / total_books) * 100
            
            results.append({
                'field': display_name,
                'count': count_with_data,
                'percentage': percentage,
                'missing': total_books - count_with_data
            })
        else:
            results.append({
                'field': display_name,
                'count': 0,
                'percentage': 0.0,
                'missing': total_books
            })
    
    # Sort by completeness (highest first)
    results.sort(key=lambda x: x['count'], reverse=True)
    
    # Display results
    print("📊 DATA COMPLETENESS ANALYSIS")
    print("=" * 60)
    print(f"{'Field':<25} {'Count':<8} {'%':<8} {'Missing':<8}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['field']:<25} {result['count']:<8} {result['percentage']:<7.1f}% {result['missing']:<8}")
    
    print("\n" + "=" * 60)
    
    # Summary statistics
    complete_fields = [r for r in results if r['percentage'] == 100.0]
    mostly_complete = [r for r in results if 80.0 <= r['percentage'] < 100.0]
    partial_data = [r for r in results if 20.0 <= r['percentage'] < 80.0]
    sparse_data = [r for r in results if 0.0 < r['percentage'] < 20.0]
    no_data = [r for r in results if r['percentage'] == 0.0]
    
    print("📈 SUMMARY BY COMPLETENESS:")
    print(f"  🟢 Complete (100%): {len(complete_fields)} fields")
    if complete_fields:
        for field in complete_fields:
            print(f"     - {field['field']}")
    
    print(f"  🟡 Mostly Complete (80-99%): {len(mostly_complete)} fields")
    if mostly_complete:
        for field in mostly_complete:
            print(f"     - {field['field']}: {field['count']} books ({field['percentage']:.1f}%)")
    
    print(f"  🟠 Partial Data (20-79%): {len(partial_data)} fields")
    if partial_data:
        for field in partial_data:
            print(f"     - {field['field']}: {field['count']} books ({field['percentage']:.1f}%)")
    
    print(f"  🔴 Sparse Data (<20%): {len(sparse_data)} fields")
    if sparse_data:
        for field in sparse_data:
            print(f"     - {field['field']}: {field['count']} books ({field['percentage']:.1f}%)")
    
    print(f"  ⚫ No Data (0%): {len(no_data)} fields")
    if no_data:
        for field in no_data:
            print(f"     - {field['field']}")
    
    # Special analysis for themes and tones (multi-value fields)
    print("\n" + "=" * 60)
    print("🎯 SPECIAL ANALYSIS FOR THEMES & TONES:")
    
    # Themes analysis
    if 'themes' in df.columns:
        themes_with_data = df['themes'].notna() & (df['themes'].astype(str).str.strip() != '')
        themes_count = themes_with_data.sum()
        
        # Count unique themes
        all_themes = []
        for themes_str in df[df['themes'].notna()]['themes']:
            if pd.notna(themes_str) and str(themes_str).strip():
                themes = [theme.strip() for theme in str(themes_str).split(',')]
                all_themes.extend(themes)
        
        unique_themes = len(set(all_themes))
        print(f"  📚 Themes: {themes_count} books have themes ({themes_count/total_books*100:.1f}%)")
        print(f"      - {unique_themes} unique themes identified")
    
    # Tones analysis 
    if 'tone' in df.columns:
        tones_with_data = df['tone'].notna() & (df['tone'].astype(str).str.strip() != '')
        tones_count = tones_with_data.sum()
        
        # Count unique tones
        all_tones = []
        for tone_str in df[df['tone'].notna()]['tone']:
            if pd.notna(tone_str) and str(tone_str).strip():
                tones = [tone.strip() for tone in str(tone_str).split(',')]
                all_tones.extend(tones)
        
        unique_tones = len(set(all_tones))
        print(f"  🎵 Tones: {tones_count} books have tones ({tones_count/total_books*100:.1f}%)")
        print(f"      - {unique_tones} unique tones identified")
    
    # Age ranges analysis
    if 'age_range_llm' in df.columns:
        age_with_data = df['age_range_llm'].notna() & (df['age_range_llm'].astype(str).str.strip() != '')
        age_count = age_with_data.sum()
        
        unique_age_ranges = df[df['age_range_llm'].notna()]['age_range_llm'].nunique()
        print(f"  👶 Age Ranges: {age_count} books have age ranges ({age_count/total_books*100:.1f}%)")
        print(f"      - {unique_age_ranges} unique age ranges")
    
    print("\n" + "=" * 60)
    print("✅ Analysis Complete!")

if __name__ == "__main__":
    analyze_data_completeness()