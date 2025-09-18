#!/usr/bin/env python3
"""
Check Age Range Distribution to understand what "Other" includes
"""

import pandas as pd

def check_age_distribution():
    """Check actual age range distribution in the catalog"""
    
    # Load the catalog
    try:
        df = pd.read_csv('/Users/chaerinnoh/Desktop/kidlit/data/raw/books_final_complete.csv')
        total_books = len(df)
        print(f"📚 Total Books: {total_books}")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error loading catalog: {e}")
        return
    
    # Get age range distribution
    if 'age_range_llm' in df.columns:
        age_counts = df['age_range_llm'].value_counts().sort_index()
        print("📊 FULL AGE RANGE DISTRIBUTION:")
        print("-" * 40)
        
        # Define the main categories shown in chart
        main_categories = {
            '0-2': ['0-2'],
            '3-5': ['3-5'],
            '6-8': ['6-8'],
            '9-12': ['9-12'],
            '13+': ['13+', '13-17', '14+', '15+', '16+', '17+', '18+']
        }
        
        main_total = 0
        other_ranges = []
        
        for age_range, count in age_counts.items():
            percentage = (count / total_books) * 100
            print(f"{age_range}: {count} books ({percentage:.1f}%)")
            
            # Check if this age range is in main categories
            found_in_main = False
            for category, ranges in main_categories.items():
                if age_range in ranges:
                    main_total += count
                    found_in_main = True
                    break
            
            if not found_in_main:
                other_ranges.append((age_range, count, percentage))
        
        print("\n" + "=" * 60)
        print("📋 CATEGORIZATION FOR CHART:")
        print("-" * 40)
        
        # Calculate main categories
        for category, ranges in main_categories.items():
            category_count = 0
            for age_range in ranges:
                if age_range in age_counts:
                    category_count += age_counts[age_range]
            
            if category_count > 0:
                percentage = (category_count / total_books) * 100
                print(f"{category}: {category_count} books ({percentage:.1f}%)")
        
        # Calculate "Other" category
        other_total = sum(count for _, count, _ in other_ranges)
        other_percentage = (other_total / total_books) * 100
        print(f"Other: {other_total} books ({other_percentage:.1f}%)")
        
        if other_ranges:
            print("\n🔍 'OTHER' CATEGORY INCLUDES:")
            print("-" * 30)
            for age_range, count, percentage in other_ranges:
                print(f"  • {age_range}: {count} books ({percentage:.1f}%)")
        
        print(f"\nTotal accounted for: {main_total + other_total} / {total_books} books")
        
    else:
        print("❌ No age_range_llm column found")

if __name__ == "__main__":
    check_age_distribution()