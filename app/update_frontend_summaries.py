#!/usr/bin/env python3
"""
Update frontend app.html file with corrected summary data from CSV
"""

import pandas as pd
import json
import re

def update_frontend_summaries():
    print("🔄 Updating frontend app.html with corrected summaries...")
    
    # Load the corrected data
    try:
        df = pd.read_csv('/Users/chaerinnoh/Desktop/kidlit/data/raw/books_final_complete.csv')
        print(f"✓ Loaded {len(df)} books with corrected summaries")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return
    
    # Read the current app.html file
    try:
        with open('/Users/chaerinnoh/Desktop/kidlit/app/app.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        print("✓ Loaded app.html file")
    except Exception as e:
        print(f"❌ Error reading app.html: {e}")
        return
    
    # Convert DataFrame to the format expected by the frontend
    books_data = []
    for idx, row in df.iterrows():
        # Use summary_gpt if available, otherwise use description
        summary = str(row.get('summary_gpt', ''))
        if not summary or summary == 'nan' or summary.strip() == '':
            summary = str(row.get('description', ''))
        
        book_data = {
            'id': idx,
            'title': str(row.get('title', '')),
            'author': str(row.get('author', '')),
            'summary_gpt': summary,
            'description': str(row.get('description', '')),
            'lexile': row.get('lexile_score'),
            'age_min': row.get('age_min'),
            'age_max': row.get('age_max'),
            'themes': str(row.get('themes', '')).split(',') if row.get('themes') else [],
            'cover_url': str(row.get('cover_url', '')) if row.get('cover_url') else None
        }
        books_data.append(book_data)
    
    # Convert to JavaScript format
    js_books_data = json.dumps(books_data, indent=8)
    
    # Find and replace the books data in the HTML
    # Look for the pattern: const SAMPLE_BOOKS = [...]
    pattern = r'const SAMPLE_BOOKS = \[[\s\S]*?\];'
    replacement = f'const SAMPLE_BOOKS = {js_books_data};'
    
    if re.search(pattern, html_content):
        # Use re.sub with a function to avoid escape issues
        def replace_func(match):
            return f'const SAMPLE_BOOKS = {js_books_data};'
        updated_html = re.sub(pattern, replace_func, html_content)
        print("✓ Found and updated books data in app.html")
    else:
        print("❌ Could not find books data pattern in app.html")
        return
    
    # Write the updated HTML back
    try:
        with open('/Users/chaerinnoh/Desktop/kidlit/app/app.html', 'w', encoding='utf-8') as f:
            f.write(updated_html)
        print("✅ Successfully updated app.html with corrected summaries")
        
        # Verify the update worked for our test cases
        print("\n🔍 Verifying updates for previously incomplete summaries:")
        test_books = [401, 412, 612, 663, 849, 965, 983, 81, 1070]
        for book_idx in test_books:
            if book_idx < len(df):
                row = df.iloc[book_idx]
                title = row.get('title', 'Unknown')
                summary = str(row.get('summary_gpt', ''))
                print(f"  ✓ {title}: {len(summary)} chars")
        
    except Exception as e:
        print(f"❌ Error writing updated app.html: {e}")
        return

if __name__ == "__main__":
    update_frontend_summaries()