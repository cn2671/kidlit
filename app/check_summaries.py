#!/usr/bin/env python3
"""
Check all book summaries for completeness
"""

import pandas as pd
import re

def check_summary_completeness():
    print("🔍 Checking book summaries for completeness...")
    
    # Load the catalog
    try:
        df = pd.read_csv('/Users/chaerinnoh/Desktop/kidlit/data/raw/books_final_complete.csv')
        print(f"✓ Loaded catalog with {len(df)} books")
    except Exception as e:
        print(f"❌ Error loading catalog: {e}")
        return
    
    # Check for potentially incomplete summaries
    incomplete_summaries = []
    short_summaries = []
    missing_summaries = []
    
    for idx, row in df.iterrows():
        title = row.get('title', 'Unknown')
        
        # Check both summary_gpt and description fields
        summary_gpt = str(row.get('summary_gpt', ''))
        description = str(row.get('description', ''))
        
        # Use summary_gpt if available, otherwise use description
        summary = summary_gpt if summary_gpt and summary_gpt != 'nan' and summary_gpt.strip() else description
        
        # Check for missing summaries
        if not summary or summary == 'nan' or summary.strip() == '':
            missing_summaries.append({
                'index': idx,
                'title': title,
                'issue': 'Missing both summary_gpt and description'
            })
            continue
        
        # Check for summaries that might be cut off
        # Look for common patterns of incomplete text
        incomplete_patterns = [
            r'\.\.\.$',  # ends with ...
            r'[a-z]$',   # ends mid-word (lowercase letter, no punctuation)
            r'\s+$',     # ends with whitespace
            r'[,;:]$',   # ends with comma, semicolon, or colon
            r'\band\s*$', # ends with "and"
            r'\bor\s*$',  # ends with "or"
            r'\bthe\s*$', # ends with "the"
            r'\bin\s*$',  # ends with "in"
            r'\bto\s*$',  # ends with "to"
            r'\bof\s*$',  # ends with "of"
            r'\bwhen\s*$', # ends with "when"
            r'\bwho\s*$',  # ends with "who"
            r'\bwhat\s*$', # ends with "what"
            r'\bwhere\s*$', # ends with "where"
            r'\bwhy\s*$',  # ends with "why"
            r'\bhow\s*$',  # ends with "how"
            r'\bwill\s*$', # ends with "will"
            r'\bcan\s*$',  # ends with "can"
        ]
        
        is_incomplete = False
        for pattern in incomplete_patterns:
            if re.search(pattern, summary, re.IGNORECASE):
                is_incomplete = True
                break
        
        if is_incomplete:
            incomplete_summaries.append({
                'index': idx,
                'title': title,
                'summary': summary,
                'issue': 'Potentially incomplete (suspicious ending)'
            })
        
        # Check for very short summaries (likely incomplete)
        if len(summary.strip()) < 50:
            short_summaries.append({
                'index': idx,
                'title': title,
                'summary': summary,
                'issue': f'Very short summary ({len(summary.strip())} chars)'
            })
    
    # Report findings
    print(f"\n📊 Summary Analysis Results:")
    print(f"Total books: {len(df)}")
    print(f"Missing summaries: {len(missing_summaries)}")
    print(f"Potentially incomplete summaries: {len(incomplete_summaries)}")
    print(f"Very short summaries: {len(short_summaries)}")
    
    # Show examples of issues
    if missing_summaries:
        print(f"\n❌ Books with missing summaries:")
        for item in missing_summaries[:10]:  # Show first 10
            print(f"  - \"{item['title']}\" (row {item['index']})")
        if len(missing_summaries) > 10:
            print(f"  ... and {len(missing_summaries) - 10} more")
    
    if incomplete_summaries:
        print(f"\n⚠️ Books with potentially incomplete summaries:")
        for item in incomplete_summaries:  # Show all
            print(f"  - \"{item['title']}\" (row {item['index']})")
            print(f"    Summary: \"{item['summary']}\"")
            print()
    
    if short_summaries:
        print(f"\n📏 Books with very short summaries:")
        for item in short_summaries:  # Show all
            print(f"  - \"{item['title']}\" (row {item['index']})")
            print(f"    Summary: \"{item['summary']}\"")
            print()
    
    if not missing_summaries and not incomplete_summaries and not short_summaries:
        print(f"\n✅ All summaries appear to be complete!")
    
    return {
        'missing': missing_summaries,
        'incomplete': incomplete_summaries,
        'short': short_summaries
    }

if __name__ == "__main__":
    check_summary_completeness()