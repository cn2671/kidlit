#!/usr/bin/env python3
"""
Check the 9 books that previously had incomplete summaries
"""

import pandas as pd

def check_fixed_books():
    # Load the catalog
    df = pd.read_csv('/Users/chaerinnoh/Desktop/kidlit/data/raw/books_final_complete.csv')
    
    # The 9 books that were manually fixed
    book_rows = [401, 412, 612, 663, 849, 965, 983, 81, 1070]
    
    print('📖 Checking the 9 previously incomplete summaries:\n')
    
    for row_idx in book_rows:
        if row_idx < len(df):
            row = df.iloc[row_idx]
            title = row.get('title', 'Unknown')
            summary = str(row.get('summary_gpt', ''))
            
            print(f'Row {row_idx}: "{title}"')
            print(f'Summary: {summary}')
            print(f'Length: {len(summary)} chars')
            print(f'Ends properly: {summary.strip().endswith((".", "!", "?"))}')
            print()
        else:
            print(f'Row {row_idx}: NOT FOUND')
            print()

if __name__ == "__main__":
    check_fixed_books()