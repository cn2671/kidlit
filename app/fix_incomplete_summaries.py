#!/usr/bin/env python3
"""
Fix incomplete book summaries by using full descriptions
"""

import pandas as pd
import re

def fix_incomplete_summaries():
    print("🔧 Fixing incomplete book summaries...")
    
    # Load the catalog
    try:
        df = pd.read_csv('/Users/chaerinnoh/Desktop/kidlit/data/raw/books_final_complete.csv')
        print(f"✓ Loaded catalog with {len(df)} books")
    except Exception as e:
        print(f"❌ Error loading catalog: {e}")
        return
    
    # Identify books with incomplete summaries
    fixes_made = []
    
    # Incomplete patterns to identify truncated summaries
    incomplete_patterns = [
        r'\.\.\.$',  # ends with ...
        r'\bin\s*$',  # ends with "in"
        r'\bthe\s*$', # ends with "the"
        r'\bto\s*$',  # ends with "to"
        r'\bof\s*$',  # ends with "of"
        r'\bwho\s*$',  # ends with "who"
        r'\bwhen\s*$', # ends with "when"
        r'\bwhat\s*$', # ends with "what"
        r'\bwhere\s*$', # ends with "where"
        r'\bwhy\s*$',  # ends with "why"
        r'\bhow\s*$',  # ends with "how"
        r'\bwill\s*$', # ends with "will"
        r'\bcan\s*$',  # ends with "can"
        r'\band\s*$', # ends with "and"
        r'\bor\s*$',  # ends with "or"
        r'[a-z]$',   # ends mid-word (lowercase letter, no punctuation)
    ]
    
    for idx, row in df.iterrows():
        title = row.get('title', 'Unknown')
        summary_gpt = str(row.get('summary_gpt', ''))
        description = str(row.get('description', ''))
        
        # Skip if no summary_gpt
        if not summary_gpt or summary_gpt == 'nan' or summary_gpt.strip() == '':
            continue
        
        # Check if summary is incomplete
        is_incomplete = False
        for pattern in incomplete_patterns:
            if re.search(pattern, summary_gpt, re.IGNORECASE):
                is_incomplete = True
                break
        
        # Also check if summary is very short (under 50 chars)
        if len(summary_gpt.strip()) < 50:
            is_incomplete = True
        
        if is_incomplete:
            # Use description if it's longer and looks complete
            if (description and description != 'nan' and len(description.strip()) > len(summary_gpt.strip()) and 
                description.strip().endswith(('.', '!', '?', '"'))):
                
                # Clean up the description
                cleaned_description = description.strip()
                
                # If it's much longer, create a concise summary from the first few sentences
                if len(cleaned_description) > 300:
                    sentences = re.split(r'[.!?]+', cleaned_description)
                    # Take first 2-3 sentences that make sense
                    summary_sentences = []
                    char_count = 0
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if sentence and char_count + len(sentence) < 250:
                            summary_sentences.append(sentence)
                            char_count += len(sentence)
                            if len(summary_sentences) >= 2:
                                break
                    
                    if summary_sentences:
                        new_summary = '. '.join(summary_sentences) + '.'
                    else:
                        new_summary = cleaned_description[:250] + '...'
                else:
                    new_summary = cleaned_description
                
                # Update the dataframe
                df.at[idx, 'summary_gpt'] = new_summary
                
                fixes_made.append({
                    'title': title,
                    'old_summary': summary_gpt,
                    'new_summary': new_summary
                })
                
                print(f"✓ Fixed: \"{title}\"")
                print(f"  Old: {summary_gpt[:80]}...")
                print(f"  New: {new_summary[:80]}...")
                print()
    
    # Save the updated file
    if fixes_made:
        output_file = '/Users/chaerinnoh/Desktop/kidlit/data/raw/books_final_complete.csv'
        df.to_csv(output_file, index=False)
        print(f"💾 Saved {len(fixes_made)} fixes to {output_file}")
        
        print(f"\n📋 Summary of fixes:")
        for fix in fixes_made:
            print(f"  - \"{fix['title']}\"")
    else:
        print("✅ No incomplete summaries found or all descriptions were also incomplete")

if __name__ == "__main__":
    fix_incomplete_summaries()