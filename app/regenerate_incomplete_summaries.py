#!/usr/bin/env python3
"""
Re-generate incomplete book summaries using robust prompts with validation
"""

import pandas as pd
import re
import openai
import os
from typing import Optional

# Books that need summary regeneration (from our analysis)
BOOKS_TO_FIX = [
    {"row": 401, "title": "Pigs"},
    {"row": 412, "title": "The Tale of Mrs. Tittlemouse"},
    {"row": 612, "title": "The Two Towers (The Lord of the Rings, #2)"},
    {"row": 663, "title": "Escape to Witch Mountain"},
    {"row": 849, "title": "All Creatures Great and Small (All Creatures Great and Small, #1-2)"},
    {"row": 965, "title": "Frog and Toad Are Friends (Frog and Toad, #1)"},
    {"row": 983, "title": "And to Think That I Saw It on Mulberry Street"},
    {"row": 81, "title": "Where the Red Fern Grows"},
    {"row": 1070, "title": "The Giraffe and the Pelly and Me"}
]

def create_robust_prompt(title: str, author: str, description: str) -> str:
    """Create a robust prompt with explicit completion requirements"""
    return f"""Create a children's book summary with these strict requirements:

BOOK: {title}
AUTHOR: {author}
DESCRIPTION: {description}

SUMMARY REQUIREMENTS:
- Write exactly 2 complete sentences
- First sentence: Who is the main character and what happens to them?
- Second sentence: What do they learn, achieve, or how does the story end?
- Each sentence MUST end with a period (.), exclamation point (!), or question mark (?)
- Total length: 50-150 words
- Use engaging language suitable for parents selecting books for their children
- Ensure both sentences are grammatically complete and make sense

SUMMARY:"""

def validate_summary(summary: str) -> tuple[bool, str]:
    """Validate that the summary meets our requirements"""
    if not summary or len(summary.strip()) < 10:
        return False, "Summary too short"
    
    # Check for proper ending punctuation
    if not summary.strip().endswith(('.', '!', '?')):
        return False, "Summary doesn't end with proper punctuation"
    
    # Check sentence count (split by sentence-ending punctuation)
    sentences = re.split(r'[.!?]+', summary.strip())
    complete_sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(complete_sentences) < 1:
        return False, "No complete sentences found"
    
    # Check length
    if len(summary) < 50:
        return False, f"Summary too short ({len(summary)} chars, need 50+)"
    if len(summary) > 200:  # Allow a bit more flexibility
        return False, f"Summary too long ({len(summary)} chars, max 200)"
    
    # Check for common incomplete patterns
    incomplete_patterns = [
        r'\.\.\.$',  # ends with ...
        r'\bin\s*$',  # ends with "in"
        r'\bthe\s*$', # ends with "the"
        r'\bto\s*$',  # ends with "to"
        r'\bof\s*$',  # ends with "of"
        r'\bwho\s*$',  # ends with "who"
        r'\band\s*$', # ends with "and"
        r'\bor\s*$',  # ends with "or"
        r'\bthrough\s*$', # ends with "through"
        r'\bthis\s*$', # ends with "this"
        r"'s\s*$",   # ends with 's
    ]
    
    for pattern in incomplete_patterns:
        if re.search(pattern, summary, re.IGNORECASE):
            return False, f"Summary appears incomplete (ends with suspicious pattern)"
    
    return True, "Valid"

def generate_summary_with_openai(title: str, author: str, description: str, max_retries: int = 3) -> Optional[str]:
    """Generate summary using OpenAI with validation and retries"""
    
    # Check if OpenAI API key is available
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️ OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        return None
    
    openai.api_key = api_key
    
    for attempt in range(max_retries):
        try:
            prompt = create_robust_prompt(title, author, description)
            
            # Add retry-specific instructions
            if attempt > 0:
                prompt += f"\n\nIMPORTANT: Previous attempt failed validation. Ensure your response:"
                prompt += f"\n- Is exactly 2 complete sentences"
                prompt += f"\n- Ends with proper punctuation (. ! ?)"
                prompt += f"\n- Contains 50-150 words"
                prompt += f"\n- Has no incomplete phrases or cut-off words"
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at writing engaging, complete book summaries for children's literature. Always write exactly 2 complete sentences that end with proper punctuation."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.7
            )
            
            summary = response.choices[0].message.content.strip()
            
            # Validate the summary
            is_valid, error_msg = validate_summary(summary)
            
            if is_valid:
                print(f"✓ Generated valid summary (attempt {attempt + 1}): {summary[:60]}...")
                return summary
            else:
                print(f"⚠️ Attempt {attempt + 1} failed validation: {error_msg}")
                print(f"   Generated: {summary}")
                
        except Exception as e:
            print(f"❌ Error on attempt {attempt + 1}: {e}")
    
    print(f"❌ Failed to generate valid summary after {max_retries} attempts")
    return None

def create_fallback_summary(title: str, author: str, description: str) -> str:
    """Create a simple fallback summary when AI generation fails"""
    # Extract key info from description for a basic summary
    clean_desc = description[:100].strip()
    if len(description) > 100:
        # Find the last complete word
        last_space = clean_desc.rfind(' ')
        if last_space > 50:
            clean_desc = clean_desc[:last_space]
    
    # Create a basic two-sentence structure
    summary = f"{clean_desc}. This beloved children's book offers an engaging story perfect for young readers."
    
    return summary

def regenerate_incomplete_summaries():
    """Main function to regenerate incomplete summaries"""
    print("🔧 Regenerating incomplete book summaries with robust prompts...")
    
    # Load the catalog
    try:
        df = pd.read_csv('/Users/chaerinnoh/Desktop/kidlit/data/raw/books_final_complete.csv')
        print(f"✓ Loaded catalog with {len(df)} books")
    except Exception as e:
        print(f"❌ Error loading catalog: {e}")
        return
    
    successful_fixes = []
    failed_fixes = []
    
    for book_info in BOOKS_TO_FIX:
        row_idx = book_info["row"]
        expected_title = book_info["title"]
        
        # Get book data
        if row_idx >= len(df):
            print(f"❌ Row {row_idx} not found in dataset")
            failed_fixes.append(expected_title)
            continue
            
        row = df.iloc[row_idx]
        title = row.get('title', 'Unknown')
        author = row.get('author', 'Unknown')
        description = str(row.get('description', ''))
        current_summary = str(row.get('summary_gpt', ''))
        
        print(f"\n📖 Processing: {title}")
        print(f"   Current summary: {current_summary[:80]}...")
        
        # Verify this is the right book
        if title != expected_title:
            print(f"⚠️ Title mismatch. Expected: {expected_title}, Got: {title}")
        
        # Try to generate new summary
        new_summary = generate_summary_with_openai(title, author, description)
        
        if new_summary:
            # Update the dataframe
            df.at[row_idx, 'summary_gpt'] = new_summary
            successful_fixes.append(title)
            print(f"✅ Updated summary: {new_summary}")
        else:
            # Use fallback
            fallback_summary = create_fallback_summary(title, author, description)
            df.at[row_idx, 'summary_gpt'] = fallback_summary
            failed_fixes.append(title)
            print(f"⚠️ Used fallback summary: {fallback_summary}")
    
    # Save the updated file
    if successful_fixes or failed_fixes:
        output_file = '/Users/chaerinnoh/Desktop/kidlit/data/raw/books_final_complete.csv'
        df.to_csv(output_file, index=False)
        print(f"\n💾 Saved updates to {output_file}")
        
        print(f"\n📊 Results:")
        print(f"✅ Successfully regenerated: {len(successful_fixes)} summaries")
        print(f"⚠️ Used fallback for: {len(failed_fixes)} summaries")
        
        if successful_fixes:
            print(f"\n✅ Successful regenerations:")
            for title in successful_fixes:
                print(f"   - {title}")
                
        if failed_fixes:
            print(f"\n⚠️ Fallback summaries used:")
            for title in failed_fixes:
                print(f"   - {title}")
    else:
        print("\n❌ No summaries were updated")

if __name__ == "__main__":
    regenerate_incomplete_summaries()