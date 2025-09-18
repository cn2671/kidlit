#!/usr/bin/env python3
"""
Manual fixes for the 9 books with incomplete summaries
"""

import pandas as pd

# Manually crafted complete summaries for the 9 problematic books
MANUAL_SUMMARIES = {
    401: {
        "title": "Pigs",
        "new_summary": "Megan opens the gate to the pig pen while feeding the pigs, leading to a mischievous adventure as the pigs escape and cause chaos around the farm. Through her experience, she learns about responsibility and the importance of following farm safety rules."
    },
    412: {
        "title": "The Tale of Mrs. Tittlemouse",
        "new_summary": "Mrs. Tittlemouse, a tidy little wood mouse, struggles to keep her burrow clean when uninvited visitors like Mr. Jackson the toad make messes in her home. After much effort cleaning and organizing, she finally manages to restore order and hosts a proper tea party for her well-behaved friends."
    },
    612: {
        "title": "The Two Towers (The Lord of the Rings, #2)",
        "new_summary": "The Fellowship has broken apart as Frodo and Sam continue toward Mordor with the treacherous Gollum as their guide, while Aragorn, Legolas, and Gimli pursue the orcs who captured Merry and Pippin. The forces of good and evil gather their strength for the epic battles that will determine the fate of Middle-earth."
    },
    663: {
        "title": "Escape to Witch Mountain",
        "new_summary": "Twins Tony and Tia discover they possess extraordinary psychic abilities and must escape from evil forces who want to exploit their powers. With the help of a kind stranger, they journey to Witch Mountain to uncover the truth about their mysterious past and find others like themselves."
    },
    849: {
        "title": "All Creatures Great and Small (All Creatures Great and Small, #1-2)",
        "new_summary": "Young veterinarian James Herriot begins his career in the beautiful Yorkshire Dales, where he encounters a colorful cast of animals and their equally memorable owners. Through heartwarming and often humorous stories, he learns valuable lessons about life, compassion, and the deep bond between humans and animals."
    },
    965: {
        "title": "Frog and Toad Are Friends (Frog and Toad, #1)",
        "new_summary": "Frog and Toad share five delightful adventures that celebrate the joys and challenges of friendship, from waiting for seeds to grow to losing a button on a walk. Through their gentle interactions and mutual support, young readers learn about loyalty, kindness, and what it means to be a true friend."
    },
    983: {
        "title": "And to Think That I Saw It on Mulberry Street",
        "new_summary": "Young Marco transforms his ordinary walk home from school into an extraordinary adventure using the power of his imagination, turning a simple horse and wagon into increasingly fantastic sights. Dr. Seuss's first children's book celebrates creativity and the wonderful possibilities that exist when we let our imaginations run free."
    },
    81: {
        "title": "Where the Red Fern Grows",
        "new_summary": "A determined young boy works tirelessly to save money for two hunting hounds, fulfilling his dream of owning and training the best coon dogs in the Ozark Mountains. The story follows his adventures with Old Dan and Little Ann, exploring themes of dedication, love, and the special bond between a boy and his dogs."
    },
    1070: {
        "title": "The Giraffe and the Pelly and Me",
        "new_summary": "A young boy befriends a giraffe, a pelican, and a monkey who run a window-cleaning business from an old sweet shop. Together they help catch a burglar and are rewarded by the Duke, allowing the boy to fulfill his dream of owning a candy store filled with all his favorite sweets."
    }
}

def apply_manual_summary_fixes():
    """Apply manually crafted summaries to fix the incomplete ones"""
    print("🔧 Applying manual summary fixes...")
    
    # Load the catalog
    try:
        df = pd.read_csv('/Users/chaerinnoh/Desktop/kidlit/data/raw/books_final_complete.csv')
        print(f"✓ Loaded catalog with {len(df)} books")
    except Exception as e:
        print(f"❌ Error loading catalog: {e}")
        return
    
    successful_fixes = []
    
    for row_idx, fix_data in MANUAL_SUMMARIES.items():
        expected_title = fix_data["title"]
        new_summary = fix_data["new_summary"]
        
        # Verify row exists
        if row_idx >= len(df):
            print(f"❌ Row {row_idx} not found in dataset")
            continue
            
        row = df.iloc[row_idx]
        actual_title = row.get('title', 'Unknown')
        current_summary = str(row.get('summary_gpt', ''))
        
        print(f"\n📖 Fixing: {actual_title}")
        print(f"   Current: {current_summary[:80]}...")
        print(f"   New: {new_summary[:80]}...")
        
        # Verify this is the right book
        if expected_title not in actual_title:
            print(f"⚠️ Title mismatch. Expected: {expected_title}, Got: {actual_title}")
        
        # Update the summary
        df.at[row_idx, 'summary_gpt'] = new_summary
        successful_fixes.append(actual_title)
        print(f"✅ Updated successfully")
    
    # Save the updated file
    if successful_fixes:
        output_file = '/Users/chaerinnoh/Desktop/kidlit/data/raw/books_final_complete.csv'
        df.to_csv(output_file, index=False)
        print(f"\n💾 Saved {len(successful_fixes)} manual fixes to {output_file}")
        
        print(f"\n✅ Successfully fixed summaries:")
        for title in successful_fixes:
            print(f"   - {title}")
    else:
        print("\n❌ No summaries were updated")

if __name__ == "__main__":
    apply_manual_summary_fixes()