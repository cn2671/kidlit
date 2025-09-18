import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

def update_fourth_batch():
    """Update with fourth batch of verified scores"""
    
    # Load the tracking file
    tracking_path = DATA_DIR / "lexile_collection_tracking.csv"
    df = pd.read_csv(tracking_path)
    
    # Fourth batch of verified scores (all standard Lexile)
    verified_scores = [
        {
            "title": "Hatchet (Brian'S Saga, #1)",
            "verified_lexile": "1020",
            "lexile_source": "Multiple Educational Sources",
            "notes": "AR 5.7, Ages 10-14, Grades 5-8, Survival adventure, Gary Paulsen classic"
        },
        {
            "title": "Island Of The Blue Dolphins", 
            "verified_lexile": "1000",
            "lexile_source": "Multiple Educational Sources",
            "notes": "Grades 4-6, Newbery Medal 1961, Survival story, Scott O'Dell classic"
        },
        {
            "title": "A Wrinkle In Time (Time Quintet, #1)",
            "verified_lexile": "740",
            "lexile_source": "LightSail Educational Platform",
            "notes": "ATOS 4.7, Guided Reading R, Ages 9-12, Newbery Medal, Science fiction classic"
        },
        {
            "title": "The Lion, The Witch And The Wardrobe (Chronicles Of Narnia, #1)",
            "verified_lexile": "940",
            "lexile_source": "Multiple Educational Sources", 
            "notes": "ATOS 5.7, Guided Reading T, Grades 4-7, Fantasy classic, C.S. Lewis"
        },
        {
            "title": "Diary Of A Wimpy Kid (Diary Of A Wimpy Kid, #1)",
            "verified_lexile": "950",
            "lexile_source": "LightSail Educational Platform",
            "notes": "Grades 4-6, Ages 8-12, Graphic novel format, Contemporary humor series"
        }
    ]
    
    # Update the dataframe
    today = datetime.now().strftime("%Y-%m-%d")
    
    updated_count = 0
    for score_data in verified_scores:
        # Find the matching row
        mask = df['title'] == score_data['title']
        
        if mask.any():
            # Update the row
            df.loc[mask, 'verified_lexile'] = score_data['verified_lexile']
            df.loc[mask, 'lexile_source'] = score_data['lexile_source']
            df.loc[mask, 'collection_date'] = today
            df.loc[mask, 'notes'] = score_data['notes']
            df.loc[mask, 'status'] = 'Complete'
            
            # Parse prefix for new entries (all standard)
            prefix = None
            numeric = float(score_data['verified_lexile'])
            
            df.loc[mask, 'lexile_prefix'] = prefix if prefix else ''
            df.loc[mask, 'lexile_numeric'] = numeric
            df.loc[mask, 'book_type'] = 'Standard_Lexile'
            
            print(f"✅ Updated: {score_data['title'][:40]} - {score_data['verified_lexile']}L")
            updated_count += 1
        else:
            print(f"⚠️  Could not find: {score_data['title']}")
    
    # Save updated tracking file
    df.to_csv(tracking_path, index=False)
    
    print(f"\n💾 Updated {updated_count} verified Lexile scores in tracking file")
    
    # Show current progress toward 25-score target
    completed = df[df['status'] == 'Complete']
    total = len(df)
    
    print(f"\n📊 COLLECTION PROGRESS:")
    print(f"  ✅ Completed: {len(completed)}/{total}")
    print(f"  📈 Progress: {100*len(completed)/total:.1f}%")
    print(f"  🎯 Target Progress: {len(completed)}/25 toward Phase 1 goal")
    
    # Analyze by book type
    if len(completed) > 0:
        book_types = completed['book_type'].value_counts()
        print(f"\n📚 BY BOOK TYPE:")
        for book_type, count in book_types.items():
            if pd.notna(book_type) and book_type != '':
                print(f"  {book_type.replace('_', ' ')}: {count}")
    
    # Show reading level distribution
    standard_books = completed[completed['book_type'] == 'Standard_Lexile']
    if len(standard_books) > 0:
        lexile_scores = standard_books['lexile_numeric'].dropna()
        print(f"\n📈 STANDARD LEXILE DISTRIBUTION:")
        print(f"  Range: {lexile_scores.min():.0f}L - {lexile_scores.max():.0f}L")
        print(f"  Average: {lexile_scores.mean():.0f}L")
        print(f"  Books < 500L: {sum(lexile_scores < 500)} (Early readers)")
        print(f"  Books 500-800L: {sum((lexile_scores >= 500) & (lexile_scores < 800))} (Elementary)")
        print(f"  Books 800L+: {sum(lexile_scores >= 800)} (Middle grade+)")
    
    return df

if __name__ == "__main__":
    results = update_fourth_batch()