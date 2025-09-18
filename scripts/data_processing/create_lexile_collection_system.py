import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

def create_manual_collection_system():
    """Create a manual system for collecting verified Lexile scores"""
    
    print("📋 CREATING VERIFIED LEXILE COLLECTION SYSTEM")
    print("═════════════════════════════════════════════")
    
    # Load current dataset
    enhanced_path = DATA_DIR / "books_final_enhanced.csv"
    df = pd.read_csv(enhanced_path)
    
    print(f"📚 Current dataset: {len(df)} books")
    
    # Create list of high-priority books for collection
    popular_books = [
        # Classic children's literature
        ("Where the Wild Things Are", "Maurice Sendak"),
        ("The Cat in the Hat", "Dr. Seuss"),
        ("Green Eggs and Ham", "Dr. Seuss"),
        ("Goodnight Moon", "Margaret Wise Brown"),
        ("The Very Hungry Caterpillar", "Eric Carle"),
        ("Bridge to Terabithia", "Katherine Paterson"),
        ("Charlotte's Web", "E.B. White"),
        ("Matilda", "Roald Dahl"),
        ("The BFG", "Roald Dahl"),
        ("Charlie and the Chocolate Factory", "Roald Dahl"),
        ("The Giving Tree", "Shel Silverstein"),
        ("Corduroy", "Don Freeman"),
        ("Madeline", "Ludwig Bemelmans"),
        ("The Polar Express", "Chris Van Allsburg"),
        ("Stellaluna", "Janell Cannon"),
        
        # Popular series
        ("Harry Potter and the Sorcerer's Stone", "J.K. Rowling"),
        ("Diary of a Wimpy Kid", "Jeff Kinney"),
        ("Magic Tree House", "Mary Pope Osborne"),
        ("Junie B. Jones", "Barbara Park"),
        ("Captain Underpants", "Dav Pilkey"),
        ("Dog Man", "Dav Pilkey"),
        ("The Magic School Bus", "Joanna Cole"),
        
        # Award winners
        ("Wonder", "R.J. Palacio"),
        ("Number the Stars", "Lois Lowry"),
        ("The Giver", "Lois Lowry"),
        ("Holes", "Louis Sachar"),
        ("Hatchet", "Gary Paulsen"),
        ("Island of the Blue Dolphins", "Scott O'Dell"),
        ("A Wrinkle in Time", "Madeleine L'Engle"),
        ("The Lion, the Witch and the Wardrobe", "C.S. Lewis"),
    ]
    
    # Find these books in our dataset
    collection_list = []
    found_count = 0
    
    for title, author in popular_books:
        # Search for the book
        title_matches = df[df['title_clean'].str.contains(title, case=False, na=False)]
        
        if len(title_matches) > 0:
            # Try to find exact author match
            exact_match = title_matches[title_matches['author_clean'].str.contains(
                author.split()[0], case=False, na=False)]
            
            if len(exact_match) > 0:
                book = exact_match.iloc[0]
                found_count += 1
            else:
                book = title_matches.iloc[0]
                found_count += 1
            
            collection_list.append({
                'title': book['title_clean'],
                'author': book['author_clean'],
                'current_ml_estimate': book.get('lexile_enhanced', 'N/A'),
                'ml_confidence': book.get('lexile_confidence_enhanced', 'N/A'),
                'search_query': f'"{title}" "{author}" lexile level',
                'google_search_url': f'https://www.google.com/search?q={title.replace(" ", "+")}+{author.replace(" ", "+")}+lexile+level',
                'verified_lexile': '',
                'lexile_source': '',
                'collection_date': '',
                'notes': '',
                'status': 'Pending'
            })
    
    print(f"✅ Found {found_count} priority books in dataset")
    
    # Add some random sampling from different ranges
    print(f"\n📊 Adding sample books from different Lexile ranges...")
    
    # Sample books from different ranges
    ranges = [
        (0, 400, "Early_Readers", 10),
        (400, 700, "Transitional", 15), 
        (700, 1000, "Middle_Grade", 15),
        (1000, 1300, "Advanced", 10),
        (1300, 1600, "Young_Adult", 5)
    ]
    
    for min_lex, max_lex, label, count in ranges:
        range_books = df[
            (df['lexile_enhanced'] >= min_lex) & 
            (df['lexile_enhanced'] < max_lex) &
            (df['summary_gpt'].notna()) &
            (df['summary_gpt'].str.len() > 50)
        ]
        
        if len(range_books) > 0:
            # Take first N books (deterministic sampling)
            sample_size = min(count, len(range_books))
            sample_books = range_books.head(sample_size)
            
            for _, book in sample_books.iterrows():
                collection_list.append({
                    'title': book['title_clean'],
                    'author': book['author_clean'],
                    'current_ml_estimate': book.get('lexile_enhanced', 'N/A'),
                    'ml_confidence': book.get('lexile_confidence_enhanced', 'N/A'),
                    'search_query': f'"{book["title_clean"]}" "{book["author_clean"]}" lexile level',
                    'google_search_url': f'https://www.google.com/search?q={book["title_clean"].replace(" ", "+")}+{book["author_clean"].replace(" ", "+")}+lexile+level',
                    'verified_lexile': '',
                    'lexile_source': '',
                    'collection_date': '',
                    'notes': '',
                    'status': f'Sample_{label}'
                })
            
            print(f"  {label} ({min_lex}-{max_lex}L): {sample_size} books")
    
    # Create collection tracking spreadsheet
    collection_df = pd.DataFrame(collection_list)
    collection_path = DATA_DIR / "lexile_collection_tracking.csv"
    collection_df.to_csv(collection_path, index=False)
    
    print(f"\n💾 Collection tracking saved: {collection_path}")
    print(f"📊 Total books to collect: {len(collection_list)}")
    
    # Create instruction manual
    instructions = """# VERIFIED LEXILE SCORE COLLECTION INSTRUCTIONS

## How to Use This System

### Step 1: Open the Tracking File
Open `lexile_collection_tracking.csv` in Excel or Google Sheets

### Step 2: For Each Book Row:

1. **Copy the Google Search URL** from the 'google_search_url' column
2. **Paste it into your browser** 
3. **Look for Google's AI Overview** at the top of results
4. **Find the Lexile score** - it will say something like:
   - "Bridge to Terabithia has a Lexile level of 810L"
   - "The Lexile measure is 740L"
   - "Lexile score of 630L"

### Step 3: Record Your Findings

Fill in these columns:
- **verified_lexile**: Just the number (e.g., 810)
- **lexile_source**: Where you found it (e.g., "Google AI Overview", "Scholastic", "MetaMetrics")
- **collection_date**: Today's date (YYYY-MM-DD)
- **notes**: Any additional info (AR level, grade level, etc.)
- **status**: Change from "Pending" to "Complete"

### Step 4: Alternative Sources

If Google doesn't have the info, try:
- Scholastic Book Wizard
- Renaissance AR BookFinder  
- TeachingBooks.net
- Publisher websites
- School district reading lists

### Example Entry:
```
Title: Bridge to Terabithia
Author: Katherine Paterson
Search Query: "Bridge to Terabithia" "Katherine Paterson" lexile level
Verified Lexile: 810
Lexile Source: Google AI Overview
Collection Date: 2025-09-03
Notes: AR Level 4.6, Grades 3-8
Status: Complete
```

## Priority Order:
1. **Start with "Pending" status books** (high-priority classics)
2. **Then work on "Sample_" books** to get diverse range coverage
3. **Focus on books you recognize** - they're more likely to have verified scores

## Goal:
- **Phase 1**: 25 verified scores (enough for initial validation)
- **Phase 2**: 50 verified scores (good model improvement)  
- **Phase 3**: 100+ verified scores (comprehensive retraining)

## Tips:
- Classic children's books almost always have verified Lexile scores
- Popular series books are well-documented
- Award winners typically have reading level data
- Very obscure books may not have verified scores - skip these
"""

    instructions_path = DATA_DIR / "lexile_collection_instructions.md"
    with open(instructions_path, 'w') as f:
        f.write(instructions)
    
    print(f"📄 Instructions saved: {instructions_path}")
    
    # Create analysis script for after collection
    analysis_code = '''
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def analyze_collected_lexile_scores():
    """Analyze collected verified Lexile scores"""
    
    # Load collection data
    df = pd.read_csv('data/lexile_collection_tracking.csv')
    
    # Filter completed entries
    completed = df[
        (df['status'] == 'Complete') & 
        (df['verified_lexile'] != '') &
        (df['verified_lexile'].notna())
    ].copy()
    
    if len(completed) == 0:
        print("❌ No completed collections found yet!")
        print("📋 Continue collecting verified Lexile scores")
        return
    
    print(f"📊 ANALYZING {len(completed)} VERIFIED LEXILE SCORES")
    print(f"{'='*50}")
    
    # Convert to numeric
    completed['verified_lexile'] = pd.to_numeric(completed['verified_lexile'], errors='coerce')
    completed['current_ml_estimate'] = pd.to_numeric(completed['current_ml_estimate'], errors='coerce')
    
    # Remove any conversion errors
    valid = completed.dropna(subset=['verified_lexile', 'current_ml_estimate'])
    
    if len(valid) < 5:
        print(f"⚠️  Only {len(valid)} valid comparisons - need more data")
        return
    
    # Calculate accuracy metrics
    verified_scores = valid['verified_lexile'].values
    ml_scores = valid['current_ml_estimate'].values
    
    mae = mean_absolute_error(verified_scores, ml_scores)
    rmse = np.sqrt(np.mean((verified_scores - ml_scores) ** 2))
    r2 = r2_score(verified_scores, ml_scores)
    
    print(f"🎯 MODEL ACCURACY vs VERIFIED SCORES:")
    print(f"  Mean Absolute Error: {mae:.1f} Lexile points")
    print(f"  Root Mean Square Error: {rmse:.1f} Lexile points")  
    print(f"  R² Score: {r2:.3f}")
    
    # Error distribution analysis
    errors = np.abs(ml_scores - verified_scores)
    
    print(f"\\n📈 ERROR DISTRIBUTION:")
    print(f"  Excellent (≤100L): {sum(errors <= 100)}/{len(errors)} ({100*sum(errors <= 100)/len(errors):.1f}%)")
    print(f"  Good (≤200L): {sum(errors <= 200)}/{len(errors)} ({100*sum(errors <= 200)/len(errors):.1f}%)")
    print(f"  Acceptable (≤300L): {sum(errors <= 300)}/{len(errors)} ({100*sum(errors <= 300)/len(errors):.1f}%)")
    print(f"  Poor (>300L): {sum(errors > 300)}/{len(errors)} ({100*sum(errors > 300)/len(errors):.1f}%)")
    
    # Show worst predictions for analysis
    valid['error'] = errors
    worst = valid.nlargest(5, 'error')[['title', 'author', 'verified_lexile', 'current_ml_estimate', 'error']]
    
    print(f"\\n❌ LARGEST ERRORS (for investigation):")
    for _, row in worst.iterrows():
        print(f"  {row['title']}: Verified {row['verified_lexile']}L vs ML {row['current_ml_estimate']}L (error: {row['error']:.0f}L)")
    
    # Show best predictions
    best = valid.nsmallest(5, 'error')[['title', 'author', 'verified_lexile', 'current_ml_estimate', 'error']]
    
    print(f"\\n✅ BEST PREDICTIONS:")
    for _, row in best.iterrows():
        print(f"  {row['title']}: Verified {row['verified_lexile']}L vs ML {row['current_ml_estimate']}L (error: {row['error']:.0f}L)")
    
    # Recommendations
    print(f"\\n🎯 RECOMMENDATIONS:")
    if mae < 100:
        print(f"  ✅ Excellent model performance - ready for production!")
    elif mae < 200:
        print(f"  ⭐ Good model performance - minor improvements possible")
    elif mae < 300:
        print(f"  ⚠️  Moderate performance - consider model retraining")
    else:
        print(f"  ❌ Poor performance - model needs significant improvement")
    
    print(f"\\n📊 Next steps:")
    print(f"  • If MAE < 200L: Model is ready for use")
    print(f"  • If MAE > 200L: Collect more verified scores and retrain")
    print(f"  • Target: 100+ verified scores for comprehensive retraining")
    
    return valid, mae, r2

if __name__ == "__main__":
    analyze_collected_lexile_scores()
'''
    
    analysis_path = DATA_DIR / "analyze_lexile_collection.py"
    with open(analysis_path, 'w') as f:
        f.write(analysis_code)
    
    print(f"📊 Analysis script saved: {analysis_path}")
    
    print(f"\n🚀 LEXILE COLLECTION SYSTEM IS READY!")
    print(f"{'='*50}")
    print(f"📋 Next Steps:")
    print(f"  1. Open: {collection_path}")
    print(f"  2. Read: {instructions_path}")
    print(f"  3. Start collecting verified Lexile scores manually")
    print(f"  4. After 25+ scores: python {analysis_path}")
    print(f"  5. Use results to improve the model")
    
    print(f"\n💡 COLLECTION STRATEGY:")
    print(f"  • Start with classics (Harry Potter, Dr. Seuss, etc.)")
    print(f"  • Use Google search: 'Book Title Author lexile level'")
    print(f"  • Look for AI Overview with Lexile scores")
    print(f"  • Record findings in the tracking spreadsheet")
    print(f"  • Goal: 25 verified scores for initial validation")

if __name__ == "__main__":
    create_manual_collection_system()