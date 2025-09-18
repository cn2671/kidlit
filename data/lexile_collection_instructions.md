# VERIFIED LEXILE SCORE COLLECTION INSTRUCTIONS

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
