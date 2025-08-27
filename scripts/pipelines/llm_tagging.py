import os
import re
import json
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from scripts.core.config import get_openai_client

ROOT = Path(__file__).resolve().parents[2]   
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Initialize OpenAI client 
client = get_openai_client()

# Load data
df = pd.read_csv(DATA_DIR / "books_cleaned.csv")
results = []

def clean_description(desc):
    if pd.isna(desc) or not isinstance(desc, str):
        return ""
    
    desc = desc.strip()
    desc = re.sub(r"https?://\S+", "", desc)  # remove URLs
    desc = desc.replace("\n", " ")

    # Remove marketing phrases
    desc = re.sub(r"(This (classic|book|story)[^.!?]*?(makes a )?(great|perfect) gift[^.!?]*[.!?])", "", desc, flags=re.IGNORECASE)
    
    # Remove author bios / other works
    desc = re.sub(r"by [A-Z][a-z]+ [A-Z][a-z]+", "", desc)  
    desc = re.sub(r"(including|such as) [^.!?]+(\.|$)", "", desc, flags=re.IGNORECASE)

    # Remove promotional name-dropping
    desc = re.sub(r"(New York Times bestselling[^.]*\.)", "", desc, flags=re.IGNORECASE)
    desc = re.sub(r"([Ss]hel Silverstein[^.]*\.)", "", desc, flags=re.IGNORECASE)

    # Truncate if too long
    desc = desc.split("Also contained in")[0]
    return desc[:1000]

def get_enrichment_from_gpt(description):
    prompt = f"""
You are a children's literature expert. Analyze the following book description and return a JSON object with:

- "summary": a neutral, concise 2–3 sentence summary for parents (no marketing language or other books)
    Do NOT mention author names, reviews, awards, comparisons to other books, or include URLs.
- "themes": a list of 3–5 key themes (single words)
- "tone": the emotional tone of the book (e.g. "whimsical", "calm", "funny", "adventurous", "emotional", etc.)
- "age_range": a suggested reading age range (e.g., "3–5" or "6–8")

Only return valid JSON. No commentary.


Description:
\"\"\"{description}\"\"\"
"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=400
        )
        content = response.choices[0].message.content.strip()
        print("\n🔎 Raw GPT response:")
        print(content)

        data = json.loads(content)
        return data

    except Exception as e:
        print(f"❌ GPT or JSON error: {e}")
        return {
            "summary": "",
            "themes": [],
            "tone": "",
            "age_range": ""
        }


def clean_summary(summary, author_name=None):
    if not isinstance(summary, str):
        return ""

    # Remove author name
    if author_name:
        author_regex = re.escape(author_name.strip())
        summary = re.sub(rf"\b{author_regex}\b", "", summary, flags=re.IGNORECASE)
        summary = re.sub(rf"by\s+{author_regex}", "", summary, flags=re.IGNORECASE)

    # Remove promotional/commercial phrases
    commercial_phrases = [
        r"\b(beloved|classic|timeless|masterful|must-read|bestselling|special edition|award-winning|heartwarming|perfect gift|makes a (great|perfect) gift|timeless tale|modern classic|instant classic|#1 bestseller|top seller|critically acclaimed|highly recommended)[^.]*\.",
        r"\bNew York Times bestselling[^.\n]*\.",
        r"\bWinner of [^.\n]*\.",
        r"\bAs seen on [^.\n]*\.",
        r"\bNow a major motion picture[^.\n]*\.",
        r"\bFrom the author of[^.\n]*\.",
        r"\bby [A-Z][a-z]+ [A-Z][a-z]+", 
    ]
    for phrase in commercial_phrases:
        summary = re.sub(phrase, "", summary, flags=re.IGNORECASE)

    # Remove URLs and markdown links
    summary = re.sub(r"https?://\S+", "", summary)
    summary = re.sub(r"www\.\S+", "", summary)
    summary = re.sub(r"\[.*?\]\(.*?\)", "", summary)

    # Truncate to 3 sentences max
    sentences = re.split(r'(?<=[.!?]) +', summary)
    summary = " ".join(sentences[:3]).strip()

    # Collapse spaces
    summary = re.sub(r"\s+", " ", summary).strip()

    return summary


# Process each book
for idx, row in df.iterrows():
    try:
        print(f"\n📚 Processing: {row['title_clean']} by {row['author_clean']}")
        
        # Check if required fields exist
        if pd.isna(row.get('title_clean')) or pd.isna(row.get('author_clean')):
            print("⚠️ Skipping due to missing title or author.")
            continue
            
        description = clean_description(row.get("description", ""))
        
        # Skip if no description
        if not description.strip():
            print("⚠️ Skipping due to empty description.")
            continue
            
        enriched = get_enrichment_from_gpt(description)

        # Check if GPT returned a valid summary
        if not enriched.get("summary") or len(enriched["summary"].split()) < 10:
            print("⚠️ Skipping due to short or empty summary from GPT.")
            continue

        # Create a copy of the row to avoid modifying the original
        row_copy = row.copy()
        row_copy["summary_gpt"] = clean_summary(enriched["summary"], row["author_clean"])
        row_copy["themes"] = ", ".join(enriched.get("themes", []))
        row_copy["tone"] = enriched.get("tone", "")
        row_copy["age_range"] = enriched.get("age_range", "")
        
        results.append(row_copy)
        time.sleep(1)
        
    except Exception as e:
        print(f"❌ Error processing row {idx}: {e}")
        continue

# Save output
if not results:
    print("❌ No results to process. Check input data.")
    exit()

llm_df = pd.DataFrame(results)

# Check if we have the required columns
if "summary_gpt" not in llm_df.columns:
    print("❌ No summary_gpt column found in results.")
    exit()

def is_incomplete_summary(summary):
    if not isinstance(summary, str):
        return True
    summary = summary.strip()
    # Check if summary is too short
    if len(summary.split()) < 6:
        return True
    # Check if summary doesn't end with proper punctuation
    if not summary.endswith(('.', '!', '?')):
        return True
    # Check if summary is empty after cleaning
    if not summary:
        return True
    return False

llm_df["needs_fix"] = llm_df["summary_gpt"].apply(is_incomplete_summary)
bad_rows = llm_df[llm_df["needs_fix"]]

print(f"\n⚠️ Incomplete or bad summaries: {len(bad_rows)}")
if not bad_rows.empty:
    print("Examples of bad summaries:")
    for idx, row in bad_rows.head(5).iterrows():
        print(f"  - {row.get('title', 'Unknown')}: '{row.get('summary_gpt', '')}'")

# Filter out bad ones
df_cleaned = llm_df[~llm_df["needs_fix"]]
out_path = DATA_DIR / "books_llm_tags.csv"
df_cleaned.to_csv(out_path, index=False)
print(f"\nSaved cleaned data to {out_path.resolve()}")

print(f"\n🔎 Total processed: {len(results)}")
print(f"✅ Rows with good summaries: {len(df_cleaned)}")
print(f"❌ Rows with bad summaries: {len(bad_rows)}")

