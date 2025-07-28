import os
import re
import json
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from .env 
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file or environment variables.")
client = OpenAI(api_key=api_key) 

# Load data
df = pd.read_csv("books_cleaned.csv").head(10)  # test on 10 rows
results = []

def clean_description(desc):
    if pd.isna(desc) or not isinstance(desc, str):
        return ""
    
    desc = desc.strip()
    desc = re.sub(r"https?://\S+", "", desc)  # remove URLs
    desc = desc.replace("\n", " ")

    # Remove “gift” or marketing phrases
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
            max_tokens=300
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
        summary = re.sub(rf"\\b{author_regex}\\b", "", summary, flags=re.IGNORECASE)

    # Remove promotional adjectives or phrases
    commercial_phrases = [
        r"\\b(beloved|classic|timeless|masterful|must-read|bestselling|special edition|award-winning|heartwarming|perfect gift|makes a great gift|makes a perfect gift|timeless tale|modern classic|instant classic|#1 bestseller|top seller|critically acclaimed|highly recommended)\\b[^.]*\\.",
        r"\\bNew York Times bestselling[^\\.\\n]*\\.",
        r"\\bWinner of [^\\.\\n]*\\.",
        r"\\bAs seen on [^\\.\\n]*\\.",
        r"\\bNow a major motion picture[^\\.\\n]*\\.",
        r"\\bFrom the author of[^\\.\\n]*\\.",
        r"\\bby [A-Z][a-z]+ [A-Z][a-z]+",  # author bios
    ]
    for phrase in commercial_phrases:
        summary = re.sub(phrase, "", summary, flags=re.IGNORECASE)

    # Remove URLs and markdown links
    summary = re.sub(r"https?://\\S+", "", summary)
    summary = re.sub(r"www\\.\\S+", "", summary)
    summary = re.sub(r"\[.*?\]\(.*?\)", "", summary)

    # Truncate if > 3 sentences
    sentences = re.split(r'(?<=[.!?]) +', summary)
    summary = " ".join(sentences[:3])

    # Collapse spaces and clean up
    summary = re.sub(r"\\s+", " ", summary).strip()

    return summary

# Process each book
for _, row in df.iterrows():
    print(f"\n📚 Processing: {row['title_clean']} by {row['author_clean']}")
    description = clean_description(row.get("description", ""))
    enriched = get_enrichment_from_gpt(description)

    row["summary_gpt"] = clean_summary(enriched["summary"], row["author_clean"])
    row["themes"] = ", ".join(enriched["themes"])
    row["tone"] = enriched["tone"]
    row["age_range"] = enriched["age_range"]
    results.append(row)
    time.sleep(1)

# Save output
pd.DataFrame(results).to_csv("books_gpt_enriched_sample.csv", index=False)
print("\n✅ Saved enriched data to books_gpt_enriched_sample.csv")








# TESTING

if __name__ == "__main__":
    test_summary = """
A poignant tale of love and acceptance between a tree and a little boy, beautifully written and illustrated by Shel Silverstein. This classic story explores the idea of unconditional giving.
"""
    print(clean_summary(test_summary, "Shel Silverstein"))
