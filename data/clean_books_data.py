import pandas as pd
import re

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.title()

def clean_books(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    # Normalize title and author
    df["title_clean"] = df["title"].apply(clean_text)
    df["author_clean"] = df["author"].apply(clean_text)

    # Fill missing values
    df["description"] = df["description"].fillna("")
    df["cover_url"] = df["cover_url"].fillna("")

    # Drop duplicates
    df = df.drop_duplicates(subset=["title_clean", "author_clean"])

    # Filter out rows with very short or missing descriptions
    df = df[df["description"].str.len() > 20]

    # Save cleaned data
    df.to_csv(output_csv, index=False)
    print(f"✅ Cleaned data saved to {output_csv}")

if __name__ == "__main__":
    clean_books(
        input_csv="books_with_descriptions.csv",
        output_csv="books_cleaned.csv"
    )