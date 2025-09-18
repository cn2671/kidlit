import pandas as pd
import re
from pathlib import Path

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.title()

def clean_books(input_csv, output_csv, folder="data"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    in_path  = Path(input_csv)  if Path(input_csv).is_absolute()  else folder / input_csv
    out_path = Path(output_csv) if Path(output_csv).is_absolute() else folder / output_csv

    df = pd.read_csv(in_path)

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
    df.to_csv(out_path, index=False)
    print(f"Cleaned data saved to {out_path.resolve()}")

if __name__ == "__main__":
    clean_books(
        input_csv="books_with_descriptions.csv",
        output_csv="books_cleaned.csv",
        folder="data"
    )