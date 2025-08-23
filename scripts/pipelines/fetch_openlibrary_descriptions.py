import csv
import requests
import time
from pathlib import Path

def fetch_description(work_url):
    try:
        resp = requests.get(work_url + ".json", timeout=20)
        resp.raise_for_status()
        data = resp.json()
        desc = data.get("description")
        return desc.get("value") if isinstance(desc, dict) else (desc or "")
    except Exception as e:
        print(f"Failed to fetch description from {work_url}: {e}")
        return ""

def enrich_with_description(input_csv, output_csv, folder="data"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    in_path  = Path(input_csv)
    out_path = Path(output_csv)

    if not in_path.is_absolute() and in_path.parent == Path("."):
        in_path = folder / in_path
    if not out_path.is_absolute() and out_path.parent == Path("."):
        out_path = folder / out_path

    # Read
    with in_path.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        books = list(reader)

    # Enrich
    for book in books:
        work_url = book.get("openlibrary_url", "")
        if work_url:
            print(f"Fetching description for: {book.get('title','(untitled)')}")
            book["description"] = fetch_description(work_url)
        else:
            book["description"] = ""
        time.sleep(1)

    # Save 
    fieldnames = list(books[0].keys()) if books else [
        "title","author","goodreads_url","openlibrary_url","description"
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(books)

    print(f"Saved enriched file with descriptions to {out_path.resolve()}")

if __name__ == "__main__":
    enrich_with_description(
        input_csv="books_with_openlibrary_matches.csv",
        output_csv="books_with_descriptions.csv",
        folder="data" 
    )
