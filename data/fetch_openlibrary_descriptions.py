import csv
import requests
import time

def fetch_description(work_url):
    try:
        response = requests.get(work_url + ".json")
        response.raise_for_status()
        data = response.json()

        desc = data.get("description")
        if isinstance(desc, dict):
            return desc.get("value")
        elif isinstance(desc, str):
            return desc
        return ""
    except Exception as e:
        print(f"Failed to fetch description from {work_url}: {e}")
        return ""

def enrich_with_description(input_csv, output_csv):
    with open(input_csv, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        books = list(reader)

    for book in books:
        work_url = book.get("openlibrary_url", "")
        if work_url:
            print(f"Fetching description for: {book['title']}")
            description = fetch_description(work_url)
            book["description"] = description
        else:
            book["description"] = ""

        time.sleep(1)  # be polite

    # Save to new CSV
    fieldnames = list(books[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(books)

    print(f"Saved enriched file with descriptions to {output_csv}")


# Run the enrichment
if __name__ == "__main__":
    enrich_with_description(
        input_csv="books_with_openlibrary_matches.csv",
        output_csv="books_with_descriptions.csv"
    )
