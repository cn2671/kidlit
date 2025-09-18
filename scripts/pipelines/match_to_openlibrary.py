import csv
import requests
import time
import re
from pathlib import Path

def normalize_title(title: str) -> str:
    """
    Remove any parenthetical chunks (e.g., '(Series, #1)') and collapse extra spaces.
    """
    if not title:
        return title
    # remove all (...) groups
    stripped = re.sub(r"\s*\([^)]*\)", "", title).strip()
    # collapse multiple spaces
    stripped = re.sub(r"\s{2,}", " ", stripped)
    return stripped

def names_match(a: str, b: str) -> bool:
    """
    Very light author match: case-insensitive and only checks that the last token matches
    (helps with 'Dr. Seuss' vs 'Theodor Seuss Geisel' mismatches).
    """
    if not a or not b:
        return True  
    a_last = a.strip().lower().split()[-1]
    b_last = b.strip().lower().split()[-1]
    return a_last == b_last

def pick_best_doc(docs, wanted_title_norm, wanted_author):
    wanted_title_norm = wanted_title_norm.lower()
    for d in docs:
        ol_title = (d.get("title") or "").lower()
        ol_title_norm = normalize_title(ol_title).lower()
        if ol_title_norm == wanted_title_norm and names_match(wanted_author, (d.get("author_name") or [None])[0]):
            return d
    # fallback: just return the first doc
    return docs[0] if docs else None

def search_openlibrary(title, author):
    """
    Try search with original title; if no match, try with parentheses stripped.
    """
    base_url = "https://openlibrary.org/search.json"

    # Original title, normalized (parentheses removed)
    candidates = [title]
    norm = normalize_title(title)
    if norm != title:
        candidates.append(norm)

    for t in candidates:
        params = {
            "title": t,
            "author": author,
            "limit": 5 
        }
        try:
            response = requests.get(base_url, params=params, timeout=15)
        except requests.RequestException:
            continue

        if response.status_code != 200:
            continue

        data = response.json()
        docs = data.get("docs", [])
        if not docs:
            continue

        chosen = pick_best_doc(docs, normalize_title(t), author)
        if not chosen:
            continue

        return {
            "ol_title": chosen.get("title"),
            "ol_author": (chosen.get("author_name") or [None])[0],
            "cover_id": chosen.get("cover_i"),
            "work_key": chosen.get("key")
        }

    return None

def enrich_with_openlibrary(input_csv, output_csv, folder=None):
    folder = Path(folder) if folder else Path(".")
    folder.mkdir(parents=True, exist_ok=True)
    in_path  = Path(input_csv)  if Path(input_csv).is_absolute()  else folder / input_csv
    out_path = Path(output_csv) if Path(output_csv).is_absolute() else folder / output_csv

    with in_path.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        books = list(reader)

    enriched_books = []
    for book in books:
        title = book.get("title", "")
        author = book.get("author", "")
        print(f"Searching: {title} by {author}")

        # Use normalized title for display/reference
        result = search_openlibrary(title, author)

        if result:
            cover_url = (
                f"https://covers.openlibrary.org/b/id/{result['cover_id']}-L.jpg"
                if result['cover_id'] else ""
            )
            work_url = (
                f"https://openlibrary.org{result['work_key']}" if result['work_key'] else ""
            )
            book.update({
                "ol_title": result["ol_title"] or "",
                "ol_author": result["ol_author"] or "",
                "cover_url": cover_url,
                "openlibrary_url": work_url
            })
        else:
            book.update({
                "ol_title": "",
                "ol_author": "",
                "cover_url": "",
                "openlibrary_url": ""
            })

        enriched_books.append(book)
        time.sleep(1) 

    # Save results
    fieldnames = list(enriched_books[0].keys()) if enriched_books else ["title","author","goodreads_url","ol_title","ol_author","cover_url","openlibrary_url"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched_books)

    print(f"Saved enriched data to {out_path.resolve()}")

if __name__ == "__main__":
    enrich_with_openlibrary(
        input_csv="goodreads_top_children_books.csv",
        output_csv="books_with_openlibrary_matches.csv",
        folder="data"
    )
