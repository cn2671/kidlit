import csv
import requests
import time

def search_openlibrary(title, author):
    base_url = "https://openlibrary.org/search.json"
    params = {
        "title": title,
        "author": author,
        "limit": 1
    }

    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        return None

    data = response.json()
    if not data["docs"]:
        return None

    doc = data["docs"][0]
    return {
        "ol_title": doc.get("title"),
        "ol_author": doc.get("author_name", [None])[0],
        "cover_id": doc.get("cover_i"),
        "work_key": doc.get("key")
    }


def enrich_with_openlibrary(input_csv, output_csv):
    with open(input_csv, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        books = list(reader)

    enriched_books = []
    for book in books:
        title = book["title"]
        author = book["author"]
        print(f"Searching: {title} by {author}")
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
                "ol_title": result["ol_title"],
                "ol_author": result["ol_author"],
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
    fieldnames = list(enriched_books[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched_books)
    print(f"Saved enriched data to {output_csv}")


#
if __name__ == "__main__":
    enrich_with_openlibrary(
        input_csv="goodreads_top_children_books.csv",
        output_csv="books_with_openlibrary_matches.csv"
    )
