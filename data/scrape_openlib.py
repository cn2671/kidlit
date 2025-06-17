import requests
import time
import pandas as pd

BASE_SEARCH_URL = "https://openlibrary.org/search.json"
BASE_WORK_URL = "https://openlibrary.org/works/"

def get_work_details(work_key):
    """Fetch detailed book data."""
    url = f"{BASE_WORK_URL}{work_key}.json"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {}

def scrape_openlibrary_books(query="children's books", max_books=100):
    books = []
    page = 1
    while len(books) < max_books:
        params = {"q": query, "limit": 50, "page": page}
        try:
            res = requests.get(BASE_SEARCH_URL, params=params)
            res.raise_for_status()
            docs = res.json().get("docs", [])
        except Exception as e:
            print(f"Error on page {page}: {e}")
            break
        
        for doc in docs:
            if len(books) >= max_books:
                break
            title = doc.get("title")
            authors = ", ".join(doc.get("author_name", []))
            work_key = doc.get("key", "").replace("/works/", "")
            cover_id = doc.get("cover_i")
            cover_url = f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg" if cover_id else None

            # Get detailed description
            work_data = get_work_details(work_key)
            desc = work_data.get("description")
            if isinstance(desc, dict):
                desc = desc.get("value")
            elif not isinstance(desc, str):
                desc = None

            books.append({
                "title": title,
                "author": authors,
                "work_key": work_key,
                "description": desc,
                "cover_url": cover_url
            })

            time.sleep(0.2)  

        page += 1

    return pd.DataFrame(books)

# Run the scraper and save to CSV
df = scrape_openlibrary_books(max_books=100)
df.to_csv("childrens_books_dataset.csv", index=False)
print("✅ Scraping complete. Data saved to 'childrens_books_dataset.csv'")
