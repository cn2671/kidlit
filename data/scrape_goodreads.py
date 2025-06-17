import requests
from bs4 import BeautifulSoup
import time
import csv

def scrape_goodreads_list(list_url, num_books=100):
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ReadBuddyBot/1.0)"
    }

    books = []
    page = 1

    while len(books) < num_books:
        url = f"{list_url}?page={page}"
        print(f"Fetching {url} ...")
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        rows = soup.select("tr[itemtype='http://schema.org/Book']")

        for row in rows:
            if len(books) >= num_books:
                break

            # Title
            title_tag = row.select_one("a.bookTitle span")
            title = title_tag.text.strip() if title_tag else None

            # Author
            author_tag = row.select_one("a.authorName span")
            author = author_tag.text.strip() if author_tag else None

            # Goodreads book URL
            link_tag = row.select_one("a.bookTitle")
            book_url = f"https://www.goodreads.com{link_tag['href']}" if link_tag else None

            books.append({
                "title": title,
                "author": author,
                "goodreads_url": book_url
            })

        page += 1
        time.sleep(1) 

    return books


def save_books_to_csv(books, filename="goodreads_top_children_books.csv"):
    keys = books[0].keys()
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(books)
    print(f"Saved {len(books)} books to {filename}")


# --- Run the scraper ---
if __name__ == "__main__":
    list_url = "https://www.goodreads.com/list/show/86.Best_Children_s_Books"
    books = scrape_goodreads_list(list_url, num_books=150)
    save_books_to_csv(books)
