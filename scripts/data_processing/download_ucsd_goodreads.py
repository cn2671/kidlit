import requests
import gzip
import json
import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "external"

def download_file(url, destination):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=destination.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"Downloaded: {destination}")

def download_ucsd_goodreads_children():
    """Download UCSD Goodreads children's book dataset"""
    
    print("📚 DOWNLOADING UCSD GOODREADS CHILDREN'S DATASET")
    print("════════════════════════════════════════════════")
    
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    base_url = "https://McAuleyLab.ucsd.edu/public_datasets/gdrive/goodreads/"
    
    # Files to download (focusing on children's books and metadata)
    files_to_download = {
        "goodreads_books_children.json.gz": "Children's books metadata",
        "goodreads_interactions_children.json.gz": "Children's book interactions", 
        "goodreads_reviews_children.json.gz": "Children's book reviews",
        "goodreads_book_authors.json.gz": "Book authors metadata",
        "goodreads_book_genres_initial.json.gz": "Book genres mapping"
    }
    
    downloaded_files = []
    
    for filename, description in files_to_download.items():
        url = base_url + filename
        destination = DATA_DIR / filename
        
        print(f"\n📥 Downloading {description}")
        print(f"   URL: {url}")
        print(f"   Destination: {destination}")
        
        try:
            # Check if file already exists
            if destination.exists():
                file_size = destination.stat().st_size / (1024 * 1024)  # MB
                print(f"   File already exists ({file_size:.1f} MB). Skipping...")
                downloaded_files.append(destination)
                continue
            
            download_file(url, destination)
            downloaded_files.append(destination)
            
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Failed to download {filename}: {e}")
            # Try alternative file if children's specific version doesn't exist
            if "children" in filename:
                print(f"   🔄 Children's subset may not exist, will use main dataset later")
    
    # If children's specific files don't exist, download main book metadata
    if not any("children" in str(f) for f in downloaded_files):
        print(f"\n📥 Downloading main books dataset (will filter for children later)")
        main_books_url = base_url + "goodreads_books.json.gz"
        main_books_dest = DATA_DIR / "goodreads_books.json.gz"
        
        try:
            if not main_books_dest.exists():
                download_file(main_books_url, main_books_dest)
                downloaded_files.append(main_books_dest)
            else:
                print(f"   Main books file already exists. Skipping...")
                downloaded_files.append(main_books_dest)
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Failed to download main books dataset: {e}")
    
    print(f"\n📊 DOWNLOAD SUMMARY")
    print(f"═══════════════════")
    print(f"Successfully downloaded: {len(downloaded_files)} files")
    
    for file_path in downloaded_files:
        file_size = file_path.stat().st_size / (1024 * 1024)  # MB
        print(f"  - {file_path.name}: {file_size:.1f} MB")
    
    return downloaded_files

def extract_children_books_sample():
    """Extract a sample of children's books from the main dataset"""
    
    print(f"\n🔍 EXTRACTING CHILDREN'S BOOKS SAMPLE")
    print(f"═════════════════════════════════════")
    
    main_books_file = DATA_DIR / "goodreads_books.json.gz"
    
    if not main_books_file.exists():
        print(f"Main books file not found. Cannot extract children's sample.")
        return None
    
    children_keywords = [
        'children', 'child', 'kids', 'kid', 'young', 'juvenile', 'elementary',
        'picture book', 'early reader', 'middle grade', 'baby', 'toddler',
        'preschool', 'kindergarten', 'ages 0', 'ages 1', 'ages 2', 'ages 3',
        'ages 4', 'ages 5', 'ages 6', 'ages 7', 'ages 8', 'ages 9', 'ages 10'
    ]
    
    children_books = []
    total_processed = 0
    
    print(f"Processing Goodreads books dataset...")
    
    try:
        with gzip.open(main_books_file, 'rt', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line_num > 50000:  # Process only first 50k books as sample
                    break
                
                try:
                    book = json.loads(line.strip())
                    total_processed += 1
                    
                    # Check if book likely relates to children
                    text_to_check = (
                        str(book.get('title', '')).lower() + ' ' +
                        str(book.get('description', '')).lower() + ' ' +
                        ' '.join(book.get('popular_shelves', [{}]))[:500].lower()
                    )
                    
                    if any(keyword in text_to_check for keyword in children_keywords):
                        children_books.append(book)
                        
                        if len(children_books) >= 1000:  # Limit to 1000 children's books
                            break
                
                except json.JSONDecodeError:
                    continue
                
                if line_num % 10000 == 0:
                    print(f"  Processed {line_num:,} books, found {len(children_books)} children's books")
        
        print(f"\n📚 EXTRACTION RESULTS")
        print(f"════════════════════")
        print(f"Total books processed: {total_processed:,}")
        print(f"Children's books found: {len(children_books):,}")
        
        if children_books:
            # Save children's books sample
            output_file = DATA_DIR / "goodreads_children_sample.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                for book in children_books:
                    f.write(json.dumps(book) + '\n')
            
            print(f"Saved children's sample to: {output_file}")
            
            # Create summary statistics
            print(f"\n📈 SAMPLE STATISTICS")
            print(f"═══════════════════")
            
            # Average rating
            ratings = [float(book.get('average_rating', 0)) for book in children_books if book.get('average_rating')]
            if ratings:
                print(f"Average rating: {sum(ratings)/len(ratings):.2f}")
            
            # Publication years
            years = [book.get('publication_year') for book in children_books if book.get('publication_year')]
            if years:
                print(f"Publication years: {min(years)} - {max(years)}")
            
            # Top authors
            authors = {}
            for book in children_books:
                for author in book.get('authors', []):
                    name = author.get('name', 'Unknown')
                    authors[name] = authors.get(name, 0) + 1
            
            print(f"\nTop 5 authors:")
            for author, count in sorted(authors.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {author}: {count} books")
        
        return children_books
    
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return None

def create_dataset_info():
    """Create information file about the downloaded datasets"""
    
    info_content = f"""
# UCSD Goodreads Dataset Information

## Source
- **Dataset**: UCSD Book Graph - Goodreads Dataset
- **URL**: https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html
- **Collection Date**: Late 2017 (updated May 2019)

## Dataset Description
The UCSD Book Graph contains comprehensive Goodreads data including:
- 2,360,655 books (1,521,962 works, 400,390 book series, 829,529 authors)
- 876,145 users
- 228,648,342 user-book interactions
- 112,131,203 reads and 104,551,549 ratings

## Downloaded Files
- `goodreads_books.json.gz`: Main book metadata (~2GB)
- `goodreads_book_authors.json.gz`: Author information
- `goodreads_book_genres_initial.json.gz`: Genre mappings
- `goodreads_children_sample.json`: Extracted children's books sample

## Usage Restrictions
- **Academic use only**
- Do not redistribute or use for commercial purposes
- Must cite required papers when using

## Citation Requirements
1. "Item Recommendation on Monotonic Behavior Chains" (RecSys'18)
2. "Fine-Grained Spoiler Detection from Large-Scale Review Corpora" (ACL'19)

## Integration with KidLit Project
This dataset can be used for:
- Cross-referencing book metadata and ratings
- Feature engineering for recommendation system
- Validation of Lexile estimates
- Enhanced book discovery and recommendation

## Next Steps
1. Process children's book sample for relevant features
2. Match books with existing KidLit dataset
3. Extract additional metadata (genres, ratings, reviews)
4. Use for model validation and enhancement
"""
    
    info_file = DATA_DIR / "UCSD_Goodreads_README.md"
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(info_content.strip())
    
    print(f"📄 Dataset information saved to: {info_file}")

if __name__ == "__main__":
    # Download UCSD Goodreads datasets
    downloaded_files = download_ucsd_goodreads_children()
    
    # Extract children's books sample if needed
    if any("goodreads_books.json.gz" in str(f) for f in downloaded_files):
        children_sample = extract_children_books_sample()
    
    # Create dataset information file
    create_dataset_info()
    
    print(f"\n✅ UCSD GOODREADS DATASET DOWNLOAD COMPLETE!")
    print(f"   Files available in: {DATA_DIR}")
    print(f"   Ready for integration with KidLit recommendation system")