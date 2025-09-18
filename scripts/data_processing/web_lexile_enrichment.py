#!/usr/bin/env python3
"""
Web-Enhanced Lexile Score Collection System
Uses web scraping and search to find official Lexile scores
"""

import os
import sys
import pandas as pd
import json
import time
import re
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
from urllib.parse import quote_plus, urljoin
from bs4 import BeautifulSoup
import random

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebLexileEnricher:
    """
    Uses web search and scraping to find official Lexile scores
    """
    
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.cache_file = self.output_dir / "web_lexile_cache.json"
        self.results_file = self.output_dir / "web_enriched_lexile_scores.csv"
        
        # Load existing cache
        self.cache = self._load_cache()
        
        # Session for requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
        logger.info("🌐 Web Lexile Enricher initialized")
    
    def _load_cache(self) -> Dict:
        """Load cached results to avoid re-processing"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                logger.info(f"📚 Loaded {len(cache)} cached results")
                return cache
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save cache to avoid losing progress"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save cache: {e}")
    
    def search_scholastic_book_wizard(self, title: str, author: str) -> Dict:
        """
        Search Scholastic Book Wizard for Lexile scores
        This is one of the most reliable free sources
        """
        try:
            # Scholastic Book Wizard search
            search_url = "https://www.scholastic.com/teachers/books-and-authors/"
            search_query = f"{title} {author} lexile"
            
            # Note: This is a template - actual implementation would need
            # to handle Scholastic's specific search API/interface
            
            logger.info(f"🔍 Searching Scholastic for: {title}")
            
            # Mock search result - replace with actual scraping logic
            return {
                'source': 'scholastic_book_wizard',
                'lexile_score': None,  # Would extract from search results
                'confidence': 'medium',
                'search_query': search_query,
                'found': False
            }
            
        except Exception as e:
            logger.error(f"Error searching Scholastic: {e}")
            return {'source': 'scholastic_book_wizard', 'error': str(e), 'found': False}
    
    def search_renaissance_ar(self, title: str, author: str) -> Dict:
        """
        Search Renaissance Learning AR BookFinder
        Another reliable source for reading levels
        """
        try:
            # AR BookFinder search
            logger.info(f"🔍 Searching AR BookFinder for: {title}")
            
            # Mock search - replace with actual implementation
            return {
                'source': 'ar_bookfinder',
                'lexile_score': None,
                'ar_level': None,  # AR has its own reading levels
                'confidence': 'medium',
                'found': False
            }
            
        except Exception as e:
            logger.error(f"Error searching AR BookFinder: {e}")
            return {'source': 'ar_bookfinder', 'error': str(e), 'found': False}
    
    def search_publisher_websites(self, title: str, author: str) -> Dict:
        """
        Search major publisher websites for book information
        Publishers often list Lexile scores on book pages
        """
        publishers = [
            'scholastic.com',
            'penguinrandomhouse.com', 
            'harpercollins.com',
            'simonandschuster.com',
            'macmillan.com'
        ]
        
        for publisher in publishers:
            try:
                logger.info(f"🔍 Searching {publisher} for: {title}")
                
                # Search publisher site
                # This would implement actual search logic for each publisher
                
                # Mock result - replace with actual implementation
                result = {
                    'source': publisher,
                    'lexile_score': None,
                    'confidence': 'high',  # Publisher data is usually accurate
                    'found': False
                }
                
                if result['found']:
                    return result
                    
            except Exception as e:
                logger.error(f"Error searching {publisher}: {e}")
                continue
        
        return {'source': 'publisher_search', 'found': False}
    
    def search_google_books_api(self, title: str, author: str) -> Dict:
        """
        Search Google Books API for book information
        Sometimes includes reading level data
        """
        try:
            # Google Books API search
            query = f"intitle:{title} inauthor:{author}"
            url = f"https://www.googleapis.com/books/v1/volumes?q={quote_plus(query)}"
            
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for item in data.get('items', [])[:3]:  # Check first 3 results
                    volume_info = item.get('volumeInfo', {})
                    
                    # Look for reading level in various fields
                    description = volume_info.get('description', '')
                    categories = volume_info.get('categories', [])
                    
                    # Search for Lexile mentions in description
                    lexile_match = re.search(r'(\d+)L\b', description, re.IGNORECASE)
                    if lexile_match:
                        return {
                            'source': 'google_books_api',
                            'lexile_score': int(lexile_match.group(1)),
                            'confidence': 'medium',
                            'found': True,
                            'description_snippet': description[:200]
                        }
            
            return {'source': 'google_books_api', 'found': False}
            
        except Exception as e:
            logger.error(f"Error searching Google Books: {e}")
            return {'source': 'google_books_api', 'error': str(e), 'found': False}
    
    def search_worldcat(self, title: str, author: str) -> Dict:
        """
        Search WorldCat for library catalog entries
        Libraries often include reading level information
        """
        try:
            # WorldCat search
            query = f'ti:"{title}" au:"{author}"'
            url = f"http://www.worldcat.org/search?q={quote_plus(query)}&qt=advanced"
            
            logger.info(f"🔍 Searching WorldCat for: {title}")
            
            # Mock search - actual implementation would scrape WorldCat results
            return {
                'source': 'worldcat',
                'lexile_score': None,
                'confidence': 'medium',
                'found': False
            }
            
        except Exception as e:
            logger.error(f"Error searching WorldCat: {e}")
            return {'source': 'worldcat', 'error': str(e), 'found': False}
    
    def comprehensive_search(self, title: str, author: str) -> Dict:
        """
        Perform comprehensive search across all sources
        """
        cache_key = f"{title}|{author}".lower()
        
        # Check cache first
        if cache_key in self.cache:
            logger.info(f"📖 Using cached result for '{title}' by {author}")
            return self.cache[cache_key]
        
        logger.info(f"🔍 Comprehensive search for: '{title}' by {author}")
        
        # Search all sources
        search_results = []
        
        # 1. Google Books API (fastest and most reliable)
        google_result = self.search_google_books_api(title, author)
        search_results.append(google_result)
        if google_result.get('found'):
            result = google_result
            result['search_method'] = 'google_books_primary'
        else:
            # 2. Scholastic Book Wizard
            scholastic_result = self.search_scholastic_book_wizard(title, author)
            search_results.append(scholastic_result)
            
            # 3. Publisher websites
            publisher_result = self.search_publisher_websites(title, author)
            search_results.append(publisher_result)
            
            # 4. Renaissance AR
            ar_result = self.search_renaissance_ar(title, author)
            search_results.append(ar_result)
            
            # 5. WorldCat
            worldcat_result = self.search_worldcat(title, author)
            search_results.append(worldcat_result)
            
            # Find best result
            result = self._select_best_result(search_results)
        
        # Add metadata
        result.update({
            'title': title,
            'author': author,
            'search_date': datetime.now().isoformat(),
            'all_sources_checked': [r.get('source') for r in search_results]
        })
        
        # Cache result
        self.cache[cache_key] = result
        self._save_cache()
        
        # Rate limiting
        time.sleep(random.uniform(1, 3))
        
        return result
    
    def _select_best_result(self, results: List[Dict]) -> Dict:
        """
        Select the best result from multiple sources
        Priority: found results with high confidence
        """
        found_results = [r for r in results if r.get('found', False)]
        
        if not found_results:
            return {
                'lexile_score': None,
                'confidence': 'none',
                'source': 'comprehensive_search',
                'found': False,
                'note': 'No Lexile score found in any source'
            }
        
        # Prioritize by source reliability
        source_priority = {
            'google_books_api': 3,
            'scholastic_book_wizard': 4,
            'publisher_search': 5,
            'ar_bookfinder': 2,
            'worldcat': 1
        }
        
        # Sort by priority and confidence
        found_results.sort(key=lambda x: (
            source_priority.get(x.get('source', ''), 0),
            {'high': 3, 'medium': 2, 'low': 1}.get(x.get('confidence', 'low'), 0)
        ), reverse=True)
        
        best_result = found_results[0]
        best_result['search_method'] = 'comprehensive_best_match'
        
        return best_result
    
    def enrich_catalog(self, catalog_path: str = None, sample_size: int = None) -> pd.DataFrame:
        """
        Enrich catalog with web-searched Lexile scores
        """
        # Load catalog
        if catalog_path is None:
            possible_paths = [
                ROOT / "data" / "processed" / "books_cleaned.csv",
                ROOT / "data" / "books_catalog.csv",
                ROOT / "webapp" / "books.csv"
            ]
            
            catalog_path = None
            for path in possible_paths:
                if path.exists():
                    catalog_path = path
                    break
            
            if catalog_path is None:
                raise FileNotFoundError("Could not find catalog file")
        
        logger.info(f"📚 Loading catalog from: {catalog_path}")
        catalog = pd.read_csv(catalog_path)
        
        if sample_size:
            catalog = catalog.head(sample_size)
            logger.info(f"🔬 Processing sample of {sample_size} books")
        
        logger.info(f"📊 Processing {len(catalog)} books")
        
        # Process each book
        enriched_books = []
        
        for idx, book in catalog.iterrows():
            title = book.get('title', '')
            author = book.get('author', '')
            
            if not title or not author:
                logger.warning(f"Skipping book {idx}: missing title or author")
                continue
            
            logger.info(f"📖 Processing {idx+1}/{len(catalog)}: '{title}' by {author}")
            
            # Comprehensive search
            search_result = self.comprehensive_search(title, author)
            
            # Combine with original data
            enriched_book = book.to_dict()
            enriched_book.update({
                'web_lexile_score': search_result.get('lexile_score'),
                'web_lexile_confidence': search_result.get('confidence'),
                'web_lexile_source': search_result.get('source'),
                'web_search_method': search_result.get('search_method'),
                'original_lexile_score': book.get('lexile_score'),
                'enrichment_date': datetime.now().isoformat()
            })
            
            enriched_books.append(enriched_book)
            
            # Save progress periodically
            if (idx + 1) % 25 == 0:
                self._save_progress(enriched_books)
                logger.info(f"💾 Progress saved: {idx+1}/{len(catalog)} books")
        
        # Create final DataFrame
        enriched_df = pd.DataFrame(enriched_books)
        enriched_df.to_csv(self.results_file, index=False)
        
        # Generate report
        self._generate_report(enriched_df)
        
        logger.info(f"✅ Web enrichment completed: {self.results_file}")
        return enriched_df
    
    def _save_progress(self, enriched_books: List[Dict]):
        """Save intermediate progress"""
        progress_file = self.output_dir / "web_enrichment_progress.csv"
        pd.DataFrame(enriched_books).to_csv(progress_file, index=False)
    
    def _generate_report(self, enriched_df: pd.DataFrame):
        """Generate enrichment summary report"""
        total_books = len(enriched_df)
        found_scores = enriched_df['web_lexile_score'].notna().sum()
        
        report = f"""
Web Lexile Enrichment Report
============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Summary:
- Total books processed: {total_books}
- Lexile scores found: {found_scores} ({found_scores/total_books*100:.1f}%)
- Success rate: {found_scores/total_books*100:.1f}%

Source Distribution:
{enriched_df['web_lexile_source'].value_counts().to_string()}

Confidence Distribution:
{enriched_df['web_lexile_confidence'].value_counts().to_string()}

Method Distribution:
{enriched_df['web_search_method'].value_counts().to_string()}

Files Generated:
- Enriched catalog: {self.results_file}
- Cache file: {self.cache_file}
"""
        
        report_file = self.output_dir / 'web_enrichment_report.txt'
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info("📊 Web Enrichment Report:")
        logger.info(f"   Found scores: {found_scores}/{total_books} ({found_scores/total_books*100:.1f}%)")

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enrich catalog with web-searched Lexile scores")
    parser.add_argument("--catalog", help="Path to catalog CSV file")
    parser.add_argument("--sample", type=int, help="Process only N books for testing")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    
    args = parser.parse_args()
    
    enricher = WebLexileEnricher(output_dir=args.output_dir)
    
    try:
        enriched_df = enricher.enrich_catalog(
            catalog_path=args.catalog,
            sample_size=args.sample
        )
        
        logger.info("🎉 Web enrichment completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Web enrichment failed: {e}")
        raise

if __name__ == "__main__":
    main()