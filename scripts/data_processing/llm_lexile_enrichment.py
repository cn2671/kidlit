#!/usr/bin/env python3
"""
LLM-Enhanced Lexile Score Data Enrichment System
Uses LLMs to systematically search for and extract official Lexile scores
"""

import os
import sys
import pandas as pd
import json
import time
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMLexileEnricher:
    """
    Uses LLM capabilities to search for and extract official Lexile scores
    for books in your catalog
    """
    
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.enriched_data = []
        self.cache_file = self.output_dir / "llm_lexile_cache.json"
        self.results_file = self.output_dir / "enriched_lexile_scores.csv"
        
        # Load existing cache
        self.cache = self._load_cache()
        
        logger.info("🚀 LLM Lexile Enricher initialized")
    
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
    
    def search_lexile_score(self, title: str, author: str) -> Dict:
        """
        Use LLM to search for official Lexile score for a book
        
        Returns:
        {
            'lexile_score': int or None,
            'confidence': 'high'|'medium'|'low',
            'source': str,
            'raw_response': str
        }
        """
        # Check cache first
        cache_key = f"{title}|{author}".lower()
        if cache_key in self.cache:
            logger.info(f"📖 Using cached result for '{title}' by {author}")
            return self.cache[cache_key]
        
        prompt = self._create_search_prompt(title, author)
        
        try:
            # This would integrate with your preferred LLM API
            # For now, creating a structured approach for manual processing
            result = self._mock_llm_search(title, author, prompt)
            
            # Cache the result
            self.cache[cache_key] = result
            self._save_cache()
            
            return result
            
        except Exception as e:
            logger.error(f"Error searching for {title}: {e}")
            return {
                'lexile_score': None,
                'confidence': 'low',
                'source': 'error',
                'raw_response': str(e)
            }
    
    def _create_search_prompt(self, title: str, author: str) -> str:
        """Create optimized prompt for Lexile score search"""
        return f"""
Search for the OFFICIAL Lexile reading level for this children's book:

Title: "{title}"
Author: {author}

Please search these reliable sources in order:
1. Publisher websites (Scholastic, Penguin Random House, etc.)
2. Educational databases (Reading A-Z, Lexile.com)
3. Library catalogs (WorldCat, major library systems)
4. Educational publisher catalogs

IMPORTANT: Only return OFFICIAL Lexile scores, not estimates.

Response format (use EXACTLY this format):
LEXILE: [number]L (example: 450L)
SOURCE: [specific website/database name]
CONFIDENCE: [HIGH/MEDIUM/LOW]
NOTES: [any additional relevant information]

If no official Lexile score is found, respond:
LEXILE: NOT_FOUND
SOURCE: N/A
CONFIDENCE: N/A
NOTES: No official Lexile score located in reliable sources

Examples of what to look for:
- "Lexile: 450L"
- "Reading Level: 520L" 
- "Lexile Measure: 380L"
- "Grade Level Equivalent with Lexile score"
"""
    
    def _mock_llm_search(self, title: str, author: str, prompt: str) -> Dict:
        """
        Mock LLM search - replace this with actual LLM API calls
        This provides a template for the response structure
        """
        # In real implementation, this would call OpenAI/Claude/etc.
        # For now, return template structure
        
        logger.info(f"🔍 Would search for: '{title}' by {author}")
        
        # Template response - in real usage, parse LLM response
        return {
            'lexile_score': None,  # Will be extracted from LLM response
            'confidence': 'low',   # Determined by source reliability
            'source': 'mock',      # Extracted from LLM response
            'raw_response': prompt,
            'searched_at': datetime.now().isoformat()
        }
    
    def parse_llm_response(self, response: str) -> Dict:
        """Parse structured LLM response to extract Lexile data"""
        result = {
            'lexile_score': None,
            'confidence': 'low',
            'source': 'unknown',
            'raw_response': response
        }
        
        try:
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                
                if line.startswith('LEXILE:'):
                    lexile_text = line.replace('LEXILE:', '').strip()
                    if lexile_text != 'NOT_FOUND':
                        # Extract number from formats like "450L", "520 L", etc.
                        match = re.search(r'(\d+)', lexile_text)
                        if match:
                            result['lexile_score'] = int(match.group(1))
                
                elif line.startswith('SOURCE:'):
                    result['source'] = line.replace('SOURCE:', '').strip()
                
                elif line.startswith('CONFIDENCE:'):
                    confidence = line.replace('CONFIDENCE:', '').strip().lower()
                    if confidence in ['high', 'medium', 'low']:
                        result['confidence'] = confidence
        
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
        
        return result
    
    def enrich_catalog(self, catalog_path: str = None, sample_size: int = None) -> pd.DataFrame:
        """
        Enrich entire catalog with LLM-searched Lexile scores
        
        Args:
            catalog_path: Path to catalog CSV
            sample_size: Limit processing to N books for testing
        """
        # Load catalog
        if catalog_path is None:
            # Try to find catalog automatically
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
            
            # Search for Lexile score
            lexile_data = self.search_lexile_score(title, author)
            
            # Combine original book data with enriched Lexile data
            enriched_book = book.to_dict()
            enriched_book.update({
                'llm_lexile_score': lexile_data['lexile_score'],
                'llm_lexile_confidence': lexile_data['confidence'],
                'llm_lexile_source': lexile_data['source'],
                'original_lexile_score': book.get('lexile_score', None),
                'enrichment_date': datetime.now().isoformat()
            })
            
            enriched_books.append(enriched_book)
            
            # Save progress periodically
            if (idx + 1) % 50 == 0:
                self._save_progress(enriched_books)
                logger.info(f"💾 Saved progress: {idx+1} books processed")
            
            # Rate limiting to be respectful
            time.sleep(1)
        
        # Create enriched DataFrame
        enriched_df = pd.DataFrame(enriched_books)
        
        # Save final results
        enriched_df.to_csv(self.results_file, index=False)
        logger.info(f"💾 Saved enriched catalog to: {self.results_file}")
        
        # Generate summary report
        self._generate_report(enriched_df)
        
        return enriched_df
    
    def _save_progress(self, enriched_books: List[Dict]):
        """Save intermediate progress"""
        progress_file = self.output_dir / "enrichment_progress.csv"
        pd.DataFrame(enriched_books).to_csv(progress_file, index=False)
    
    def _generate_report(self, enriched_df: pd.DataFrame):
        """Generate enrichment summary report"""
        total_books = len(enriched_df)
        found_scores = enriched_df['llm_lexile_score'].notna().sum()
        
        confidence_counts = enriched_df['llm_lexile_confidence'].value_counts()
        source_counts = enriched_df['llm_lexile_source'].value_counts()
        
        report = f"""
LLM Lexile Enrichment Report
============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Summary:
- Total books processed: {total_books}
- Lexile scores found: {found_scores} ({found_scores/total_books*100:.1f}%)
- Success rate: {found_scores/total_books*100:.1f}%

Confidence Distribution:
{confidence_counts.to_string()}

Source Distribution:
{source_counts.to_string()}

Comparison with Original Scores:
- Books with original scores: {enriched_df['original_lexile_score'].notna().sum()}
- Books with new LLM scores: {found_scores}
- Books improved by enrichment: {found_scores - enriched_df['original_lexile_score'].notna().sum()}

Files Generated:
- Enriched catalog: {self.results_file}
- Cache file: {self.cache_file}
- This report: {self.output_dir / 'enrichment_report.txt'}
"""
        
        report_file = self.output_dir / 'enrichment_report.txt'
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info("📊 Enrichment Report:")
        logger.info(f"   Found scores: {found_scores}/{total_books} ({found_scores/total_books*100:.1f}%)")
        logger.info(f"   Report saved: {report_file}")

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enrich book catalog with LLM-searched Lexile scores")
    parser.add_argument("--catalog", help="Path to catalog CSV file")
    parser.add_argument("--sample", type=int, help="Process only N books for testing")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    
    args = parser.parse_args()
    
    enricher = LLMLexileEnricher(output_dir=args.output_dir)
    
    try:
        enriched_df = enricher.enrich_catalog(
            catalog_path=args.catalog,
            sample_size=args.sample
        )
        
        logger.info("🎉 Enrichment completed successfully!")
        logger.info(f"📊 Results saved to: {enricher.results_file}")
        
    except Exception as e:
        logger.error(f"❌ Enrichment failed: {e}")
        raise

if __name__ == "__main__":
    main()