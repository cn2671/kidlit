import pandas as pd
import requests
import textstat
import re
import json
from pathlib import Path
from datetime import datetime
import time
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

class EnhancedTextExtractor:
    """Enhanced text extraction focusing on actual book content"""
    
    def __init__(self):
        self.results = []
        self.failed_extractions = []
        
        # Known Gutenberg IDs for classics
        self.gutenberg_ids = {
            "The Adventures of Tom Sawyer": 74,
            "Alice's Adventures in Wonderland": 11,
            "The Secret Garden": 113,
            "Anne of Green Gables": 45,
            "Peter Pan": 16,
            "The Wind in the Willows": 289,
            "Treasure Island": 120,
            "The Jungle Book": 236
        }
    
    def extract_gutenberg_full_text(self, title, author):
        """Extract full text from Project Gutenberg with better cleaning"""
        try:
            # Check if we have a known ID
            gutenberg_id = None
            for known_title, book_id in self.gutenberg_ids.items():
                if any(word.lower() in title.lower() for word in known_title.split() if len(word) > 3):
                    gutenberg_id = book_id
                    break
            
            if not gutenberg_id:
                return None, "No Gutenberg ID found"
            
            url = f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}-0.txt"
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                text = response.text
                
                # Advanced text cleaning
                lines = text.split('\n')
                
                # Find actual content start
                start_idx = 0
                for i, line in enumerate(lines[:200]):
                    if any(marker in line.upper() for marker in 
                           ['CHAPTER I', 'CHAPTER 1', '*** START OF', 'CHAPTER ONE']):
                        start_idx = i
                        break
                
                # Find content end
                end_idx = len(lines)
                for i in range(len(lines) - 1, max(0, len(lines) - 200), -1):
                    if '*** END OF' in lines[i] or 'End of Project Gutenberg' in lines[i]:
                        end_idx = i
                        break
                
                # Extract clean text
                content_lines = lines[start_idx:end_idx]
                clean_text = '\n'.join(content_lines)
                
                # Remove excessive whitespace and formatting
                clean_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', clean_text)
                clean_text = re.sub(r'^\s*\n', '', clean_text)
                
                # Take meaningful sample (first 3000 words)
                words = clean_text.split()
                if len(words) > 500:  # Ensure we have substantial content
                    sample_words = words[:3000]
                    sample_text = ' '.join(sample_words)
                    return sample_text, f"Gutenberg ID {gutenberg_id}"
                else:
                    return None, "Insufficient content after cleaning"
            else:
                return None, f"HTTP {response.status_code}"
                
        except Exception as e:
            return None, f"Gutenberg error: {str(e)}"
    
    def extract_google_books_content(self, title, author):
        """Enhanced Google Books extraction focusing on actual content"""
        try:
            # Clean and encode search terms
            clean_title = re.sub(r'[^\w\s]', '', title).strip()
            clean_author = re.sub(r'[^\w\s]', '', author).strip()
            
            # Try multiple query strategies
            queries = [
                f'intitle:"{clean_title}" inauthor:"{clean_author}"',
                f'{clean_title} {clean_author}',
                f'intitle:{clean_title.split()[0]} inauthor:{clean_author.split()[-1]}'
            ]
            
            for query in queries:
                encoded_query = quote(query)
                url = f"https://www.googleapis.com/books/v1/volumes?q={encoded_query}&maxResults=10"
                
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'items' not in data:
                        continue
                    
                    for item in data['items']:
                        volume_info = item.get('volumeInfo', {})
                        volume_id = item.get('id')
                        
                        # Better title/author matching
                        api_title = volume_info.get('title', '').lower()
                        api_authors = volume_info.get('authors', [])
                        
                        # Check for good match
                        title_match = any(word.lower() in api_title 
                                        for word in clean_title.split() if len(word) > 3)
                        author_match = any(author_part.lower() in ' '.join(api_authors).lower()
                                         for author_part in clean_author.split())
                        
                        if title_match and author_match and volume_id:
                            # Try to get preview content
                            content = self.extract_preview_content(volume_id, volume_info)
                            if content:
                                return content
                
                time.sleep(0.5)  # Rate limiting between queries
            
            return None, "No suitable content found"
            
        except Exception as e:
            return None, f"Google Books error: {str(e)}"
    
    def extract_preview_content(self, volume_id, volume_info):
        """Extract actual preview content from Google Books"""
        try:
            # Get detailed volume information
            detail_url = f"https://www.googleapis.com/books/v1/volumes/{volume_id}"
            detail_response = requests.get(detail_url, timeout=10)
            
            if detail_response.status_code != 200:
                return None
            
            detail_data = detail_response.json()
            access_info = detail_data.get('accessInfo', {})
            
            # Check if preview is available
            if access_info.get('viewability') not in ['PARTIAL', 'ALL_PAGES']:
                return None
            
            # Try different content sources in order of preference
            content_sources = [
                # 1. Try to find actual preview text (this would require more complex scraping)
                # For now, we'll use available text fields
                
                # 2. Use search snippets if available
                self.get_search_snippets(volume_id),
                
                # 3. Extract from description if it contains actual book text
                self.extract_content_from_description(volume_info.get('description', '')),
                
                # 4. Use preview link if available (would need web scraping)
                None
            ]
            
            for content in content_sources:
                if content and len(content) > 300:  # Ensure substantial content
                    return content, "Google Books content"
            
            return None, "No substantial preview content"
            
        except Exception as e:
            return None, f"Preview extraction error: {str(e)}"
    
    def get_search_snippets(self, volume_id):
        """Try to get search snippets that might contain actual book text"""
        try:
            # This would require additional API calls or web scraping
            # For now, return None as this needs more complex implementation
            return None
        except:
            return None
    
    def extract_content_from_description(self, description):
        """Extract actual book content from description if present"""
        try:
            if not description or len(description) < 200:
                return None
            
            # Look for patterns that suggest actual book content
            content_indicators = [
                'chapter one', 'chapter 1', 'once upon a time',
                'it was', 'the story begins', 'in the beginning'
            ]
            
            desc_lower = description.lower()
            
            # If description contains story indicators, it might be actual content
            if any(indicator in desc_lower for indicator in content_indicators):
                # Clean up HTML and formatting
                clean_desc = re.sub(r'<[^>]+>', '', description)
                clean_desc = re.sub(r'\s+', ' ', clean_desc).strip()
                
                # Only use if it's substantial and doesn't look like marketing text
                marketing_terms = ['bestseller', 'award-winning', 'acclaimed', 'perfect for']
                if (len(clean_desc) > 400 and 
                    sum(term in desc_lower for term in marketing_terms) < 2):
                    return clean_desc
            
            return None
            
        except:
            return None
    
    def calculate_readability_metrics(self, text, source_info):
        """Calculate comprehensive readability metrics"""
        try:
            if not text or len(text.strip()) < 200:
                return None, "Text too short for reliable analysis"
            
            # Clean text for analysis
            clean_text = re.sub(r'\s+', ' ', text.strip())
            
            # Calculate multiple metrics
            fk_grade = textstat.flesch_kincaid_grade(clean_text)
            flesch_ease = textstat.flesch_reading_ease(clean_text)
            automated_ri = textstat.automated_readability_index(clean_text)
            smog = textstat.smog_index(clean_text)
            dale_chall = textstat.dale_chall_readability_score(clean_text)
            
            # Text statistics
            words = len(clean_text.split())
            sentences = textstat.sentence_count(clean_text)
            syllables = textstat.syllable_count(clean_text)
            
            metrics = {
                'text_source': source_info,
                'word_count': words,
                'sentence_count': sentences,
                'syllable_count': syllables,
                'avg_words_per_sentence': words / max(sentences, 1),
                'avg_syllables_per_word': syllables / max(words, 1),
                'flesch_kincaid_grade': fk_grade,
                'flesch_reading_ease': flesch_ease,
                'automated_readability_index': automated_ri,
                'smog_index': smog,
                'dale_chall_score': dale_chall
            }
            
            return metrics, None
            
        except Exception as e:
            return None, f"Metrics calculation error: {str(e)}"
    
    def process_verified_books_enhanced(self):
        """Enhanced processing of verified books"""
        
        tracking_path = DATA_DIR / "lexile_collection_tracking.csv"
        df = pd.read_csv(tracking_path)
        completed = df[df['status'] == 'Complete'].copy()
        
        print(f"🚀 ENHANCED TEXT EXTRACTION FOR FLESCH-KINCAID ANALYSIS")
        print(f"{'='*70}")
        print(f"📊 Processing {len(completed)} verified books...")
        
        for idx, row in completed.iterrows():
            title = row['title']
            author = row['author']
            verified_lexile = row['lexile_numeric']
            book_type = row['book_type']
            
            print(f"\n📖 Processing: {title[:45]}...")
            
            # Try text extraction methods in order of preference
            text = None
            source = None
            
            # 1. Try Project Gutenberg for public domain works
            if any(classic_word in title.lower() for classic_word in 
                   ['tom sawyer', 'alice', 'secret garden', 'anne', 'peter pan', 'treasure']):
                print("  🔍 Trying Project Gutenberg...")
                text, error = self.extract_gutenberg_full_text(title, author)
                if text:
                    source = error  # Error message contains source info
                    print(f"  ✅ Gutenberg: {len(text.split())} words extracted")
                else:
                    print(f"  ❌ Gutenberg failed: {error}")
            
            # 2. Try enhanced Google Books if no Gutenberg text
            if not text:
                print("  🔍 Trying enhanced Google Books...")
                result = self.extract_google_books_content(title, author)
                if result and len(result) == 2:
                    text, source = result
                    if text:
                        print(f"  ✅ Google Books: {len(text.split())} words extracted")
                    else:
                        print(f"  ❌ Google Books failed: {source}")
                else:
                    print(f"  ❌ Google Books failed: {result}")
            
            # 3. Calculate readability if we have text
            if text and len(text.split()) > 100:
                metrics, calc_error = self.calculate_readability_metrics(text, source)
                
                if metrics:
                    result_data = {
                        'title': title,
                        'author': author,
                        'verified_lexile': verified_lexile,
                        'book_type': book_type,
                        **metrics
                    }
                    
                    self.results.append(result_data)
                    fk_grade = metrics['flesch_kincaid_grade']
                    print(f"  📊 FK Grade: {fk_grade:.1f} | Verified: {verified_lexile}L | Words: {metrics['word_count']}")
                else:
                    print(f"  ❌ Metrics calculation failed: {calc_error}")
                    self.failed_extractions.append({'title': title, 'reason': calc_error})
            else:
                print(f"  ❌ No suitable text extracted")
                self.failed_extractions.append({'title': title, 'reason': 'No text content'})
            
            # Rate limiting
            time.sleep(1.0)
        
        return self.analyze_enhanced_results()
    
    def analyze_enhanced_results(self):
        """Analyze enhanced results with better FK-Lexile mapping"""
        
        if not self.results:
            print("\n❌ No results to analyze!")
            return None
        
        results_df = pd.DataFrame(self.results)
        
        print(f"\n🎯 ENHANCED FLESCH-KINCAID ANALYSIS")
        print(f"{'='*50}")
        print(f"✅ Successfully processed: {len(results_df)} books")
        print(f"❌ Failed extractions: {len(self.failed_extractions)}")
        
        # Show detailed results
        print(f"\n📚 DETAILED READABILITY ANALYSIS:")
        print(f"{'Title':<35} {'Lexile':<7} {'FK':<6} {'Words':<6} {'Source':<15}")
        print(f"{'-'*75}")
        
        for _, row in results_df.iterrows():
            title_short = row['title'][:34]
            lexile = int(row['verified_lexile'])
            fk = row['flesch_kincaid_grade']
            words = row['word_count']
            source = str(row['text_source'])[:14]
            print(f"{title_short:<35} {lexile:<7}L {fk:<6.1f} {words:<6} {source:<15}")
        
        # Improved FK to Lexile conversion using correlation analysis
        correlation = results_df['verified_lexile'].corr(results_df['flesch_kincaid_grade'])
        print(f"\n📊 STATISTICAL ANALYSIS:")
        print(f"  Lexile ↔ FK Grade Correlation: {correlation:.3f}")
        
        # Create better mapping function based on actual data
        # Linear regression: Lexile = a * FK_Grade + b
        import numpy as np
        from scipy.stats import linregress
        
        fk_grades = results_df['flesch_kincaid_grade'].values
        lexiles = results_df['verified_lexile'].values
        
        slope, intercept, r_value, p_value, std_err = linregress(fk_grades, lexiles)
        
        print(f"  Linear Regression R²: {r_value**2:.3f}")
        print(f"  Mapping: Lexile ≈ {slope:.1f} × FK_Grade + {intercept:.1f}")
        
        # Apply improved mapping
        results_df['fk_predicted_lexile'] = slope * results_df['flesch_kincaid_grade'] + intercept
        results_df['fk_error'] = abs(results_df['verified_lexile'] - results_df['fk_predicted_lexile'])
        
        avg_error = results_df['fk_error'].mean()
        median_error = results_df['fk_error'].median()
        
        print(f"\n🎯 IMPROVED MAPPING PERFORMANCE:")
        print(f"  Average Error: {avg_error:.1f}L")
        print(f"  Median Error: {median_error:.1f}L")
        print(f"  Original ML Model Error: 522.0L")
        
        if avg_error < 522:
            improvement = 522 - avg_error
            print(f"  🎉 {improvement:.1f}L improvement over ML model!")
        else:
            print(f"  ⚠️  Still needs improvement")
        
        # Save enhanced results
        results_path = DATA_DIR / "enhanced_flesch_kincaid_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\n💾 Enhanced results saved: {results_path}")
        
        return results_df

def main():
    extractor = EnhancedTextExtractor()
    results = extractor.process_verified_books_enhanced()
    return results

if __name__ == "__main__":
    main()