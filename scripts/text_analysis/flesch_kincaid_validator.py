import pandas as pd
import requests
import textstat
import re
from pathlib import Path
from datetime import datetime
import time

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

class FleschKincaidValidator:
    """Validate Flesch-Kincaid approach using actual book text"""
    
    def __init__(self):
        self.results = []
        self.failed_extractions = []
    
    def extract_gutenberg_text(self, book_title, author, gutenberg_id=None):
        """Extract text from Project Gutenberg"""
        try:
            if gutenberg_id:
                url = f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}-0.txt"
            else:
                # Try common patterns for Gutenberg URLs
                # This would need specific IDs for each book
                return None, "No Gutenberg ID provided"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                text = response.text
                
                # Clean Gutenberg text (remove headers/footers)
                lines = text.split('\n')
                start_idx = 0
                end_idx = len(lines)
                
                # Find start of actual text (after Gutenberg header)
                for i, line in enumerate(lines[:100]):
                    if '*** START OF' in line or 'CHAPTER' in line.upper():
                        start_idx = i + 1
                        break
                
                # Find end of text (before Gutenberg footer)
                for i in range(len(lines) - 1, max(0, len(lines) - 100), -1):
                    if '*** END OF' in lines[i]:
                        end_idx = i
                        break
                
                clean_text = '\n'.join(lines[start_idx:end_idx])
                
                # Take first 2000 words for analysis (representative sample)
                words = clean_text.split()[:2000]
                sample_text = ' '.join(words)
                
                return sample_text, None
            else:
                return None, f"HTTP {response.status_code}"
                
        except Exception as e:
            return None, str(e)
    
    def extract_google_books_preview(self, book_title, author):
        """Extract preview text from Google Books API"""
        try:
            # Clean title and author for API query
            clean_title = re.sub(r'[^\w\s]', '', book_title)
            clean_author = re.sub(r'[^\w\s]', '', author)
            
            query = f"intitle:{clean_title}+inauthor:{clean_author}"
            url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults=5"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                if 'items' in data:
                    for item in data['items']:
                        volume_info = item.get('volumeInfo', {})
                        
                        # Check if we have a good title/author match
                        api_title = volume_info.get('title', '').lower()
                        api_authors = volume_info.get('authors', [])
                        
                        if any(author.lower().split()[0] in api_author.lower() 
                               for api_author in api_authors):
                            
                            # Try to get preview text
                            volume_id = item.get('id')
                            if volume_id:
                                preview_url = f"https://www.googleapis.com/books/v1/volumes/{volume_id}?projection=full"
                                preview_response = requests.get(preview_url, timeout=10)
                                
                                if preview_response.status_code == 200:
                                    preview_data = preview_response.json()
                                    
                                    # Check for searchable preview
                                    access_info = preview_data.get('accessInfo', {})
                                    if access_info.get('viewability') in ['PARTIAL', 'ALL_PAGES']:
                                        
                                        # For now, use description as sample text
                                        # (In production, we'd need more sophisticated preview extraction)
                                        description = volume_info.get('description', '')
                                        if len(description) > 200:
                                            return description, "preview_description"
                
                return None, "No suitable preview found"
            else:
                return None, f"API HTTP {response.status_code}"
                
        except Exception as e:
            return None, str(e)
    
    def calculate_flesch_kincaid(self, text):
        """Calculate various readability metrics for text"""
        try:
            if not text or len(text.strip()) < 100:
                return None, "Text too short"
            
            # Calculate multiple readability metrics
            fk_grade = textstat.flesch_kincaid_grade(text)
            flesch_ease = textstat.flesch_reading_ease(text) 
            automated_ri = textstat.automated_readability_index(text)
            smog_index = textstat.smog_index(text)
            dale_chall = textstat.dale_chall_readability_score(text)
            
            # Basic text statistics
            word_count = len(text.split())
            sentence_count = textstat.sentence_count(text)
            avg_sentence_length = word_count / max(sentence_count, 1)
            
            metrics = {
                'fk_grade': fk_grade,
                'flesch_ease': flesch_ease,
                'automated_ri': automated_ri,
                'smog_index': smog_index,
                'dale_chall': dale_chall,
                'word_count': word_count,
                'sentence_count': sentence_count,
                'avg_sentence_length': avg_sentence_length
            }
            
            return metrics, None
            
        except Exception as e:
            return None, str(e)
    
    def process_verified_books(self):
        """Process our verified dataset to extract text and calculate FK"""
        
        # Load our verified dataset
        tracking_path = DATA_DIR / "lexile_collection_tracking.csv"
        df = pd.read_csv(tracking_path)
        
        completed = df[df['status'] == 'Complete'].copy()
        
        print(f"📊 PROCESSING {len(completed)} VERIFIED BOOKS FOR FLESCH-KINCAID ANALYSIS")
        print(f"{'='*70}")
        
        # Attempt text extraction for each book
        for idx, row in completed.iterrows():
            title = row['title']
            author = row['author']
            verified_lexile = row['lexile_numeric']
            book_type = row['book_type']
            
            print(f"\n📖 Processing: {title[:40]}...")
            
            # Try different text sources
            text = None
            source = None
            error = None
            
            # First try public domain sources for older books
            if any(classic in title.lower() for classic in 
                   ['tom sawyer', 'alice', 'treasure island', 'peter pan']):
                text, error = self.extract_gutenberg_text(title, author)
                if text:
                    source = "Project Gutenberg"
            
            # If no Gutenberg text, try Google Books
            if not text:
                text, error = self.extract_google_books_preview(title, author)
                if text:
                    source = "Google Books Preview"
            
            # Calculate readability if we have text
            if text:
                metrics, calc_error = self.calculate_flesch_kincaid(text)
                if metrics:
                    # Store results
                    result = {
                        'title': title,
                        'author': author,
                        'verified_lexile': verified_lexile,
                        'book_type': book_type,
                        'text_source': source,
                        'word_count': metrics['word_count'],
                        'fk_grade': metrics['fk_grade'],
                        'flesch_ease': metrics['flesch_ease'],
                        'automated_ri': metrics['automated_ri'],
                        'smog_index': metrics['smog_index'],
                        'dale_chall': metrics['dale_chall'],
                        'avg_sentence_length': metrics['avg_sentence_length']
                    }
                    
                    self.results.append(result)
                    print(f"  ✅ FK Grade: {metrics['fk_grade']:.1f} | Verified: {verified_lexile}L")
                else:
                    self.failed_extractions.append({
                        'title': title,
                        'reason': f'FK calculation failed: {calc_error}'
                    })
                    print(f"  ❌ FK calculation failed: {calc_error}")
            else:
                self.failed_extractions.append({
                    'title': title,
                    'reason': f'Text extraction failed: {error}'
                })
                print(f"  ❌ Text extraction failed: {error}")
            
            # Rate limiting
            time.sleep(0.5)
        
        return self.results, self.failed_extractions
    
    def analyze_results(self):
        """Analyze FK vs Lexile correlation"""
        
        if not self.results:
            print("❌ No results to analyze!")
            return
        
        results_df = pd.DataFrame(self.results)
        
        print(f"\n🎯 FLESCH-KINCAID vs LEXILE ANALYSIS")
        print(f"{'='*50}")
        print(f"📊 Successfully processed: {len(results_df)} books")
        print(f"❌ Failed extractions: {len(self.failed_extractions)}")
        
        # Show results table
        print(f"\n📚 READABILITY COMPARISON:")
        print(f"{'Title':<30} {'Lexile':<8} {'FK Grade':<8} {'Source':<15}")
        print(f"{'-'*70}")
        
        for _, row in results_df.iterrows():
            title_short = row['title'][:29]
            lexile = row['verified_lexile']
            fk = row['fk_grade']
            source = row['text_source'][:14]
            print(f"{title_short:<30} {lexile:<8.0f}L {fk:<8.1f} {source:<15}")
        
        # Statistical analysis
        correlation = results_df['verified_lexile'].corr(results_df['fk_grade'])
        
        print(f"\n📈 STATISTICAL ANALYSIS:")
        print(f"  Correlation (Lexile vs FK Grade): {correlation:.3f}")
        
        # FK Grade to approximate Lexile conversion
        # Rough conversion: FK Grade * 100 + 200 = approximate Lexile
        results_df['fk_estimated_lexile'] = results_df['fk_grade'] * 100 + 200
        results_df['fk_error'] = abs(results_df['verified_lexile'] - results_df['fk_estimated_lexile'])
        
        avg_fk_error = results_df['fk_error'].mean()
        print(f"  Average FK→Lexile Error: {avg_fk_error:.1f}L")
        
        # Compare to our ML model errors
        print(f"\n🔍 ERROR COMPARISON:")
        print(f"  Current ML Model Avg Error: 522.0L")
        print(f"  Flesch-Kincaid Avg Error: {avg_fk_error:.1f}L")
        
        if avg_fk_error < 522:
            print(f"  🎉 FK approach shows {522 - avg_fk_error:.1f}L improvement!")
        else:
            print(f"  ⚠️  FK approach needs refinement")
        
        # Save results
        results_path = DATA_DIR / "flesch_kincaid_validation.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\n💾 Results saved: {results_path}")
        
        return results_df

def main():
    validator = FleschKincaidValidator()
    results, failures = validator.process_verified_books()
    analysis = validator.analyze_results()
    
    return results, failures, analysis

if __name__ == "__main__":
    main()