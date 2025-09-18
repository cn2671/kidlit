import pandas as pd
import requests
import textstat
import re
import numpy as np
from pathlib import Path
from datetime import datetime
import time
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

class BalancedTextAnalyzer:
    """Balanced approach: improve text quality while maintaining extraction success"""
    
    def __init__(self):
        self.results = []
        self.failed_extractions = []
        
        # Known high-quality sources
        self.gutenberg_classics = {
            "tom sawyer": 74,
            "alice": 11,
            "secret garden": 113,
            "anne of green gables": 45,
            "peter pan": 16,
            "treasure island": 120,
            "jungle book": 236
        }
    
    def extract_gutenberg_sample(self, title, author):
        """Extract from Project Gutenberg if available"""
        try:
            title_lower = title.lower()
            gutenberg_id = None
            
            for keyword, book_id in self.gutenberg_classics.items():
                if keyword in title_lower:
                    gutenberg_id = book_id
                    break
            
            if not gutenberg_id:
                return None, "No Gutenberg match"
            
            url = f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}-0.txt"
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                text = response.text
                
                # Find content boundaries
                lines = text.split('\n')
                start_idx = 0
                
                # Look for chapter start or story beginning
                for i, line in enumerate(lines[:100]):
                    if any(marker in line.upper() for marker in 
                           ['CHAPTER I', 'CHAPTER 1', 'CHAPTER ONE', 'ONCE UPON']):
                        start_idx = i
                        break
                
                # Extract substantial sample
                content_lines = lines[start_idx:start_idx+200]  # First ~200 lines
                sample_text = ' '.join(content_lines)
                
                # Clean and validate
                sample_text = re.sub(r'\s+', ' ', sample_text).strip()
                words = sample_text.split()
                
                if len(words) > 300:  # Ensure substantial content
                    return ' '.join(words[:1500]), f"Project Gutenberg (ID {gutenberg_id})"
                    
            return None, "Insufficient Gutenberg content"
            
        except Exception as e:
            return None, f"Gutenberg error: {str(e)}"
    
    def extract_google_books_improved(self, title, author):
        """Improved Google Books extraction with better filtering"""
        try:
            # Clean search terms
            clean_title = re.sub(r'[^\w\s\-]', '', title).strip()
            clean_author = re.sub(r'[^\w\s\-]', '', author).strip()
            
            # Primary search
            query = f'intitle:"{clean_title}" inauthor:"{clean_author}"'
            url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults=3"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                if 'items' in data:
                    for item in data['items']:
                        volume_info = item.get('volumeInfo', {})
                        
                        # Verify this is a good match
                        api_title = volume_info.get('title', '').lower()
                        api_authors = volume_info.get('authors', [])
                        
                        title_words = clean_title.lower().split()
                        title_match = sum(1 for word in title_words if len(word) > 2 and word in api_title)
                        title_score = title_match / max(len(title_words), 1)
                        
                        author_match = any(clean_author.lower().split()[0] in author.lower() 
                                         for author in api_authors)
                        
                        if title_score > 0.5 and author_match:
                            # Extract description and try to improve it
                            description = volume_info.get('description', '')
                            improved_content = self.improve_description_content(description, title)
                            
                            if improved_content:
                                return improved_content, "Google Books (improved)"
            
            return None, "No suitable Google Books content"
            
        except Exception as e:
            return None, f"Google Books error: {str(e)}"
    
    def improve_description_content(self, description, title):
        """Improve description content to be more representative of book text"""
        try:
            if not description or len(description) < 100:
                return None
            
            # Clean HTML and formatting
            clean_desc = re.sub(r'<[^>]+>', '', description)
            clean_desc = re.sub(r'\s+', ' ', clean_desc).strip()
            
            # Remove obvious marketing language
            marketing_phrases = [
                r'winner of.*?award', r'bestselling.*?author', r'acclaimed.*?book',
                r'perfect for.*?readers', r'ages \d+-\d+', r'grade \d+'
            ]
            
            for phrase in marketing_phrases:
                clean_desc = re.sub(phrase, '', clean_desc, flags=re.IGNORECASE)
            
            # Look for actual story content indicators
            story_indicators = [
                'once upon', 'in the beginning', 'chapter one', 'the story',
                'it was', 'there was', 'long ago', 'one day'
            ]
            
            desc_lower = clean_desc.lower()
            story_score = sum(1 for indicator in story_indicators if indicator in desc_lower)
            
            # Check for dialogue or narrative elements
            dialogue_markers = clean_desc.count('"') + clean_desc.count("'")
            
            # If it looks like story content, use it
            if (story_score > 0 or dialogue_markers > 4) and len(clean_desc) > 200:
                # Take first substantial portion
                sentences = clean_desc.split('. ')
                narrative_sentences = []
                
                for sentence in sentences:
                    # Skip sentences that are clearly marketing
                    if not any(marketing in sentence.lower() for marketing in 
                              ['award', 'bestseller', 'perfect for', 'acclaimed']):
                        narrative_sentences.append(sentence)
                        
                        # Stop when we have enough narrative content
                        if len(' '.join(narrative_sentences)) > 300:
                            break
                
                result = '. '.join(narrative_sentences)
                if len(result.split()) > 50:  # Ensure minimum content
                    return result
            
            # If no clear narrative, but substantial content, use carefully
            elif len(clean_desc.split()) > 100:
                # Remove most obvious marketing, keep descriptive content
                words = clean_desc.split()
                if len(words) > 150:
                    return ' '.join(words[:150])  # Use first 150 words
                    
            return None
            
        except:
            return None
    
    def calculate_enhanced_metrics(self, text, source):
        """Calculate enhanced readability metrics with quality indicators"""
        try:
            if not text or len(text.strip()) < 50:
                return None, "Text too short"
            
            clean_text = re.sub(r'\s+', ' ', text.strip())
            words = clean_text.split()
            
            # Basic readability metrics
            fk_grade = textstat.flesch_kincaid_grade(clean_text)
            flesch_ease = textstat.flesch_reading_ease(clean_text)
            automated_ri = textstat.automated_readability_index(clean_text)
            
            # Text quality indicators
            word_count = len(words)
            sentence_count = textstat.sentence_count(clean_text)
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            # Content quality score
            dialogue_score = (clean_text.count('"') + clean_text.count("'")) / len(words) * 100
            narrative_score = sum(1 for word in words if word.lower() in 
                                ['was', 'were', 'said', 'went', 'came', 'saw', 'looked']) / len(words) * 100
            
            quality_score = min(100, dialogue_score * 2 + narrative_score)
            
            metrics = {
                'text_source': source,
                'word_count': word_count,
                'sentence_count': sentence_count,
                'avg_word_length': avg_word_length,
                'flesch_kincaid_grade': fk_grade,
                'flesch_reading_ease': flesch_ease,
                'automated_readability_index': automated_ri,
                'quality_score': quality_score,
                'text_sample': clean_text[:200] + '...' if len(clean_text) > 200 else clean_text
            }
            
            return metrics, None
            
        except Exception as e:
            return None, f"Metrics error: {str(e)}"
    
    def process_books_balanced(self):
        """Process books with balanced approach"""
        
        tracking_path = DATA_DIR / "lexile_collection_tracking.csv"
        df = pd.read_csv(tracking_path)
        completed = df[df['status'] == 'Complete'].copy()
        
        print(f"🎯 BALANCED TEXT ANALYSIS FOR FLESCH-KINCAID VALIDATION")
        print(f"{'='*70}")
        print(f"📊 Processing {len(completed)} verified books with improved methods...")
        
        for idx, row in completed.iterrows():
            title = row['title']
            author = row['author']
            verified_lexile = row['lexile_numeric']
            book_type = row['book_type']
            
            print(f"\n📖 {title[:50]}...")
            
            # Try extraction methods in priority order
            text = None
            source = None
            
            # 1. Try Project Gutenberg for classics
            text, error = self.extract_gutenberg_sample(title, author)
            if text:
                source = error
                print(f"  ✅ {source}: {len(text.split())} words")
            else:
                # 2. Try improved Google Books
                text, error = self.extract_google_books_improved(title, author)
                if text:
                    source = error
                    print(f"  ✅ {source}: {len(text.split())} words")
                else:
                    print(f"  ❌ No content: {error}")
            
            # Calculate metrics if we have text
            if text:
                metrics, calc_error = self.calculate_enhanced_metrics(text, source)
                
                if metrics:
                    result_data = {
                        'title': title,
                        'author': author,
                        'verified_lexile': verified_lexile,
                        'book_type': book_type,
                        **metrics
                    }
                    
                    self.results.append(result_data)
                    fk = metrics['flesch_kincaid_grade']
                    quality = metrics['quality_score']
                    print(f"    📊 FK: {fk:.1f} | Quality: {quality:.1f} | Verified: {verified_lexile}L")
                else:
                    print(f"    ❌ Metrics failed: {calc_error}")
                    self.failed_extractions.append({'title': title, 'reason': calc_error})
            else:
                self.failed_extractions.append({'title': title, 'reason': 'No text extracted'})
            
            time.sleep(0.8)  # Rate limiting
        
        return self.analyze_balanced_results()
    
    def analyze_balanced_results(self):
        """Analyze results with improved statistical methods"""
        
        if not self.results:
            print("\n❌ No results to analyze!")
            return None
        
        results_df = pd.DataFrame(self.results)
        
        print(f"\n🎯 BALANCED FLESCH-KINCAID ANALYSIS RESULTS")
        print(f"{'='*60}")
        print(f"✅ Successfully processed: {len(results_df)} books")
        print(f"❌ Failed extractions: {len(self.failed_extractions)}")
        
        if len(results_df) < 3:
            print("⚠️  Insufficient data for statistical analysis")
            return results_df
        
        # Enhanced results display
        print(f"\n📚 COMPREHENSIVE ANALYSIS:")
        print(f"{'Title':<35} {'Lexile':<7} {'FK':<6} {'Quality':<7} {'Source':<12}")
        print(f"{'-'*75}")
        
        for _, row in results_df.iterrows():
            title_short = row['title'][:34]
            lexile = int(row['verified_lexile'])
            fk = row['flesch_kincaid_grade']
            quality = row['quality_score']
            source = row['text_source'].split('(')[0].strip()[:11]
            print(f"{title_short:<35} {lexile:<7}L {fk:<6.1f} {quality:<7.1f} {source:<12}")
        
        # Statistical analysis
        fk_values = results_df['flesch_kincaid_grade'].values
        lexile_values = results_df['verified_lexile'].values
        
        correlation = np.corrcoef(fk_values, lexile_values)[0, 1]
        
        print(f"\n📊 STATISTICAL ANALYSIS:")
        print(f"  Sample Size: {len(results_df)} books")
        print(f"  FK-Lexile Correlation: {correlation:.3f}")
        
        # Improved regression with outlier handling
        if len(results_df) >= 5:
            # Remove extreme outliers for better fit
            q75, q25 = np.percentile(fk_values, [75, 25])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
            mask = (fk_values >= lower_bound) & (fk_values <= upper_bound)
            clean_fk = fk_values[mask]
            clean_lexile = lexile_values[mask]
            
            if len(clean_fk) >= 3:
                slope, intercept, r_value, p_value, std_err = linregress(clean_fk, clean_lexile)
                
                print(f"  Regression R²: {r_value**2:.3f}")
                print(f"  Formula: Lexile = {slope:.1f} × FK_Grade + {intercept:.1f}")
                
                # Apply improved mapping
                results_df['predicted_lexile'] = slope * results_df['flesch_kincaid_grade'] + intercept
                results_df['prediction_error'] = abs(results_df['verified_lexile'] - results_df['predicted_lexile'])
                
                avg_error = results_df['prediction_error'].mean()
                median_error = results_df['prediction_error'].median()
                
                print(f"\n🎯 PREDICTION PERFORMANCE:")
                print(f"  Average Error: {avg_error:.1f}L")
                print(f"  Median Error: {median_error:.1f}L")
                print(f"  Baseline (ML Model): 522.0L")
                
                if avg_error < 522:
                    improvement = 522 - avg_error
                    print(f"  🎉 {improvement:.1f}L improvement achieved!")
                    
                    # Show best and worst predictions
                    best_predictions = results_df.nsmallest(3, 'prediction_error')
                    worst_predictions = results_df.nlargest(3, 'prediction_error')
                    
                    print(f"\n✅ BEST PREDICTIONS:")
                    for _, row in best_predictions.iterrows():
                        print(f"  {row['title'][:30]}: {row['prediction_error']:.0f}L error")
                    
                    print(f"\n❌ WORST PREDICTIONS:")
                    for _, row in worst_predictions.iterrows():
                        print(f"  {row['title'][:30]}: {row['prediction_error']:.0f}L error")
                else:
                    print(f"  ⚠️  Still {avg_error - 522:.1f}L worse than baseline")
        
        # Save results
        results_path = DATA_DIR / "balanced_flesch_kincaid_analysis.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\n💾 Results saved: {results_path}")
        
        return results_df

def main():
    analyzer = BalancedTextAnalyzer()
    results = analyzer.process_books_balanced()
    return results

if __name__ == "__main__":
    main()