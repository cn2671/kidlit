# hybrid_query_parser.py
import re
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path for imports
ROOT = Path(__file__).resolve().parents[1]  # Go up one level from app/hybrid_query_parser.py to project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

def parse_natural_query(query: str) -> Dict:
    """
    Rule-based parser for simple, common queries
    """
    query_lower = query.lower().strip()
    
    result = {
        'themes': [],
        'age_range': None,
        'tone': None,
        'lexile_range': None,
        'title_keywords': [],
        'confidence': 'high'
    }
    
    # Age extraction patterns (order matters - ranges should come first)
    # Note: [–-] matches both en-dash (–) and regular hyphen (-)
    age_patterns = [
        r'(\d+)[–-](\d+)\s*years?',      # "5-8 years" or "5–8 years"
        r'ages?\s*(\d+)[–-](\d+)',       # "age 5-8" or "ages 5–8"
        r'for\s*ages?\s*(\d+)[–-](\d+)', # "for age 5-8" or "for ages 5–8"
        r'(\d+)\s*to\s*(\d+)',           # "5 to 8"
        r'(\d+)\s*year\s*old',           # "5 year old"
        r'ages?\s*(\d+)',                # "age 5" or "ages 6"
        r'(\d+)\s*years?\s*old',         # "5 years old"
        r'for\s*(\d+)',                  # "for 5"
    ]
    
    # Special age context patterns - these should override theme detection
    age_context_patterns = [
        (r'for\s*preschool(?:ers?)?', '3-5'),    # "for preschool" / "for preschoolers"
        (r'preschool\s*age', '3-5'),             # "preschool age"
        (r'for\s*toddlers?', '2-4'),             # "for toddlers"
        (r'for\s*kindergarten', '5-6'),          # "for kindergarten"
        (r'for\s*young\s*children', '3-6'),      # "for young children"
    ]
    
    # First check age context patterns
    for pattern, age_range in age_context_patterns:
        if re.search(pattern, query_lower):
            result['age_range'] = age_range
            break
    
    # If no age context pattern matched, try regular age patterns
    if not result['age_range']:
        for pattern in age_patterns:
            match = re.search(pattern, query_lower)
            if match:
                if len(match.groups()) == 2:  # Range like "6-8 years"
                    min_age, max_age = match.groups()
                    result['age_range'] = f"{min_age}-{max_age}"
                else:  # Single age
                    age = int(match.group(1))
                    # Expand single age to ±2 years range for better book discovery
                    min_age = max(1, age - 2)  # Don't go below 1
                    max_age = age + 2
                    result['age_range'] = f"{min_age}-{max_age}"
                break
    
    # Theme extraction - common themes
    theme_keywords = {
        'magic': ['magic', 'magical', 'wizard', 'witch', 'spell', 'fairy'],
        'friendship': ['friend', 'friendship', 'friends'],
        'adventure': ['adventure', 'quest', 'journey', 'explore'],
        'family': ['family', 'parent', 'mom', 'dad', 'sibling', 'brother', 'sister'],
        'school': ['school', 'classroom', 'teacher', 'student'],
        'animals': ['animal', 'dog', 'cat', 'pet', 'zoo', 'farm'],
        'mystery': ['mystery', 'detective', 'solve', 'clue'],
        'fantasy': ['fantasy', 'dragon', 'kingdom', 'princess', 'prince'],
        'science': ['science', 'experiment', 'space', 'robot'],
        'sports': ['sports', 'soccer', 'basketball', 'baseball', 'football'],
        'fun': ['fun', 'playful', 'entertaining'],
        'bedtime': ['bedtime', 'sleepy', 'goodnight', 'sleep'],
        'social': ['social', 'skills', 'relationships', 'communication']
    }
    
    # Handle compound phrases with & first
    compound_phrases = [
        ('light & fun', [], 'light'),  # Don't require "fun" theme, just light tone
        ('gentle & calm', [], 'calm'),
        ('fun & light', [], 'light'),  # Don't require "fun" theme, just light tone
        ('calm & gentle', [], 'calm')
    ]
    
    # Track if compound phrase was found to skip individual word extraction
    compound_phrase_found = False
    compound_phrase_words = set()
    
    for phrase, themes, tone in compound_phrases:
        if phrase in query_lower:
            result['themes'].extend(themes)
            result['tone'] = tone
            compound_phrase_found = True
            # Add words from compound phrase to blacklist
            compound_phrase_words.update(phrase.replace(' & ', ' ').split())
            break
    
    # Create a blacklist of words that shouldn't be treated as themes when in age context
    age_context_blacklist = set()
    for pattern, _ in age_context_patterns:
        if re.search(pattern, query_lower):
            # Extract words from the age context (e.g., "preschool" from "for preschool")
            words_in_context = re.findall(r'\b(?:preschool|toddler|kindergarten)\b', query_lower)
            age_context_blacklist.update(words_in_context)
    
    # Combine blacklists
    combined_blacklist = age_context_blacklist | compound_phrase_words
    
    # Regular theme extraction (but skip blacklisted words)
    for theme, keywords in theme_keywords.items():
        theme_found = False
        for keyword in keywords:
            # Use word boundaries to avoid matching substrings (e.g., "school" in "preschool")
            word_pattern = rf'\b{re.escape(keyword)}\b'
            if re.search(word_pattern, query_lower) and keyword not in combined_blacklist:
                theme_found = True
                break
        if theme_found and theme not in result['themes']:
            result['themes'].append(theme)
    
    # Tone/mood extraction
    tone_keywords = {
        'funny': ['funny', 'hilarious', 'comedy', 'humor', 'silly'],
        'scary': ['scary', 'spooky', 'horror', 'frightening'],
        'sad': ['sad', 'crying', 'emotional', 'tearjerker'],
        'inspiring': ['inspiring', 'motivational', 'uplifting'],
        'educational': ['educational', 'learning', 'teach', 'learn'],
        'light': ['light', 'fun', 'playful', 'cheerful'],
        'calm': ['calm', 'gentle', 'peaceful', 'soothing']
    }
    
    for tone, keywords in tone_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            result['tone'] = tone
            break
    
    # Reading level indicators
    level_patterns = [
        (r'easy\s+read', 'beginner'),
        (r'first\s+grade', '6-7'),
        (r'second\s+grade', '7-8'),
        (r'third\s+grade', '8-9'),
        (r'chapter\s+book', '8-12'),
        (r'picture\s+book', '3-6')
    ]
    
    for pattern, age_range in level_patterns:
        if re.search(pattern, query_lower):
            if not result['age_range']:  # Don't override explicit age
                result['age_range'] = age_range
            break
    
    # Lexile range extraction
    lexile_patterns = [
        r'lexile\s+(\d+)-(\d+)',        # "lexile 700-850"
        r'(\d+)-(\d+)\s*lexile',        # "700-850 lexile"  
        r'(\d+)\s*-\s*(\d+)L',          # "700-850L"
        r'(\d+)L\s*-\s*(\d+)L',         # "700L-850L"
    ]
    
    for pattern in lexile_patterns:
        match = re.search(pattern, query_lower)
        if match:
            min_lexile, max_lexile = match.groups()
            result['lexile_range'] = [int(min_lexile), int(max_lexile)]
            break
    
    return result

def should_use_llm(query: str, rule_result: Dict) -> bool:
    """
    Determine if query is complex enough to warrant LLM parsing
    """
    query_lower = query.lower()
    
    # Use LLM for complex queries
    complex_indicators = [
        # Long queries (more than 8 words)
        len(query.split()) > 8,
        
        # Complex sentence structures
        'like' in query_lower and 'but' in query_lower,
        'similar to' in query_lower,
        'reminds me of' in query_lower,
        'something like' in query_lower,
        
        # Emotional/subjective language
        'feel' in query_lower or 'mood' in query_lower,
        'makes me' in query_lower,
        'want something' in query_lower,
        'looking for' in query_lower and ('that' in query_lower or 'which' in query_lower),
        
        # Conditional or comparative language
        'if' in query_lower and 'then' in query_lower,
        'better than' in query_lower or 'worse than' in query_lower,
        'instead of' in query_lower,
        'not too' in query_lower,
        
        # Vague or abstract themes not caught by rules
        rule_result['themes'] == [] and any(word in query_lower for word in [
            'deep', 'meaningful', 'touching', 'inspiring', 'thought-provoking',
            'mature', 'sophisticated', 'challenging', 'complex'
        ]),
        
        # Specific book comparisons
        'harry potter' in query_lower or 'hunger games' in query_lower,
        'diary of a wimpy kid' in query_lower
    ]
    
    # Use rule-based for simple, clear queries
    simple_indicators = [
        # Simple single-word or two-word queries
        len(query.split()) <= 2,
        
        # Basic single themes (exact matches only)
        query_lower.strip() in [
            'friendship', 'magic', 'adventure', 'family', 'school', 
            'animals', 'mystery', 'fantasy', 'science', 'sports'
        ],
        
        # Simple author searches
        query_lower.startswith('books by ') and len(query.split()) <= 4,
    ]
    
    has_complex = any(complex_indicators)
    has_simple = any(simple_indicators)
    
    # Force LLM for certain patterns regardless of simple indicators
    force_llm_patterns = [
        'something like' in query_lower,
        'books like' in query_lower,
        'similar to' in query_lower,
        ('like' in query_lower and 'but' in query_lower),
        'reminds me of' in query_lower,
    ]
    
    if any(force_llm_patterns):
        return True
    
    # Use LLM if complex and not clearly simple
    return has_complex and not has_simple

def handle_book_comparisons(query: str) -> Dict:
    """
    Handle comparative queries like 'books like Harry Potter but for younger kids'
    """
    query_lower = query.lower()
    result = {
        'themes': [],
        'age_range': None,
        'tone': None,
        'lexile_range': None
    }
    
    # Book reference mappings - what themes/characteristics each book represents
    book_references = {
        'harry potter': {
            'themes': ['magic', 'adventure', 'friendship', 'fantasy'],
            'base_age': '9-12',
            'younger_age': '6-8',
            'older_age': '13+',
            'tone': 'exciting'
        },
        'diary of a wimpy kid': {
            'themes': ['humor', 'school', 'friendship', 'family'],
            'base_age': '8-12',
            'younger_age': '6-8', 
            'older_age': '13+',
            'tone': 'funny'
        },
        'magic tree house': {
            'themes': ['adventure', 'history', 'magic'],
            'base_age': '6-8',
            'younger_age': '3-5',
            'older_age': '9-12',
            'tone': 'adventurous'
        },
        'percy jackson': {
            'themes': ['mythology', 'adventure', 'fantasy', 'friendship'],
            'base_age': '10-14',
            'younger_age': '8-10',
            'older_age': '13+',
            'tone': 'exciting'
        },
        'dog man': {
            'themes': ['humor', 'adventure', 'friendship'],
            'base_age': '6-8',
            'younger_age': '3-5',
            'older_age': '9-12',
            'tone': 'funny'
        }
    }
    
    # Find book references in query
    referenced_book = None
    for book, info in book_references.items():
        if book in query_lower:
            referenced_book = info
            result['themes'] = info['themes']
            result['tone'] = info['tone']
            
            # Determine age based on modifiers
            if 'younger' in query_lower or 'easier' in query_lower:
                result['age_range'] = info['younger_age']
            elif 'older' in query_lower or 'harder' in query_lower or 'advanced' in query_lower:
                result['age_range'] = info['older_age']
            else:
                result['age_range'] = info['base_age']
            break
    
    # Handle educational modifier
    if referenced_book and ('educational' in query_lower or 'learning' in query_lower):
        result['themes'] = [t for t in result['themes'] if t != 'humor'] + ['science', 'educational']
    
    return result

def merge_parsing_results(rule_result: Dict, llm_result: Dict, comparison_result: Dict = None) -> Dict:
    """
    Merge rule-based, LLM, and comparison parsing results intelligently
    """
    merged = rule_result.copy()
    
    # If we have comparison results (from book references), prioritize those
    if comparison_result and comparison_result.get('themes'):
        merged['themes'] = comparison_result['themes']
        merged['tone'] = comparison_result.get('tone') or merged['tone']
        merged['age_range'] = comparison_result.get('age_range') or merged['age_range']
    else:
        # Original LLM merging logic, but filter out nonsensical themes
        if llm_result.get('themes'):
            # Filter out literal words that aren't meaningful themes
            meaningful_themes = [theme for theme in llm_result['themes'] 
                               if theme not in ['but', 'more', 'like', 'for', 'the', 'and', 'of', 'a']]
            combined_themes = list(set(merged['themes'] + meaningful_themes))
            merged['themes'] = combined_themes
        
        if llm_result.get('tone') and not merged['tone']:
            merged['tone'] = llm_result['tone']
        
        # Rule-based is better at age extraction
        if llm_result.get('age_range') and not merged['age_range']:
            merged['age_range'] = llm_result['age_range']
    
    # Combine lexile ranges if both exist
    if llm_result and llm_result.get('lexile_range') and not merged['lexile_range']:
        merged['lexile_range'] = llm_result['lexile_range']
    
    return merged

def hybrid_parse_query(query: str) -> Dict:
    """
    Hybrid query parser: rule-based first, then book comparisons, then LLM fallback for complex queries
    """
    # First, try rule-based parsing
    rule_based_result = parse_natural_query(query)
    
    # Check for book comparison queries first
    comparison_result = handle_book_comparisons(query)
    if comparison_result.get('themes'):
        logger.info(f"Using book comparison parsing for: {query}")
        merged = merge_parsing_results(rule_based_result, {}, comparison_result)
        merged['parsing_method'] = 'book_comparison'
        
        # If this is a direct title search (no comparison words), mark it as such
        query_lower = query.lower()
        is_direct = not ('like' in query_lower or 'similar' in query_lower or 'but' in query_lower)
        if is_direct:
            merged['is_direct_title_search'] = True
            
        return merged
    
    # Determine if query is complex enough to need LLM
    needs_llm = should_use_llm(query, rule_based_result)
    
    if not needs_llm:
        logger.info(f"Using rule-based parsing for: {query}")
        rule_based_result['parsing_method'] = 'rule_based'
        return rule_based_result
    
    # Fall back to existing LLM parser
    try:
        logger.info(f"Using LLM parsing for complex query: {query}")
        
        # Import your existing LLM parser
        from scripts.core.parse_query import parse_user_query
        
        llm_result = parse_user_query(query)
        
        # Merge rule-based and LLM results
        merged_result = merge_parsing_results(rule_based_result, llm_result)
        merged_result['parsing_method'] = 'llm'
        
        return merged_result
        
    except Exception as e:
        logger.warning(f"LLM parsing failed, using rule-based fallback: {e}")
        rule_based_result['parsing_method'] = 'rule_based_fallback'
        return rule_based_result

# Test function to verify parsing
def test_hybrid_parser():
    """Test the hybrid parser with common queries"""
    test_queries = [
        "book about magic for 7 year old",
        "friendship",
        "funny adventure story",
        "something like Harry Potter but easier",
        "mystery book for 10 year old",
        "sad book about family"
    ]
    
    for query in test_queries:
        result = hybrid_parse_query(query)
        print(f"Query: '{query}'")
        print(f"Result: {result}")
        print("-" * 50)

if __name__ == "__main__":
    test_hybrid_parser()