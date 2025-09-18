"""
Lexile utilities for book recommendations
"""

import re
import pandas as pd
from typing import Optional, Tuple, Dict, List

# Standard age to Lexile mappings based on educational research
AGE_TO_LEXILE_MAPPING = {
    (0, 2): (0, 100),      # Baby/board books
    (3, 5): (50, 300),     # Early readers/picture books  
    (4, 6): (100, 400),    # Kindergarten level
    (5, 7): (200, 500),    # Beginning readers
    (6, 8): (300, 600),    # Early chapter books
    (7, 9): (400, 700),    # Transitional readers
    (8, 10): (500, 800),   # Intermediate readers
    (9, 12): (600, 900),   # Middle grade
    (11, 14): (700, 1000), # Advanced middle grade
    (13, 18): (800, 1200), # Young adult
}

LEXILE_RANGE_LABELS = {
    (0, 200): "BR-200L (Ages 3-5)",
    (200, 400): "200-400L (Ages 5-7)", 
    (400, 600): "400-600L (Ages 6-8)",
    (600, 800): "600-800L (Ages 8-10)",
    (800, 1000): "800-1000L (Ages 10-12)",
    (1000, 1500): "1000L+ (Ages 12+)",
}

def extract_lexile_from_query(query_text: str) -> Optional[Tuple[int, int]]:
    """Extract Lexile range from user query text"""
    
    query_lower = query_text.lower()
    
    # Pattern 1: "lexile 400-600" or "400-600 lexile"
    range_match = re.search(r'lexile\s*(\d+)[\-–](\d+)|(\d+)[\-–](\d+)\s*lexile', query_lower)
    if range_match:
        if range_match.group(1) and range_match.group(2):
            return (int(range_match.group(1)), int(range_match.group(2)))
        elif range_match.group(3) and range_match.group(4):
            return (int(range_match.group(3)), int(range_match.group(4)))
    
    # Pattern 2: "lexile 500" (single level)
    single_match = re.search(r'lexile\s*(\d+)', query_lower)
    if single_match:
        level = int(single_match.group(1))
        return (level - 100, level + 100)  # ±100 range
    
    # Pattern 3: "400L" format
    l_match = re.search(r'(\d+)l\b', query_lower)
    if l_match:
        level = int(l_match.group(1))
        return (level - 50, level + 50)  # ±50 range
    
    # Pattern 4: Range without "lexile" keyword
    bare_range = re.search(r'\b(\d{2,4})[\-–](\d{2,4})\b', query_lower)
    if bare_range:
        min_val, max_val = int(bare_range.group(1)), int(bare_range.group(2))
        # Only treat as Lexile if values are in typical Lexile range
        if 50 <= min_val <= 1500 and 50 <= max_val <= 1500:
            return (min_val, max_val)
    
    return None

def parse_age_range_to_lexile(age_range_str: str) -> Tuple[Optional[int], float]:
    """Convert age range string to Lexile estimate"""
    
    age_range_str = str(age_range_str).strip()
    
    # Handle different age range formats
    # Format 1: "6-8" or "3-5" 
    range_match = re.match(r'^(\d+)[\-–](\d+)$', age_range_str)
    if range_match:
        min_age = int(range_match.group(1))
        max_age = int(range_match.group(2))
        return find_best_lexile_match(min_age, max_age)
    
    # Format 2: "13+" 
    plus_match = re.match(r'^(\d+)\+$', age_range_str)
    if plus_match:
        min_age = int(plus_match.group(1))
        max_age = min_age + 5  # Assume +5 years for "13+"
        return find_best_lexile_match(min_age, max_age)
    
    # Format 3: Single age "5"
    single_match = re.match(r'^\d+$', age_range_str)
    if single_match:
        age = int(age_range_str)
        return find_best_lexile_match(age, age + 1)
    
    return None, 0.0

def find_best_lexile_match(min_age: int, max_age: int) -> Tuple[Optional[int], float]:
    """Find the best Lexile estimate for a given age range"""
    
    best_match = None
    best_overlap = 0
    
    for (range_min, range_max), (lexile_min, lexile_max) in AGE_TO_LEXILE_MAPPING.items():
        # Calculate overlap between book age range and mapping range
        overlap_start = max(min_age, range_min)
        overlap_end = min(max_age, range_max)
        overlap = max(0, overlap_end - overlap_start)
        
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = (lexile_min, lexile_max)
    
    if best_match:
        # Use middle of Lexile range as estimate
        lexile_estimate = (best_match[0] + best_match[1]) // 2
        
        # Confidence based on age range precision
        age_span = max_age - min_age
        if age_span <= 2:
            confidence = 0.8  # Narrow range = high confidence
        elif age_span <= 4:
            confidence = 0.6  # Medium range = medium confidence  
        else:
            confidence = 0.4  # Wide range = lower confidence
            
        return lexile_estimate, confidence
    
    return None, 0.0

def filter_by_lexile_range(df: pd.DataFrame, lexile_range: Tuple[int, int]) -> pd.DataFrame:
    """Filter dataframe by Lexile score range"""
    
    if not lexile_range or len(lexile_range) != 2:
        return df
    
    min_lexile, max_lexile = lexile_range
    
    # Ensure min <= max
    if min_lexile > max_lexile:
        min_lexile, max_lexile = max_lexile, min_lexile
    
    # Filter books with Lexile scores in range
    filtered = df[
        (df['lexile_score'].notna()) & 
        (df['lexile_score'] >= min_lexile) & 
        (df['lexile_score'] <= max_lexile)
    ]
    
    return filtered

def get_lexile_range_label(lexile_score: int) -> str:
    """Get descriptive label for a Lexile score"""
    
    for (min_score, max_score), label in LEXILE_RANGE_LABELS.items():
        if min_score <= lexile_score <= max_score:
            return label
    
    # Fallback for scores outside normal ranges
    if lexile_score < 50:
        return "BR (Below Reader)"
    elif lexile_score > 1500:
        return "1500L+ (Advanced)"
    else:
        return f"{lexile_score}L"

def calculate_lexile_progression(current_lexile: int, progression_type: str = "up") -> Tuple[int, int]:
    """Calculate target Lexile range for reading progression"""
    
    if progression_type == "same":
        # Same level: ±100L range around current
        return (max(0, current_lexile - 100), current_lexile + 100)
    
    elif progression_type == "up":
        # Level up: +100-200L higher
        return (current_lexile + 50, current_lexile + 250)
    
    elif progression_type == "down":
        # Step back: 100-200L lower
        return (max(0, current_lexile - 250), max(50, current_lexile - 50))
    
    else:
        return (current_lexile - 100, current_lexile + 100)

def format_lexile_display(lexile_score: Optional[int], confidence: float = 0.0) -> str:
    """Format Lexile score for display with confidence indicator"""
    
    if not lexile_score or pd.isna(lexile_score):
        return ""
    
    lexile_int = int(lexile_score)
    
    # Confidence indicators
    if confidence >= 0.7:
        indicator = "🟢"  # High confidence
    elif confidence >= 0.4:
        indicator = "🟡"  # Medium confidence  
    else:
        indicator = "🟠"  # Low confidence (estimated)
    
    return f"{lexile_int}L {indicator}"

def get_lexile_search_suggestions() -> List[Dict[str, str]]:
    """Get suggested Lexile searches for the UI"""
    
    return [
        {"label": "BR-200L (Ages 3-5)", "query": "lexile 0-200"},
        {"label": "200-400L (Ages 5-7)", "query": "lexile 200-400"}, 
        {"label": "400-600L (Ages 6-8)", "query": "lexile 400-600"},
        {"label": "600-800L (Ages 8-10)", "query": "lexile 600-800"},
        {"label": "800-1000L (Ages 10-12)", "query": "lexile 800-1000"},
        {"label": "1000L+ (Ages 12+)", "query": "lexile 1000+"},
    ]

def analyze_lexile_distribution(df: pd.DataFrame) -> Dict:
    """Analyze Lexile score distribution in the dataset"""
    
    lexile_scores = df['lexile_score'].dropna()
    
    if len(lexile_scores) == 0:
        return {"total": 0, "distribution": {}, "stats": {}}
    
    # Calculate distribution
    distribution = {}
    for (min_score, max_score), label in LEXILE_RANGE_LABELS.items():
        count = len(lexile_scores[(lexile_scores >= min_score) & (lexile_scores < max_score)])
        distribution[label] = count
    
    # Calculate statistics
    stats = {
        "total_books": len(df),
        "books_with_lexile": len(lexile_scores),
        "coverage_percent": (len(lexile_scores) / len(df)) * 100,
        "mean_lexile": float(lexile_scores.mean()),
        "median_lexile": float(lexile_scores.median()),
        "min_lexile": int(lexile_scores.min()),
        "max_lexile": int(lexile_scores.max()),
    }
    
    return {
        "total": len(lexile_scores),
        "distribution": distribution,
        "stats": stats
    }

def validate_lexile_score(score: int) -> bool:
    """Validate if a score is a reasonable Lexile level"""
    return 0 <= score <= 1700  # Typical Lexile range

def lexile_to_grade_estimate(lexile_score: int) -> str:
    """Convert Lexile score to approximate grade level"""
    
    # Standard Lexile to grade mappings
    if lexile_score < 200:
        return "Pre-K to K"
    elif lexile_score < 400:
        return "K to 1st"
    elif lexile_score < 600:
        return "1st to 3rd"
    elif lexile_score < 700:
        return "3rd to 4th"
    elif lexile_score < 800:
        return "4th to 6th"
    elif lexile_score < 900:
        return "6th to 8th"
    elif lexile_score < 1000:
        return "8th to 10th"
    elif lexile_score < 1100:
        return "9th to 11th"
    else:
        return "10th to 12th+"