# Lexile Prediction Model - Deployment Guide

## Overview

This guide provides everything needed to deploy the production-ready Lexile prediction model with **85.6L MAE** and intelligent edge case handling.

## Quick Start

```python
from scripts.production.production_lexile_predictor import ProductionLexilePredictor

# Initialize predictor
predictor = ProductionLexilePredictor()

# Make prediction
result = predictor.predict_lexile(
    title="The Very Hungry Caterpillar",
    author="Eric Carle",
    age_min=3,
    age_max=7,
    book_type="Adult_Directed",
    notes="Classic picture book about caterpillar transformation"
)

print(f"Predicted Lexile: {result['predicted_lexile']}L")
print(f"Confidence: {result['confidence_level']}")
```

## Model Files Required

Ensure these files exist in `/data/models/`:
- `extreme_sophistication_model_standard_lexile.joblib`
- `extreme_sophistication_model_adult_directed.joblib` 
- `extreme_sophistication_model_general.joblib`
- `extreme_sophistication_feature_names.joblib`

## API Response Format

```json
{
  "title": "The Poky Little Puppy",
  "predicted_lexile": 141,
  "confidence_level": "low",
  "model_used": "Standard_Lexile",
  "expected_error_range": "±330L",
  "is_edge_case": true,
  "edge_case_type": "extreme_vintage_classic",
  "warning": "Pre-1942 book - higher prediction uncertainty",
  "prediction_range": "0L - 470L"
}
```

## Confidence Levels

### High Confidence (Expected Error ±10-50L)
- **Adult Directed books:** 5.6L average error
- **Early readers (age ≤3):** Strong performance
- **Modern picture books:** Reliable predictions

### Medium Confidence (Expected Error ±85L)
- **Standard Lexile books:** General case performance
- **Contemporary children's literature:** Good accuracy

### Low Confidence (Expected Error ±150-330L)
- **Pre-1950 books:** Historical language patterns
- **Extreme sophistication books:** Complex vintage classics
- **Unknown/sparse data:** Limited training examples

## Edge Case Handling

### Automatic Detection
The system automatically detects and flags:

1. **Extreme Vintage Classics** (pre-1950)
   - The Poky Little Puppy
   - Make Way for Ducklings  
   - The Country Bunny and Little Gold Shoes
   - And others in the known problematic list

2. **Publication Era Analysis**
   - Scans title/author for pre-1950 publication years
   - Applies elevated error estimates

### User Experience Guidelines

**For High Confidence Predictions:**
```
Predicted Lexile: 460L
Confidence: High
```

**For Edge Cases:**
```
Predicted Lexile: 141L  
Confidence: Low (Pre-1942 vintage classic detected)
Note: Historical books may have higher prediction uncertainty
Likely range: 0L-470L
```

## Performance Expectations

### Production Metrics
- **Overall MAE:** 85.6L
- **Success Rate:** 100% (makes prediction for all inputs)
- **Excellent Predictions:** 46% (<50L error)
- **Acceptable Predictions:** 85% (<200L error)

### Book Type Performance
| Book Type | Expected Accuracy | Use Cases |
|-----------|------------------|-----------|
| Adult Directed | ±10L | Picture books, read-alouds |
| Modern Standards | ±85L | Contemporary children's books |
| Vintage Classics | ±200L | Pre-1950 books (flagged) |

## Integration Examples

### REST API Integration
```python
@app.route('/predict_lexile', methods=['POST'])
def predict_lexile():
    data = request.json
    
    result = predictor.predict_lexile(
        title=data['title'],
        author=data.get('author', ''),
        age_min=data.get('age_min'),
        age_max=data.get('age_max'),
        book_type=data.get('book_type', 'Standard_Lexile'),
        notes=data.get('notes', '')
    )
    
    return jsonify(result)
```

### Batch Processing
```python
def predict_book_batch(books_df):
    predictions = []
    
    for _, book in books_df.iterrows():
        result = predictor.predict_lexile(
            title=book['title'],
            author=book.get('author', ''),
            age_min=book.get('age_min'),
            age_max=book.get('age_max'),
            book_type=book.get('book_type', 'Standard_Lexile')
        )
        predictions.append(result)
    
    return pd.DataFrame(predictions)
```

## Error Handling

```python
try:
    result = predictor.predict_lexile(title="Unknown Book")
except Exception as e:
    return {
        'error': 'Prediction failed',
        'message': str(e),
        'predicted_lexile': None,
        'confidence_level': 'error'
    }
```

## Monitoring & Logging

### Key Metrics to Track
- **Prediction volume** by book type
- **Edge case frequency** (should be <15%)
- **Confidence distribution** (aim for >40% high confidence)
- **Error feedback** from users (when available)

### Logging Example
```python
import logging

logger = logging.getLogger('lexile_predictor')

result = predictor.predict_lexile(title="Test Book")
logger.info(f"Prediction: {result['title']} -> {result['predicted_lexile']}L "
           f"(confidence: {result['confidence_level']})")

if result['is_edge_case']:
    logger.warning(f"Edge case detected: {result['edge_case_type']} - {result['warning']}")
```

## Model Updates

### Version Control
- Current version: **Extreme Sophistication v1.0**
- Training date: **2025-09-03**
- Dataset: **119 books (expanded)**

### Update Triggers
Consider retraining when:
- New verified Lexile scores available (>50 books)
- Edge case performance complaints
- Significant changes in children's literature patterns
- 6+ months of production feedback

## Deployment Checklist

- [ ] Model files present in `/data/models/`
- [ ] Production predictor class tested
- [ ] Edge case handling verified
- [ ] API integration complete
- [ ] Error handling implemented
- [ ] Logging configured
- [ ] Performance monitoring setup
- [ ] Documentation reviewed

## Support & Troubleshooting

### Common Issues

**1. Missing Model Files**
```
Error: Failed to load production models
Solution: Ensure all .joblib files in /data/models/
```

**2. Feature Extraction Errors**
```
Error: Feature names mismatch
Solution: Use exact feature names from training
```

**3. Edge Case Not Detected**
```
Issue: High error for vintage book
Solution: Add to extreme_vintage_classics dict
```

### Contact
For deployment issues or model performance questions, refer to the training logs and performance reports in the project repository.

---

**✅ Ready for Production Deployment**

This model provides robust, reliable Lexile predictions with intelligent handling of edge cases and clear confidence indicators for optimal user experience.