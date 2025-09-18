# BookBuddy - AI-Powered Reading Level Recommendations

## Quick Start

1. **Copy the required files:**
   ```bash
   # Copy the HTML content from the "Kids Book Recommendation App" artifact
   # and save it as app/app.html
   
   # Copy the Flask backend code from the "Flask Backend API" artifact  
   # and save it as app/flask_backend.py
   ```

2. **Install dependencies:**
   ```bash
   cd app
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   # OR
   ./run.sh
   ```

4. **Open your browser:**
   ```
   http://localhost:5000
   ```

## Features

### 🔍 Book Analysis
- Enter any book title and author
- Get AI-powered reading level predictions
- View Lexile score, age range, and confidence level
- See which prediction tier was used

### 🎯 Personalized Recommendations  
- Input your child's age and interests
- Get curated book recommendations
- Adjust difficulty level (easier/same/harder)
- Browse books by reading level category

### 📊 Powered by Your Trained Models
- Random Forest ensemble models
- 95%+ accuracy on age categories
- ~28 Lexile point average error
- Tiered confidence system for reliability

## API Endpoints

### Predict Reading Level
```
POST /api/predict
{
  "title": "Book Title",
  "author": "Author Name", 
  "themes": "friendship, adventure",
  "description": "Book description..."
}
```

### Get Recommendations
```
POST /api/recommendations
{
  "age": "8",
  "readingLevel": "beginning",
  "interests": "magic, friendship",
  "challenge": "same"
}
```

### Health Check
```
GET /api/health
```

## File Structure
```
app/
├── app.py              # Main application
├── flask_backend.py    # Flask API backend  
├── app.html           # Frontend interface
├── config.py          # Configuration
├── requirements.txt   # Dependencies
├── models/           # Trained ML models
├── run.sh           # Run script
└── README.md        # This file
```

## Model Performance
- **Lexile Prediction**: 27.9 MAE (Mean Absolute Error)
- **Category Classification**: 95.2% accuracy
- **Cross-validation**: Consistent performance across folds
- **Tiered System**: 94-98% accuracy depending on confidence tier

## Customization

### Adding New Authors
Update the `author_stats` dictionary in `flask_backend.py` with:
- Book count
- Average Lexile level  
- Standard deviation
- Primary age category

### Expanding Book Database
Modify the `SAMPLE_BOOKS` dictionary in `flask_backend.py` to add more book recommendations.

### Model Updates
To use newly trained models:
1. Update `MODEL_TIMESTAMP` in `config.py`
2. Copy new model files to `app/models/`
3. Restart the application

## Production Deployment

### Using Gunicorn
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Environment Variables
- `FLASK_ENV=production`
- `SECRET_KEY=your-secret-key`
- `MODEL_TIMESTAMP=your-model-timestamp`

## Troubleshooting

### Models Not Loading
- Check that model files exist in `app/models/`
- Verify the timestamp matches your trained models
- Ensure all required dependencies are installed

### Prediction Errors
- Verify input data format
- Check server logs for detailed error messages
- Test with known working book examples

## Support
Built using your trained ML models from the reading level assignment strategy. 
For issues, check the Flask logs and ensure all model files are properly copied.
