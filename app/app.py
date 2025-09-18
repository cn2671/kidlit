#!/usr/bin/env python3
"""
Setup script for deploying the Reading Level Recommendation App
Run this to set up your complete application
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def setup_app_structure():
    """Create the application directory structure"""
    print("🏗️  Setting up application structure...")
    
    # Create directories
    directories = [
        'app',
        'app/static',
        'app/templates',
        'app/models',
        'app/data'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created directory: {directory}")
    
    return True

def copy_model_files():
    """Copy trained models to app directory"""
    print("\n📊 Copying trained models...")
    
    model_source = "data/models"
    model_dest = "app/models"
    
    if not os.path.exists(model_source):
        print(f"  ❌ Models directory not found: {model_source}")
        print("  Please run the training script first!")
        return False
    
    try:
        # Copy all model files
        for file in os.listdir(model_source):
            if file.endswith('.joblib'):
                shutil.copy2(f"{model_source}/{file}", f"{model_dest}/{file}")
                print(f"  ✓ Copied: {file}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error copying models: {e}")
        return False

def create_requirements_txt():
    """Create requirements.txt for the application"""
    print("\n📦 Creating requirements.txt...")
    
    requirements = """Flask==2.3.3
Flask-CORS==4.0.0
pandas==2.1.1
numpy==1.24.3
scikit-learn==1.3.0
xgboost==1.7.6
joblib==1.3.2
gunicorn==21.2.0
python-dotenv==1.0.0
"""
    
    with open('app/requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("  ✓ Created requirements.txt")
    return True

def create_app_py():
    """Create the main Flask application file"""
    print("\n🚀 Creating main application file...")
    
    app_code = '''#!/usr/bin/env python3
"""
Reading Level Recommendation App - Main Application
"""

import os
import sys
from pathlib import Path

# Add the current directory to the Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Import the Flask backend
from flask_backend import app, reading_api

if __name__ == '__main__':
    print("🚀 Starting BookBuddy Reading Level Recommendation App")
    print("📊 Models loaded:", "✓" if reading_api else "❌")
    print("🌐 Open your browser to: http://localhost:5000")
    print("🔍 API endpoints available at: http://localhost:5000/api/")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5001)
'''
    
    with open('app/app.py', 'w') as f:
        f.write(app_code)
    
    print("  ✓ Created app.py")
    return True

def create_config_file():
    """Create configuration file"""
    print("\n⚙️  Creating configuration...")
    
    config = '''# BookBuddy Configuration
import os
from datetime import timedelta

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Model configuration
    MODEL_TIMESTAMP = "20250831_182131"  # Update with your model timestamp
    MODELS_PATH = "models"
    
    # API configuration
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Logging
    LOG_LEVEL = "INFO"

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
'''
    
    with open('app/config.py', 'w') as f:
        f.write(config)
    
    print("  ✓ Created config.py")
    return True

def create_run_script():
    """Create a simple run script"""
    print("\n🎯 Creating run script...")
    
    run_script = '''#!/bin/bash
# BookBuddy Run Script

echo "🚀 Starting BookBuddy Reading Level Recommendation App"
echo "📍 Current directory: $(pwd)"

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    echo "🐍 Activating virtual environment..."
    source venv/bin/activate
fi

# Install requirements if needed
if [ -f "requirements.txt" ]; then
    echo "📦 Installing requirements..."
    pip install -r requirements.txt
fi

# Run the application
echo "🌐 Starting Flask server..."
echo "📊 Open your browser to: http://localhost:5000"
python app.py
'''
    
    with open('app/run.sh', 'w') as f:
        f.write(run_script)
    
    # Make it executable
    os.chmod('app/run.sh', 0o755)
    
    print("  ✓ Created run.sh")
    return True

def create_html_file():
    """Create the HTML file for the frontend"""
    print("\n🌐 Setting up frontend...")
    
    # The HTML content would be copied from the artifact
    # For now, create a placeholder that references the artifact
    
    html_placeholder = '''<!DOCTYPE html>
<html>
<head>
    <title>BookBuddy - Copy HTML from artifact</title>
</head>
<body>
    <h1>Setup Instructions</h1>
    <p>Please copy the HTML content from the "Kids Book Recommendation App" artifact and save it as app.html in this directory.</p>
    <p>The Flask backend will serve this file at the root URL.</p>
</body>
</html>'''
    
    with open('app/app.html', 'w') as f:
        f.write(html_placeholder)
    
    print("  ✓ Created app.html placeholder")
    print("  ⚠️  Please copy the HTML from the artifact to replace this file!")
    return True

def update_model_timestamp():
    """Update the model timestamp in the backend code"""
    print("\n🔧 Updating model configuration...")
    
    # Try to find the latest model timestamp
    model_dir = "data/models"
    if os.path.exists(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.startswith('ensemble_') and f.endswith('.joblib')]
        if model_files:
            # Extract timestamp from filename
            latest_model = sorted(model_files)[-1]
            timestamp = latest_model.replace('ensemble_', '').replace('.joblib', '')
            
            print(f"  ✓ Found model timestamp: {timestamp}")
            
            # Update the backend code (this would need actual file modification)
            print(f"  ⚠️  Please update MODEL_TIMESTAMP in config.py to: {timestamp}")
            return timestamp
    
    print("  ⚠️  Could not find model files. Using default timestamp.")
    return "20250831_182131"

def create_readme():
    """Create README with setup instructions"""
    print("\n📖 Creating README...")
    
    readme = '''# BookBuddy - AI-Powered Reading Level Recommendations

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
'''
    
    with open('app/README.md', 'w') as f:
        f.write(readme)
    
    print("  ✓ Created README.md")
    return True

def main():
    """Main setup function"""
    print("🚀 BookBuddy App Deployment Setup")
    print("=" * 50)
    
    # Run setup steps
    steps = [
        ("Setting up directory structure", setup_app_structure),
        ("Copying model files", copy_model_files),
        ("Creating requirements.txt", create_requirements_txt),
        ("Creating main app file", create_app_py),
        ("Creating configuration", create_config_file),
        ("Creating run script", create_run_script),
        ("Setting up frontend", create_html_file),
        ("Creating documentation", create_readme)
    ]
    
    success_count = 0
    
    for step_name, step_function in steps:
        try:
            result = step_function()
            if result:
                success_count += 1
        except Exception as e:
            print(f"  ❌ Error in {step_name}: {e}")
    
    # Update model timestamp
    timestamp = update_model_timestamp()
    
    print("\n" + "=" * 50)
    print(f"🎉 Setup completed: {success_count}/{len(steps)} steps successful")
    print("\n📋 Next steps:")
    print("1. Copy HTML from the 'Kids Book Recommendation App' artifact to app/app.html")
    print("2. Copy Python code from the 'Flask Backend API' artifact to app/flask_backend.py")  
    print("3. cd app && python app.py")
    print("4. Open http://localhost:5000 in your browser")
    print("\n✨ Your reading level recommendation app is ready!")
    
if __name__ == "__main__":
    main()