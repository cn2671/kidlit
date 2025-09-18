#!/usr/bin/env python3
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
