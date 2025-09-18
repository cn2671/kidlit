#!/bin/bash
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
