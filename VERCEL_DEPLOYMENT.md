# 🚀 KidLit Curator - Vercel Deployment Guide

## 📋 Branch Structure

**Three branches for different purposes:**
- `main` - Complete development environment with all files
- `production-webapp-v1` - Full project with webapp + analysis tools
- `production-deploy` - **Lean deployment branch (RECOMMENDED FOR VERCEL)**

## 🚀 Quick Deploy Steps

### 1. Push to GitHub
```bash
git push origin production-deploy
```

### 2. Deploy to Vercel
1. **Sign up**: Go to [vercel.com](https://vercel.com) and sign up with your GitHub account
2. **Import Project**: Click "Import Project" and select your `kidlit` repository
3. **Select Branch**: Choose the `production-deploy` branch ⭐
4. **Configure**:
   - Framework Preset: **Other**
   - Root Directory: Leave as default (root)
   - Build Command: Leave empty
   - Output Directory: Leave empty
5. **Deploy**: Click "Deploy" - it will take 2-3 minutes

### 3. Your Live URL
After deployment, you'll get a URL like:
- `https://kidlit-curator-[your-username].vercel.app`
- You can also get a custom domain later

## 📦 Production-Deploy Branch Structure
**Optimized for fast, lean deployment:**
```
kidlit/ (production-deploy branch)
├── app/
│   ├── flask_backend.py           # Main Flask application
│   ├── app.html                  # Frontend interface
│   ├── hybrid_query_parser.py    # Smart search parser
│   ├── production_lexile_predictor.py # ML predictor
│   └── models/                   # All ML models (15+ files)
├── data/
│   ├── books_final_complete.csv  # Complete book catalog
│   ├── enriched_lexile_scores.csv # Enhanced predictions
│   └── age_model.joblib          # Age classification model
├── requirements.txt              # Minimal Flask dependencies
├── vercel.json                  # Vercel configuration
├── VERCEL_DEPLOYMENT.md         # This guide
└── .gitignore                   # Excludes system files
```

**What's removed for deployment:**
- Analysis scripts and notebooks
- Streamlit app dependencies
- Research and development files
- Extra data processing tools

## What's Included
✅ **Flask Backend** - Complete API with ML integration
✅ **Book Catalog** - 1,087 curated children's books
✅ **ML Models** - Lexile prediction and similarity engine
✅ **Responsive UI** - Mobile-friendly interface
✅ **Production Ready** - Optimized for deployment

## Features Available
- 🔍 **Smart Search** - Natural language book queries
- 📊 **Lexile Predictions** - AI-powered reading level analysis
- 🎯 **Similar Books** - ML-driven recommendations
- 📱 **Reading Progress** - Track favorites, read, and skipped books
- 🎨 **Beautiful UI** - Clean, modern interface

## Sharing Your App
Perfect for:
- **Resume/Portfolio** - Shows full-stack ML development
- **Friends & Family** - Let them discover great books
- **Professional Demo** - Showcase data science skills

## Environment
- **Python 3.9+** with Flask, scikit-learn, pandas
- **Serverless** deployment on Vercel
- **No database required** - uses CSV data files
- **Auto-scaling** - handles traffic spikes automatically

---
🎉 **You're ready to deploy!** Your KidLit Curator will be live in minutes.