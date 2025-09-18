#!/usr/bin/env python3
"""
Project cleanup script - removes intermediate/obsolete files while keeping production essentials
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]

def cleanup_project():
    """Clean up unnecessary files while keeping production essentials"""
    
    print("🧹 CLEANING UP PROJECT FILES")
    print("=" * 50)
    
    # Files to keep (essential for production)
    essential_files = {
        # Data files
        "data/lexile_collection_tracking_expanded.csv",  # Final training dataset
        "data/additional_test_scores.csv",               # Test cases
        "data/additional_pre1950_lexile_scores.csv",     # Pre-1950 research data
        
        # Model files  
        "data/models/extreme_sophistication_model_standard_lexile.joblib",
        "data/models/extreme_sophistication_model_adult_directed.joblib", 
        "data/models/extreme_sophistication_model_general.joblib",
        "data/models/extreme_sophistication_feature_names.joblib",
        
        # Core production scripts
        "scripts/production/production_lexile_predictor.py",
        "scripts/core/recommender.py",
        "scripts/core/lexile_utils.py",
        "scripts/core/config.py",
        
        # Final model training/testing
        "scripts/modeling/extreme_sophistication_predictor.py",
        "scripts/testing/test_extreme_sophistication_model.py",
        
        # Documentation
        "MODEL_PERFORMANCE_REPORT.md",
        "DEPLOYMENT_GUIDE.md",
        "README.md",
        "CLAUDE.md"
    }
    
    # Convert to absolute paths
    essential_paths = {ROOT / path for path in essential_files}
    
    # Files/folders to remove (intermediate/obsolete)
    cleanup_targets = []
    
    # 1. Remove intermediate CSV files (keep only essentials)
    data_dir = ROOT / "data"
    if data_dir.exists():
        for csv_file in data_dir.glob("*.csv"):
            if csv_file not in essential_paths:
                cleanup_targets.append(csv_file)
    
    # 2. Remove intermediate Python scripts
    scripts_to_remove = [
        # Data processing (one-time use)
        "scripts/data_processing/collect_verified_lexile_scores.py",
        "scripts/data_processing/merge_additional_lexile_scores.py", 
        "scripts/data_processing/update_lexile_collection.py",
        "scripts/data_processing/prefix_aware_analysis.py",
        "scripts/data_processing/feature_engineering.py",
        "scripts/data_processing/update_second_batch.py",
        "scripts/data_processing/update_test_books.py", 
        "scripts/data_processing/update_third_batch.py",
        "scripts/data_processing/integrate_pre1950_books.py",
        
        # Data collection (completed)
        "scripts/data_collection/age_data_collector.py",
        "scripts/data_collection/manual_age_enrichment.py",
        
        # Analysis (one-time)
        "scripts/analysis/final_prediction_summary.py",
        
        # Obsolete modeling attempts
        "scripts/modeling/fixed_ad_predictor.py",
        "scripts/modeling/age_based_predictor.py",
        "scripts/modeling/enhanced_predictor.py",
        "scripts/modeling/flesch_kincaid_validator.py",
        
        # Old testing scripts
        "scripts/testing/test_comprehensive_model.py",
        "scripts/testing/test_fixed_ad_model.py",
        "scripts/testing/test_age_based_model.py",
        "scripts/testing/test_enhanced_model.py"
    ]
    
    for script_path in scripts_to_remove:
        full_path = ROOT / script_path
        if full_path.exists():
            cleanup_targets.append(full_path)
    
    # 3. Remove empty or unnecessary directories
    dirs_to_check = [
        "scripts/data_collection",
        "scripts/data_processing", 
        "scripts/analysis",
        "data/raw",
        "data/processed"
    ]
    
    # 4. Remove pipelines directory (completed work)
    pipelines_dir = ROOT / "scripts/pipelines"
    if pipelines_dir.exists():
        cleanup_targets.append(pipelines_dir)
    
    # Execute cleanup
    removed_count = 0
    kept_count = 0
    
    print("\n🗑️  REMOVING FILES:")
    for target in cleanup_targets:
        try:
            if target.is_file():
                target.unlink()
                print(f"   ✅ Removed: {target.relative_to(ROOT)}")
                removed_count += 1
            elif target.is_dir():
                shutil.rmtree(target)
                print(f"   ✅ Removed directory: {target.relative_to(ROOT)}")
                removed_count += 1
        except Exception as e:
            print(f"   ❌ Failed to remove {target.relative_to(ROOT)}: {e}")
    
    # Remove empty directories
    print(f"\n📁 CHECKING EMPTY DIRECTORIES:")
    for dir_path in dirs_to_check:
        full_dir = ROOT / dir_path
        if full_dir.exists() and not any(full_dir.iterdir()):
            try:
                full_dir.rmdir()
                print(f"   ✅ Removed empty directory: {dir_path}")
                removed_count += 1
            except Exception as e:
                print(f"   ❌ Failed to remove {dir_path}: {e}")
    
    # Count kept files
    print(f"\n📋 ESSENTIAL FILES KEPT:")
    for essential_path in sorted(essential_paths):
        if essential_path.exists():
            print(f"   ✅ {essential_path.relative_to(ROOT)}")
            kept_count += 1
        else:
            print(f"   ⚠️  Missing: {essential_path.relative_to(ROOT)}")
    
    # Final summary
    print(f"\n📊 CLEANUP SUMMARY:")
    print(f"   🗑️  Files/directories removed: {removed_count}")
    print(f"   📁 Essential files kept: {kept_count}")
    
    # Show final directory structure
    print(f"\n📂 CLEAN PROJECT STRUCTURE:")
    show_tree(ROOT, max_depth=3)
    
    print(f"\n✅ Project cleanup complete!")

def show_tree(directory, max_depth=2, current_depth=0, prefix=""):
    """Show clean directory tree"""
    if current_depth > max_depth:
        return
        
    items = sorted([item for item in directory.iterdir() 
                   if not item.name.startswith('.') and not item.name.startswith('__pycache__')])
    
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        current_prefix = "└── " if is_last else "├── "
        print(f"{prefix}{current_prefix}{item.name}")
        
        if item.is_dir() and current_depth < max_depth:
            extension = "    " if is_last else "│   "
            show_tree(item, max_depth, current_depth + 1, prefix + extension)

if __name__ == "__main__":
    cleanup_project()