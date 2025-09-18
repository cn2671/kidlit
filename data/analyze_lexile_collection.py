
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def analyze_collected_lexile_scores():
    """Analyze collected verified Lexile scores"""
    
    # Load collection data
    df = pd.read_csv('data/lexile_collection_tracking.csv')
    
    # Filter completed entries
    completed = df[
        (df['status'] == 'Complete') & 
        (df['verified_lexile'] != '') &
        (df['verified_lexile'].notna())
    ].copy()
    
    if len(completed) == 0:
        print("❌ No completed collections found yet!")
        print("📋 Continue collecting verified Lexile scores")
        return
    
    print(f"📊 ANALYZING {len(completed)} VERIFIED LEXILE SCORES")
    print(f"{'='*50}")
    
    # Convert to numeric
    completed['verified_lexile'] = pd.to_numeric(completed['verified_lexile'], errors='coerce')
    completed['current_ml_estimate'] = pd.to_numeric(completed['current_ml_estimate'], errors='coerce')
    
    # Remove any conversion errors
    valid = completed.dropna(subset=['verified_lexile', 'current_ml_estimate'])
    
    if len(valid) < 5:
        print(f"⚠️  Only {len(valid)} valid comparisons - need more data")
        return
    
    # Calculate accuracy metrics
    verified_scores = valid['verified_lexile'].values
    ml_scores = valid['current_ml_estimate'].values
    
    mae = mean_absolute_error(verified_scores, ml_scores)
    rmse = np.sqrt(np.mean((verified_scores - ml_scores) ** 2))
    r2 = r2_score(verified_scores, ml_scores)
    
    print(f"🎯 MODEL ACCURACY vs VERIFIED SCORES:")
    print(f"  Mean Absolute Error: {mae:.1f} Lexile points")
    print(f"  Root Mean Square Error: {rmse:.1f} Lexile points")  
    print(f"  R² Score: {r2:.3f}")
    
    # Error distribution analysis
    errors = np.abs(ml_scores - verified_scores)
    
    print(f"\n📈 ERROR DISTRIBUTION:")
    print(f"  Excellent (≤100L): {sum(errors <= 100)}/{len(errors)} ({100*sum(errors <= 100)/len(errors):.1f}%)")
    print(f"  Good (≤200L): {sum(errors <= 200)}/{len(errors)} ({100*sum(errors <= 200)/len(errors):.1f}%)")
    print(f"  Acceptable (≤300L): {sum(errors <= 300)}/{len(errors)} ({100*sum(errors <= 300)/len(errors):.1f}%)")
    print(f"  Poor (>300L): {sum(errors > 300)}/{len(errors)} ({100*sum(errors > 300)/len(errors):.1f}%)")
    
    # Show worst predictions for analysis
    valid['error'] = errors
    worst = valid.nlargest(5, 'error')[['title', 'author', 'verified_lexile', 'current_ml_estimate', 'error']]
    
    print(f"\n❌ LARGEST ERRORS (for investigation):")
    for _, row in worst.iterrows():
        print(f"  {row['title']}: Verified {row['verified_lexile']}L vs ML {row['current_ml_estimate']}L (error: {row['error']:.0f}L)")
    
    # Show best predictions
    best = valid.nsmallest(5, 'error')[['title', 'author', 'verified_lexile', 'current_ml_estimate', 'error']]
    
    print(f"\n✅ BEST PREDICTIONS:")
    for _, row in best.iterrows():
        print(f"  {row['title']}: Verified {row['verified_lexile']}L vs ML {row['current_ml_estimate']}L (error: {row['error']:.0f}L)")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    if mae < 100:
        print(f"  ✅ Excellent model performance - ready for production!")
    elif mae < 200:
        print(f"  ⭐ Good model performance - minor improvements possible")
    elif mae < 300:
        print(f"  ⚠️  Moderate performance - consider model retraining")
    else:
        print(f"  ❌ Poor performance - model needs significant improvement")
    
    print(f"\n📊 Next steps:")
    print(f"  • If MAE < 200L: Model is ready for use")
    print(f"  • If MAE > 200L: Collect more verified scores and retrain")
    print(f"  • Target: 100+ verified scores for comprehensive retraining")
    
    return valid, mae, r2

if __name__ == "__main__":
    analyze_collected_lexile_scores()
