import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import sys

def extract_clear_features():
    """
    Extract training features from CLEAR corpus for Lexile estimation model
    """
    # Load CLEAR corpus data
    clear_path = '/Users/chaerinnoh/Desktop/kidlit/data/external/clear_corpus.csv'
    df = pd.read_csv(clear_path)
    
    print(f"Loaded CLEAR corpus with {len(df)} passages")
    print(f"Columns: {list(df.columns)}")
    
    # Create numeric Lexile targets from Lexile Band
    def convert_lexile_band(band):
        """Convert lexile band to numeric value"""
        if pd.isna(band):
            return np.nan
        
        band = str(band).strip()
        
        # Handle simple numeric bands (900, 700, etc.)
        if band.isdigit():
            return int(band)
        
        # Handle range formats like "410L-600L", "410L - 600L"
        if 'L-' in band or 'L -' in band:
            # Extract the middle of the range
            parts = band.replace(' ', '').replace('L-', '-').replace('L', '').split('-')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                return (int(parts[0]) + int(parts[1])) // 2
        
        # Handle special cases
        if band == 'BR':
            return 0
        
        return np.nan
    
    df['lexile_numeric'] = df['Lexile Band'].apply(convert_lexile_band)
    
    # Select readability features (all numeric readability metrics)
    feature_columns = [
        'Flesch-Kincaid-Grade-Level',
        'Flesch-Reading-Ease',
        'Automated Readability Index',
        'SMOG Readability',
        'New Dale-Chall Readability Formula',
        'Google WC',
        'Sentence Count',
        'Paragraphs'
    ]
    
    # Check for missing values
    print("\nMissing values by column:")
    missing_counts = df[['lexile_numeric'] + feature_columns].isnull().sum()
    print(missing_counts)
    
    # Check lexile band values
    print(f"\nUnique Lexile Bands: {df['Lexile Band'].unique()}")
    print(f"Lexile Band value counts:")
    print(df['Lexile Band'].value_counts())
    
    # Clean data - remove rows with missing lexile targets or features
    df_clean = df.dropna(subset=['lexile_numeric'] + feature_columns)
    print(f"After cleaning: {len(df_clean)} passages")
    
    # Extract features and targets
    X = df_clean[feature_columns].values
    y = df_clean['lexile_numeric'].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create feature names for reference
    feature_names = feature_columns
    
    print("\nFeature statistics:")
    feature_stats = pd.DataFrame({
        'feature': feature_names,
        'mean': np.mean(X, axis=0),
        'std': np.std(X, axis=0),
        'min': np.min(X, axis=0),
        'max': np.max(X, axis=0)
    })
    print(feature_stats)
    
    print(f"\nTarget (Lexile) statistics:")
    print(f"Mean: {np.mean(y):.1f}")
    print(f"Std: {np.std(y):.1f}")
    print(f"Range: {np.min(y):.0f} - {np.max(y):.0f}")
    
    # Save processed data
    output_dir = '/Users/chaerinnoh/Desktop/kidlit/data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save features and targets
    np.save(os.path.join(output_dir, 'clear_features.npy'), X_scaled)
    np.save(os.path.join(output_dir, 'clear_targets.npy'), y)
    
    # Save feature names and scaler
    pd.Series(feature_names).to_csv(os.path.join(output_dir, 'feature_names.csv'), index=False, header=['feature'])
    
    # Save scaler parameters
    scaler_params = pd.DataFrame({
        'feature': feature_names,
        'mean': scaler.mean_,
        'scale': scaler.scale_
    })
    scaler_params.to_csv(os.path.join(output_dir, 'scaler_params.csv'), index=False)
    
    # Save unique lexile conversions for reference
    lexile_conversions = df[['Lexile Band', 'lexile_numeric']].drop_duplicates().sort_values('lexile_numeric')
    lexile_conversions.to_csv(os.path.join(output_dir, 'lexile_conversions.csv'), index=False)
    
    print(f"\nFeatures saved to: {output_dir}")
    print(f"Feature matrix shape: {X_scaled.shape}")
    print(f"Target vector shape: {y.shape}")
    
    return X_scaled, y, feature_names, scaler

if __name__ == "__main__":
    X, y, features, scaler = extract_clear_features()