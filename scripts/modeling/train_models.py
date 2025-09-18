import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ReadingLevelModelTrainer:
    """
    Model training pipeline for reading level prediction
    Implements the tiered assignment system from the strategy document
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        self.label_encoders = {}
        
    def load_processed_data(self, features_path='data/processed/modeling_features.csv', 
                          full_data_path='data/processed/books_engineered_features.csv'):
        """Load the engineered features and prepare for modeling"""
        print("Loading processed data...")
        
        # Load feature matrix
        if isinstance(features_path, str):
            self.X = pd.read_csv(features_path)
        else:
            self.X = features_path
            
        # Load full dataset for targets
        if isinstance(full_data_path, str):
            self.df = pd.read_csv(full_data_path)
        else:
            self.df = full_data_path
            
        # Extract targets
        self.y_lexile = self.df['target_lexile'] if 'target_lexile' in self.df.columns else self.df['lexile_score']
        self.y_category = self.df['target_age_category'] if 'target_age_category' in self.df.columns else self.df['reading_level_llm']
        
        print(f"Loaded {len(self.X)} books with {self.X.shape[1]} features")
        print(f"Lexile targets: {self.y_lexile.notna().sum()} valid")
        print(f"Category targets: {(self.y_category != 'Unknown').sum() if 'Unknown' in self.y_category.values else len(self.y_category)} valid")
        
        return self.X, self.y_lexile, self.y_category
    
    def create_modeling_datasets(self):
        """Create separate datasets for regression and classification"""
        print("Creating modeling datasets...")
        
        # Ensure X and y have matching indices
        common_indices = self.X.index.intersection(self.y_lexile.index).intersection(self.y_category.index)
        self.X = self.X.loc[common_indices]
        self.y_lexile = self.y_lexile.loc[common_indices]
        self.y_category = self.y_category.loc[common_indices]
        
        print(f"After index alignment: {len(self.X)} samples")
        
        # Dataset 1: Lexile Regression (books with valid lexile scores)
        lexile_mask = self.y_lexile.notna()
        self.X_lexile = self.X.loc[lexile_mask].copy()
        self.y_lexile_clean = self.y_lexile.loc[lexile_mask].copy()
        
        # Verify alignment
        assert len(self.X_lexile) == len(self.y_lexile_clean), f"Lexile length mismatch: {len(self.X_lexile)} vs {len(self.y_lexile_clean)}"
        
        # Dataset 2: Category Classification (books with known categories)
        if 'Unknown' in self.y_category.values:
            category_mask = self.y_category != 'Unknown'
        else:
            category_mask = self.y_category.notna()
        
        self.X_category = self.X.loc[category_mask].copy()
        self.y_category_clean = self.y_category.loc[category_mask].copy()
        
        # Verify alignment
        assert len(self.X_category) == len(self.y_category_clean), f"Category length mismatch: {len(self.X_category)} vs {len(self.y_category_clean)}"
        
        # Encode category labels
        le_category = LabelEncoder()
        self.y_category_encoded = le_category.fit_transform(self.y_category_clean.astype(str))
        self.label_encoders['category'] = le_category
        
        print(f"Lexile dataset: {len(self.X_lexile)} samples")
        print(f"Category dataset: {len(self.X_category)} samples")
        print(f"Category distribution: {dict(zip(*np.unique(self.y_category_clean, return_counts=True)))}")
        
        return True
    
    def create_train_test_splits(self, test_size=0.2):
        """Create train/test splits for both tasks"""
        print("Creating train/test splits...")
        
        # Lexile regression splits
        self.X_train_lex, self.X_test_lex, self.y_train_lex, self.y_test_lex = train_test_split(
            self.X_lexile, self.y_lexile_clean, 
            test_size=test_size, random_state=self.random_state
        )
        
        # Category classification splits (stratified)
        self.X_train_cat, self.X_test_cat, self.y_train_cat, self.y_test_cat = train_test_split(
            self.X_category, self.y_category_encoded,
            test_size=test_size, stratify=self.y_category_encoded, 
            random_state=self.random_state
        )
        
        print(f"Lexile splits - Train: {len(self.X_train_lex)}, Test: {len(self.X_test_lex)}")
        print(f"Category splits - Train: {len(self.X_train_cat)}, Test: {len(self.X_test_cat)}")
        
        return True
    
    def train_lexile_models(self):
        """Train regression models for Lexile score prediction"""
        print("\n=== TRAINING LEXILE REGRESSION MODELS ===")
        
        # Model 1: Random Forest Regressor
        print("Training Random Forest Regressor...")
        rf_reg = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf_reg.fit(self.X_train_lex, self.y_train_lex)
        
        # Model 2: XGBoost Regressor
        print("Training XGBoost Regressor...")
        xgb_reg = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1
        )
        xgb_reg.fit(self.X_train_lex, self.y_train_lex)
        
        # Model 3: Gradient Boosting Regressor
        print("Training Gradient Boosting Regressor...")
        gb_reg = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=self.random_state
        )
        gb_reg.fit(self.X_train_lex, self.y_train_lex)
        
        # Store models
        self.models['lexile'] = {
            'random_forest': rf_reg,
            'xgboost': xgb_reg,
            'gradient_boosting': gb_reg
        }
        
        # Evaluate models
        lexile_performance = {}
        for name, model in self.models['lexile'].items():
            train_pred = model.predict(self.X_train_lex)
            test_pred = model.predict(self.X_test_lex)
            
            performance = {
                'train_mae': mean_absolute_error(self.y_train_lex, train_pred),
                'test_mae': mean_absolute_error(self.y_test_lex, test_pred),
                'train_rmse': np.sqrt(mean_squared_error(self.y_train_lex, train_pred)),
                'test_rmse': np.sqrt(mean_squared_error(self.y_test_lex, test_pred)),
                'test_predictions': test_pred
            }
            lexile_performance[name] = performance
            
            print(f"{name.title()} - Test MAE: {performance['test_mae']:.1f}, Test RMSE: {performance['test_rmse']:.1f}")
        
        self.performance_metrics['lexile'] = lexile_performance
        
        # Feature importance for best model
        best_model_name = min(lexile_performance.keys(), 
                            key=lambda x: lexile_performance[x]['test_mae'])
        best_model = self.models['lexile'][best_model_name]
        
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.X_train_lex.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.feature_importance['lexile'] = feature_importance
            print(f"\nTop 10 features for {best_model_name}:")
            print(feature_importance.head(10).to_string(index=False))
        
        return lexile_performance
    
    def train_category_models(self):
        """Train classification models for age category prediction"""
        print("\n=== TRAINING CATEGORY CLASSIFICATION MODELS ===")
        
        # Model 1: Random Forest Classifier
        print("Training Random Forest Classifier...")
        rf_clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',  # Handle class imbalance
            random_state=self.random_state,
            n_jobs=-1
        )
        rf_clf.fit(self.X_train_cat, self.y_train_cat)
        
        # Model 2: XGBoost Classifier
        print("Training XGBoost Classifier...")
        xgb_clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1
        )
        xgb_clf.fit(self.X_train_cat, self.y_train_cat)
        
        # Store models
        self.models['category'] = {
            'random_forest': rf_clf,
            'xgboost': xgb_clf
        }
        
        # Evaluate models
        category_performance = {}
        for name, model in self.models['category'].items():
            train_pred = model.predict(self.X_train_cat)
            test_pred = model.predict(self.X_test_cat)
            
            performance = {
                'train_accuracy': accuracy_score(self.y_train_cat, train_pred),
                'test_accuracy': accuracy_score(self.y_test_cat, test_pred),
                'test_predictions': test_pred,
                'classification_report': classification_report(
                    self.y_test_cat, test_pred, 
                    target_names=self.label_encoders['category'].classes_,
                    output_dict=True
                )
            }
            category_performance[name] = performance
            
            print(f"{name.title()} - Test Accuracy: {performance['test_accuracy']:.3f}")
        
        self.performance_metrics['category'] = category_performance
        
        # Feature importance for best model
        best_model_name = max(category_performance.keys(), 
                            key=lambda x: category_performance[x]['test_accuracy'])
        best_model = self.models['category'][best_model_name]
        
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.X_train_cat.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.feature_importance['category'] = feature_importance
            print(f"\nTop 10 features for {best_model_name}:")
            print(feature_importance.head(10).to_string(index=False))
        
        return category_performance
    
    def cross_validate_models(self, cv_folds=5):
        """Perform cross-validation for model validation"""
        print("\n=== CROSS-VALIDATION ===")
        
        cv_results = {}
        
        # Cross-validate lexile models
        print("Cross-validating Lexile models...")
        for name, model in self.models['lexile'].items():
            cv_scores = cross_val_score(
                model, self.X_lexile, self.y_lexile_clean, 
                cv=cv_folds, scoring='neg_mean_absolute_error', n_jobs=-1
            )
            cv_results[f'lexile_{name}'] = {
                'mean_mae': -cv_scores.mean(),
                'std_mae': cv_scores.std(),
                'scores': -cv_scores
            }
            print(f"Lexile {name}: MAE = {-cv_scores.mean():.1f} ± {cv_scores.std():.1f}")
        
        # Cross-validate category models
        print("Cross-validating Category models...")
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for name, model in self.models['category'].items():
            cv_scores = cross_val_score(
                model, self.X_category, self.y_category_encoded, 
                cv=skf, scoring='accuracy', n_jobs=-1
            )
            cv_results[f'category_{name}'] = {
                'mean_accuracy': cv_scores.mean(),
                'std_accuracy': cv_scores.std(),
                'scores': cv_scores
            }
            print(f"Category {name}: Accuracy = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        self.cv_results = cv_results
        return cv_results
    
    def create_ensemble_model(self):
        """Create ensemble model combining best performers"""
        print("\n=== CREATING ENSEMBLE MODEL ===")
        
        # Select best models based on performance
        best_lexile_name = min(self.performance_metrics['lexile'].keys(), 
                             key=lambda x: self.performance_metrics['lexile'][x]['test_mae'])
        best_category_name = max(self.performance_metrics['category'].keys(), 
                               key=lambda x: self.performance_metrics['category'][x]['test_accuracy'])
        
        self.ensemble = {
            'lexile_model': self.models['lexile'][best_lexile_name],
            'category_model': self.models['category'][best_category_name],
            'lexile_model_name': best_lexile_name,
            'category_model_name': best_category_name
        }
        
        print(f"Ensemble uses: Lexile={best_lexile_name}, Category={best_category_name}")
        return self.ensemble
    
    def implement_tiered_assignment(self, confidence_threshold_high=0.8, confidence_threshold_medium=0.6):
        """Implement the tiered assignment system from strategy document"""
        print("\n=== IMPLEMENTING TIERED ASSIGNMENT SYSTEM ===")
        
        # Create tiered predictions for test set
        tiered_results = []
        
        # For demonstration, use the category test set
        X_test = self.X_test_cat
        y_test_actual = self.y_test_cat  # This is a numpy array
        
        for idx in range(len(X_test)):
            sample = X_test.iloc[idx:idx+1]
            
            # Get confidence scores (assuming we have them in the data)
            avg_confidence = sample.get('avg_confidence', pd.Series([0.7])).iloc[0]
            
            # Tier assignment logic
            if avg_confidence >= confidence_threshold_high:
                tier = "Tier 1: High Confidence"
                # Use best model - need to match sample features to lexile model
                if hasattr(self, 'ensemble') and self.ensemble:
                    # Find matching sample in lexile test set
                    matching_lexile_sample = self._find_matching_sample(sample, self.X_test_lex)
                    if matching_lexile_sample is not None:
                        lexile_pred = self.ensemble['lexile_model'].predict(matching_lexile_sample)[0]
                    else:
                        lexile_pred = 400  # Default if no match
                    category_pred = self.ensemble['category_model'].predict(sample)[0]
                else:
                    lexile_pred = 400
                    category_pred = 1
            elif avg_confidence >= confidence_threshold_medium:
                tier = "Tier 2: Medium Confidence"
                # Use ensemble average
                if hasattr(self, 'ensemble') and self.ensemble:
                    matching_lexile_sample = self._find_matching_sample(sample, self.X_test_lex)
                    if matching_lexile_sample is not None and len(self.models['lexile']) > 0:
                        lexile_predictions = []
                        for model in self.models['lexile'].values():
                            lexile_predictions.append(model.predict(matching_lexile_sample)[0])
                        lexile_pred = np.mean(lexile_predictions) if lexile_predictions else 400
                    else:
                        lexile_pred = 400
                    category_pred = self.ensemble['category_model'].predict(sample)[0]
                else:
                    lexile_pred = 400
                    category_pred = 1
            else:
                tier = "Tier 3: Conservative Estimate"
                # Use conservative estimates
                lexile_pred = 400  # Conservative middle value
                category_pred = 1  # Beginning readers (conservative)
            
            tiered_results.append({
                'tier': tier,
                'confidence': avg_confidence,
                'lexile_prediction': lexile_pred,
                'category_prediction': category_pred,
                'actual_category': y_test_actual[idx]  # Use array indexing instead of .iloc
            })
        
        self.tiered_results = pd.DataFrame(tiered_results)
        
        # Analyze tier distribution
        print("Tier Distribution:")
        print(self.tiered_results['tier'].value_counts())
        
        # Accuracy by tier
        print("\nAccuracy by Tier:")
        for tier in self.tiered_results['tier'].unique():
            tier_data = self.tiered_results[self.tiered_results['tier'] == tier]
            accuracy = (tier_data['category_prediction'] == tier_data['actual_category']).mean()
            print(f"{tier}: {accuracy:.3f}")
        
        return self.tiered_results
    
    def _find_matching_sample(self, sample, target_dataset):
        """Find a matching sample in the target dataset (helper for tiered assignment)"""
        if len(target_dataset) == 0:
            return None
        
        # For simplicity, just return the first sample from target dataset
        # In production, you'd want proper matching logic
        return target_dataset.iloc[0:1]
    
    def save_models(self, save_dir='data/models/'):
        """Save trained models and metadata"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save models
        for task in ['lexile', 'category']:
            for name, model in self.models[task].items():
                filename = f"{save_dir}{task}_{name}_{timestamp}.joblib"
                joblib.dump(model, filename)
                print(f"Saved {filename}")
        
        # Save ensemble
        ensemble_filename = f"{save_dir}ensemble_{timestamp}.joblib"
        joblib.dump(self.ensemble, ensemble_filename)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'performance_metrics': self.performance_metrics,
            'feature_importance': {k: v.to_dict() for k, v in self.feature_importance.items()},
            'label_encoders': self.label_encoders,
            'cv_results': getattr(self, 'cv_results', {}),
            'feature_columns': list(self.X.columns)
        }
        
        metadata_filename = f"{save_dir}model_metadata_{timestamp}.joblib"
        joblib.dump(metadata, metadata_filename)
        
        print(f"Saved model metadata: {metadata_filename}")
        
        return timestamp
    
    def run_full_training_pipeline(self, features_path=None, full_data_path=None):
        """Run the complete model training pipeline"""
        print("=== READING LEVEL MODEL TRAINING PIPELINE ===\n")
        
        # Load and prepare data
        self.load_processed_data(features_path, full_data_path)
        self.create_modeling_datasets()
        self.create_train_test_splits()
        
        # Train models
        lexile_performance = self.train_lexile_models()
        category_performance = self.train_category_models()
        
        # Cross-validate
        cv_results = self.cross_validate_models()
        
        # Create ensemble
        ensemble = self.create_ensemble_model()
        
        # Implement tiered assignment
        tiered_results = self.implement_tiered_assignment()
        
        # Save models
        timestamp = self.save_models()
        
        print(f"\n=== TRAINING COMPLETE ===")
        print(f"Models saved with timestamp: {timestamp}")
        print("\nNext steps:")
        print("1. Evaluate on validation set")
        print("2. Test tiered assignment system")
        print("3. Deploy for book recommendations")
        
        return {
            'lexile_performance': lexile_performance,
            'category_performance': category_performance,
            'cv_results': cv_results,
            'ensemble': ensemble,
            'tiered_results': tiered_results,
            'timestamp': timestamp
        }

# Usage example:
if __name__ == "__main__":
    trainer = ReadingLevelModelTrainer()
    
    # Option 1: Load from CSV files
    results = trainer.run_full_training_pipeline(
        features_path='data/processed/modeling_features.csv',
        full_data_path='data/processed/books_engineered_features.csv'
    )
    