"""
Data preprocessing utilities for crop recommendation ML model
Handles feature engineering, data cleaning, and transformation
Updated to work with multiple dataset formats
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class CropDataPreprocessor:
    """
    Comprehensive data preprocessing for crop recommendation
    Handles missing values, outliers, feature scaling, and encoding
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = ['n', 'p', 'k', 'temperature', 'humidity', 'ph', 'rainfall']
        self.imputer = SimpleImputer(strategy='median')
        self.fitted = False
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw data by handling missing values and duplicates
        
        Args:
            df: Raw dataframe
        
        Returns:
            Cleaned dataframe
        """
        print("🔧 Cleaning data...")
        
        # Make a copy
        df_clean = df.copy()
        
        # Standardize column names (lowercase, remove spaces)
        df_clean.columns = df_clean.columns.str.lower().str.strip().str.replace(' ', '_')
        
        # Handle common column name variations
        column_mapping = {
            'crop': 'label',
            'temp': 'temperature',
            'crop_type': 'label',
            'crop_name': 'label',
            'nitrogen': 'n',
            'phosphorous': 'p',
            'phosphorus': 'p',
            'potassium': 'k'
        }
        
        df_clean.rename(columns=column_mapping, inplace=True)
        
        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed_duplicates = initial_rows - len(df_clean)
        if removed_duplicates > 0:
            print(f"   Removed {removed_duplicates} duplicate rows")
        
        # Remove rows with missing target variable
        if 'label' in df_clean.columns:
            initial_rows = len(df_clean)
            df_clean = df_clean.dropna(subset=['label'])
            removed_null_labels = initial_rows - len(df_clean)
            if removed_null_labels > 0:
                print(f"   Removed {removed_null_labels} rows with missing labels")
        
        print(f"   ✅ Clean data: {len(df_clean)} rows")
        return df_clean
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        print("🔧 Handling missing values...")
        
        df_imputed = df.copy()
        
        # Get available feature columns
        available_features = [f for f in self.feature_names if f in df_imputed.columns]
        
        if not available_features:
            print("   ⚠️  No feature columns found")
            return df_imputed
        
        # Check for missing values
        missing_summary = df_imputed[available_features].isnull().sum()
        total_missing = missing_summary.sum()
        
        if total_missing > 0:
            print(f"   Found {total_missing} missing values")
            
            # Impute numeric features with median
            numeric_features = df_imputed[available_features].select_dtypes(include=[np.number]).columns
            
            if len(numeric_features) > 0:
                df_imputed[numeric_features] = self.imputer.fit_transform(df_imputed[numeric_features])
                print(f"   ✅ Imputed missing values using median strategy")
        else:
            print("   ✅ No missing values found")
        
        return df_imputed
    
    def remove_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """Remove outliers using IQR method"""
        print(f"🔧 Removing outliers using {method.upper()} method...")
        
        df_clean = df.copy()
        initial_rows = len(df_clean)
        
        # Get available features
        available_features = [f for f in self.feature_names if f in df_clean.columns]
        
        for feature in available_features:
            if method == 'iqr':
                Q1 = df_clean[feature].quantile(0.25)
                Q3 = df_clean[feature].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df_clean = df_clean[
                    (df_clean[feature] >= lower_bound) & 
                    (df_clean[feature] <= upper_bound)
                ]
        
        removed_outliers = initial_rows - len(df_clean)
        print(f"   Removed {removed_outliers} outlier rows")
        print(f"   ✅ Remaining data: {len(df_clean)} rows")
        
        return df_clean
    
    def validate_feature_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clip feature values to realistic ranges"""
        print("🔧 Validating feature ranges...")
        
        df_valid = df.copy()
        
        # Define realistic ranges for agricultural features
        feature_ranges = {
            'n': (0, 500),           # Nitrogen (kg/ha)
            'p': (0, 200),           # Phosphorus (kg/ha)
            'k': (0, 200),           # Potassium (kg/ha)
            'temperature': (5, 50),  # Temperature (°C)
            'humidity': (0, 100),    # Humidity (%)
            'ph': (3.5, 9.5),        # pH
            'rainfall': (0, 3500)    # Rainfall (mm)
        }
        
        clipped_count = 0
        
        for feature, (min_val, max_val) in feature_ranges.items():
            if feature in df_valid.columns:
                # Count values outside range
                outside_range = (
                    (df_valid[feature] < min_val) | 
                    (df_valid[feature] > max_val)
                ).sum()
                
                if outside_range > 0:
                    clipped_count += outside_range
                    df_valid[feature] = df_valid[feature].clip(min_val, max_val)
        
        if clipped_count > 0:
            print(f"   Clipped {clipped_count} values to valid ranges")
        
        print("   ✅ Feature ranges validated")
        return df_valid
    
    def standardize_crop_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize crop names for consistency"""
        print("🔧 Standardizing crop names...")
        
        if 'label' not in df.columns:
            return df
        
        df_std = df.copy()
        
        # Clean crop names
        df_std['label'] = df_std['label'].str.strip().str.title()
        
        # Common name mappings
        name_mappings = {
            'Paddy': 'Rice',
            'Maize': 'Maize',
            'Ground Nuts': 'Groundnut',
            'Ground_Nuts': 'Groundnut',
            'Oil Seeds': 'Oilseed',
            'Oil_Seeds': 'Oilseed',
            'Millets': 'Millet',
            'Pulses': 'Pulse'
        }
        
        df_std['label'] = df_std['label'].replace(name_mappings)
        
        # Remove extra whitespace
        df_std['label'] = df_std['label'].str.replace(r'\s+', ' ', regex=True)
        
        print(f"   ✅ Standardized {df_std['label'].nunique()} unique crop names")
        return df_std
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Scale features using StandardScaler"""
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            self.fitted = True
        else:
            if not self.fitted:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def encode_labels(self, y: pd.Series, fit: bool = True) -> np.ndarray:
        """Encode crop labels to numeric values"""
        if fit:
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = self.label_encoder.transform(y)
        
        return y_encoded
    
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        remove_outliers: bool = True,
        balance_dataset: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline for training data
        
        Args:
            df: Raw dataframe
            remove_outliers: Whether to remove outliers
            balance_dataset: Whether to balance classes
        
        Returns:
            Tuple of (X_scaled, y_encoded)
        """
        print("=" * 60)
        print("🔧 PREPROCESSING PIPELINE")
        print("=" * 60)
        
        # Step 1: Clean data
        df_clean = self.clean_data(df)
        
        # Step 2: Standardize crop names
        df_std = self.standardize_crop_names(df_clean)
        
        # Step 3: Handle missing values
        df_imputed = self.handle_missing_values(df_std)
        
        # Step 4: Validate ranges
        df_valid = self.validate_feature_ranges(df_imputed)
        
        # Step 5: Remove outliers (optional)
        if remove_outliers:
            df_final = self.remove_outliers(df_valid, method='iqr')
        else:
            df_final = df_valid
        
        # Step 6: Filter crops with minimum samples (at least 20)
        crop_counts = df_final['label'].value_counts()
        valid_crops = crop_counts[crop_counts >= 20].index
        df_final = df_final[df_final['label'].isin(valid_crops)]
        
        print(f"\n   Filtered crops with <20 samples")
        print(f"   Remaining crops: {len(valid_crops)}")
        
        # Step 7: Extract features and labels
        available_features = [f for f in self.feature_names if f in df_final.columns]
        self.feature_names = available_features  # Update feature list
        
        X = df_final[available_features]
        y = df_final['label']
        
        # Step 8: Scale and encode
        X_scaled = self.scale_features(X, fit=True)
        y_encoded = self.encode_labels(y, fit=True)
        
        print("\n" + "=" * 60)
        print("✅ PREPROCESSING COMPLETED")
        print("=" * 60)
        print(f"   Final dataset: {len(X_scaled)} samples")
        print(f"   Features: {len(self.feature_names)}")
        print(f"   Features used: {self.feature_names}")
        print(f"   Classes: {len(self.label_encoder.classes_)}")
        print(f"   Crops: {', '.join(self.label_encoder.classes_[:10])}")
        if len(self.label_encoder.classes_) > 10:
            print(f"          ... and {len(self.label_encoder.classes_) - 10} more")
        
        return X_scaled, y_encoded
    
    def prepare_inference_data(self, features: Dict[str, float]) -> np.ndarray:
        """Prepare single sample for inference"""
        # Create dataframe from features
        df = pd.DataFrame([features])
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Select features in correct order
        X = df[self.feature_names]
        
        # Validate ranges
        df_valid = self.validate_feature_ranges(X)
        
        # Scale
        X_scaled = self.scale_features(df_valid, fit=False)
        
        return X_scaled
    
    def save_preprocessor(self, output_dir: Path):
        """Save scaler and encoder"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import joblib
        joblib.dump(self.scaler, output_dir / 'scaler.pkl')
        joblib.dump(self.label_encoder, output_dir / 'label_encoder.pkl')
        
        print(f"   ✅ Saved preprocessor to {output_dir}")

# Global preprocessor instance
preprocessor = CropDataPreprocessor()

# Convenience functions
def preprocess_training_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience function to preprocess training data"""
    return preprocessor.prepare_training_data(
        df,
        remove_outliers=True,
        balance_dataset=False
    )

def preprocess_inference_data(features: Dict[str, float]) -> np.ndarray:
    """Convenience function to preprocess inference data"""
    return preprocessor.prepare_inference_data(features)
