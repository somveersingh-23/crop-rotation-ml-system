"""
IMPROVED DATA PIPELINE - Minimal augmentation with better distribution
"""
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class DatasetProcessor:
    """Improved processor with minimal synthetic data"""
    
    def __init__(self):
        self.raw_dir = Path("data/raw")
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.target_samples = 180  # Reduced for better quality
        
    def load_clean_dataset(self):
        """Load Crop_recommendation.csv"""
        print("\n📂 Loading Crop_recommendation.csv...")
        
        file_path = self.raw_dir / "Crop_recommendation.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"❌ {file_path} not found!")
        
        df = pd.read_csv(file_path)
        df.columns = [col.lower().strip() for col in df.columns]
        df = df.dropna().drop_duplicates()
        
        print(f"   ✅ Loaded {len(df)} samples, {df['label'].nunique()} crops")
        return df
    
    def smart_augment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Smarter augmentation preserving data distribution"""
        print(f"\n🔨 Smart augmentation to {self.target_samples} samples per crop...")
        
        augmented_dfs = []
        
        for crop in tqdm(df['label'].unique(), desc="Processing crops"):
            crop_df = df[df['label'] == crop].copy()
            current = len(crop_df)
            
            if current >= self.target_samples:
                crop_df = crop_df.sample(n=self.target_samples, random_state=42)
                augmented_dfs.append(crop_df)
            else:
                augmented_dfs.append(crop_df)
                
                # Calculate robust statistics
                needed = self.target_samples - current
                
                # Use percentile-based approach
                synthetic = []
                for _ in range(needed):
                    # Sample a base row
                    base_idx = np.random.choice(crop_df.index)
                    base = crop_df.loc[base_idx].copy()
                    
                    # Add small Gaussian noise (5-10%)
                    sample = {
                        'n': base['n'] * np.random.uniform(0.92, 1.08),
                        'p': base['p'] * np.random.uniform(0.92, 1.08),
                        'k': base['k'] * np.random.uniform(0.92, 1.08),
                        'temperature': base['temperature'] + np.random.normal(0, 1.5),
                        'humidity': base['humidity'] + np.random.normal(0, 3),
                        'ph': base['ph'] + np.random.normal(0, 0.2),
                        'rainfall': base['rainfall'] * np.random.uniform(0.90, 1.10),
                        'label': crop
                    }
                    
                    # Clip to reasonable ranges
                    sample['n'] = np.clip(sample['n'], 0, 140)
                    sample['p'] = np.clip(sample['p'], 5, 145)
                    sample['k'] = np.clip(sample['k'], 5, 205)
                    sample['temperature'] = np.clip(sample['temperature'], 10, 44)
                    sample['humidity'] = np.clip(sample['humidity'], 14, 100)
                    sample['ph'] = np.clip(sample['ph'], 3.5, 10.0)
                    sample['rainfall'] = np.clip(sample['rainfall'], 20, 300)
                    
                    synthetic.append(sample)
                
                augmented_dfs.append(pd.DataFrame(synthetic))
        
        result = pd.concat(augmented_dfs, ignore_index=True)
        result = result.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"   ✅ Augmented to {len(result)} samples")
        return result
    
    def add_missing_crops(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add important missing crops with realistic ranges"""
        print(f"\n🔨 Adding missing crops ({self.target_samples} samples each)...")
        
        missing = {
            'Wheat': {'n': (100, 150), 'p': (50, 70), 'k': (30, 50), 'temp': (15, 22), 'hum': (40, 55), 'ph': (6.5, 7.2), 'rain': (50, 100)},
            'Sugarcane': {'n': (250, 400), 'p': (100, 150), 'k': (100, 150), 'temp': (28, 36), 'hum': (75, 92), 'ph': (6.0, 6.8), 'rain': (180, 260)},
            'Potato': {'n': (130, 200), 'p': (70, 100), 'k': (80, 120), 'temp': (14, 20), 'hum': (75, 95), 'ph': (5.0, 6.0), 'rain': (30, 60)},
            'Bajra': {'n': (30, 60), 'p': (15, 30), 'k': (15, 30), 'temp': (28, 38), 'hum': (30, 50), 'ph': (7.0, 8.5), 'rain': (30, 50)},
            'Groundnut': {'n': (10, 20), 'p': (60, 90), 'k': (60, 100), 'temp': (25, 35), 'hum': (45, 60), 'ph': (5.5, 6.5), 'rain': (40, 70)},
            'Soybean': {'n': (10, 25), 'p': (40, 60), 'k': (35, 55), 'temp': (25, 32), 'hum': (65, 85), 'ph': (6.5, 7.5), 'rain': (80, 120)},
            'Mustard': {'n': (80, 110), 'p': (20, 35), 'k': (8, 15), 'temp': (12, 18), 'hum': (40, 55), 'ph': (5.8, 6.5), 'rain': (25, 40)},
            'Barley': {'n': (90, 130), 'p': (35, 50), 'k': (18, 28), 'temp': (12, 18), 'hum': (35, 50), 'ph': (7.0, 8.0), 'rain': (35, 50)},
            'Tobacco': {'n': (120, 180), 'p': (60, 90), 'k': (70, 110), 'temp': (24, 32), 'hum': (65, 85), 'ph': (5.0, 6.0), 'rain': (110, 170)},
            'Vegetables': {'n': (70, 100), 'p': (70, 95), 'k': (60, 85), 'temp': (18, 26), 'hum': (55, 70), 'ph': (6.0, 6.8), 'rain': (55, 85)},
        }
        
        new_crops = []
        
        for crop, ranges in tqdm(missing.items(), desc="Generating"):
            for _ in range(self.target_samples):
                sample = {
                    'n': np.random.uniform(*ranges['n']),
                    'p': np.random.uniform(*ranges['p']),
                    'k': np.random.uniform(*ranges['k']),
                    'temperature': np.random.uniform(*ranges['temp']),
                    'humidity': np.random.uniform(*ranges['hum']),
                    'ph': np.random.uniform(*ranges['ph']),
                    'rainfall': np.random.uniform(*ranges['rain']),
                    'label': crop
                }
                new_crops.append(sample)
        
        new_df = pd.DataFrame(new_crops)
        combined = pd.concat([df, new_df], ignore_index=True)
        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"   ✅ Added {len(new_df)} samples for {len(missing)} crops")
        return combined
    
    def save(self, df: pd.DataFrame):
        """Save processed data"""
        print("\n💾 Saving...")
        
        output = self.processed_dir / 'crop_recommendation_clean.csv'
        df.to_csv(output, index=False)
        
        print(f"   ✅ Saved: {output}")
        print(f"   Samples: {len(df)}")
        print(f"   Crops: {df['label'].nunique()}")
        
        print(f"\n   📊 Distribution:")
        counts = df['label'].value_counts()
        for i, (crop, count) in enumerate(counts.items(), 1):
            status = "✅" if 150 <= count <= 200 else "⚠️ "
            print(f"      {i:2d}. {status} {crop:20s}: {count:3d}")
        
        print(f"\n   Balance: {counts.min()}-{counts.max()} samples per crop")
    
    def process(self):
        """Execute"""
        print("=" * 70)
        print("🌾 IMPROVED CLEAN PIPELINE")
        print("=" * 70)
        
        df = self.load_clean_dataset()
        df = self.smart_augment(df)
        df = self.add_missing_crops(df)
        self.save(df)
        
        print("\n" + "=" * 70)
        print("✅ DONE! Run: python scripts/train_model.py")
        print("=" * 70)


def main():
    DatasetProcessor().process()


if __name__ == "__main__":
    main()
