"""
DEEP LEARNING CROP RECOMMENDATION - 99% Target Accuracy
Uses TensorFlow/Keras with .h5 model format
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class CropDataPreprocessor:
    """Lightweight preprocessor without aggressive filtering"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = ['n', 'p', 'k', 'temperature', 'humidity', 'ph', 'rainfall']
        self.fitted = False
    
    def prepare_data(self, df: pd.DataFrame) -> tuple:
        """Minimal preprocessing - NO outlier removal"""
        print("=" * 60)
        print("🔧 PREPROCESSING PIPELINE")
        print("=" * 60)
        
        # Clean column names
        df = df.copy()
        df.columns = df.columns.str.lower().str.strip()
        
        # Standardize labels
        df['label'] = df['label'].str.strip().str.title()
        
        # Drop only rows with missing values
        df_clean = df.dropna()
        print(f"✅ Clean data: {len(df_clean)} rows")
        
        # Validate ranges (clip only extreme outliers)
        df_clean['n'] = df_clean['n'].clip(0, 500)
        df_clean['p'] = df_clean['p'].clip(0, 200)
        df_clean['k'] = df_clean['k'].clip(0, 205)
        df_clean['temperature'] = df_clean['temperature'].clip(5, 50)
        df_clean['humidity'] = df_clean['humidity'].clip(0, 100)
        df_clean['ph'] = df_clean['ph'].clip(3.5, 10)
        df_clean['rainfall'] = df_clean['rainfall'].clip(0, 3500)
        
        # Filter crops with minimum samples
        crop_counts = df_clean['label'].value_counts()
        valid_crops = crop_counts[crop_counts >= 30].index
        df_clean = df_clean[df_clean['label'].isin(valid_crops)]
        
        print(f"✅ Crops with ≥30 samples: {len(valid_crops)}")
        print(f"✅ Final dataset: {len(df_clean)} samples")
        
        # Extract features
        X = df_clean[self.feature_names].values
        y = df_clean['label'].values
        
        # Scale and encode
        X_scaled = self.scaler.fit_transform(X)
        y_encoded = self.label_encoder.fit_transform(y)
        
        self.fitted = True
        
        print(f"✅ Features: {len(self.feature_names)}")
        print(f"✅ Classes: {len(self.label_encoder.classes_)}")
        print("=" * 60)
        
        return X_scaled, y_encoded


class DeepCropModel:
    """Deep Neural Network for Crop Recommendation"""
    
    def __init__(self, input_dim, num_classes):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = self.build_model()
    
    def build_model(self):
        """Build improved deep learning architecture"""
        model = keras.Sequential([
            # Input layer with batch normalization
            layers.Input(shape=(self.input_dim,)),
            layers.BatchNormalization(),
            
            # First hidden layer
            layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Second hidden layer
            layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            # Third hidden layer
            layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile with advanced optimizer
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model


class CropRecommendationTrainer:
    """Train deep learning model"""
    
    def __init__(self):
        self.model = None
        self.preprocessor = CropDataPreprocessor()
        self.data_path = Path("data/processed/crop_recommendation_clean.csv")
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
    def load_data(self):
        """Load dataset"""
        print("📂 Loading dataset...")
        
        if not self.data_path.exists():
            print(f"❌ Dataset not found at {self.data_path}")
            return None
        
        df = pd.read_csv(self.data_path)
        print(f"   ✅ Loaded: {len(df)} samples")
        print(f"   ✅ Crops: {df['label'].nunique()} unique crops")
        
        return df
    
    def train(self):
        """Complete training pipeline"""
        print("=" * 70)
        print("🌾 DEEP LEARNING CROP RECOMMENDATION TRAINING")
        print("=" * 70)
        
        # Load and preprocess
        df = self.load_data()
        if df is None:
            return
        
        X, y = self.preprocessor.prepare_data(df)
        
        # Split data (85-15 for better training)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        
        print(f"\n📊 Data Split:")
        print(f"   Training: {len(X_train)} samples")
        print(f"   Testing: {len(X_test)} samples")
        
        # Build model
        print(f"\n🔧 Building model...")
        num_classes = len(self.preprocessor.label_encoder.classes_)
        deep_model = DeepCropModel(input_dim=7, num_classes=num_classes)
        self.model = deep_model.model
        
        print(self.model.summary())
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                str(self.models_dir / 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train
        print(f"\n🚀 Training model...")
        history = self.model.fit(
            X_train, y_train,
            validation_split=0.15,
            epochs=150,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        print(f"\n📊 Evaluating model...")
        train_loss, train_acc = self.model.evaluate(X_train, y_train, verbose=0)
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\n   📈 Results:")
        print(f"      Training Accuracy:  {train_acc*100:.2f}%")
        print(f"      Test Accuracy:      {test_acc*100:.2f}%")
        
        # Predictions
        y_pred = np.argmax(self.model.predict(X_test, verbose=0), axis=1)
        
        # Classification report (top 10 crops)
        top_10_classes = np.argsort(np.bincount(y_test))[-10:]
        mask = np.isin(y_test, top_10_classes)
        
        if mask.sum() > 0:
            print(f"\n   📋 Classification Report (Top 10 Crops):")
            report = classification_report(
                y_test[mask],
                y_pred[mask],
                labels=top_10_classes,
                target_names=[self.preprocessor.label_encoder.classes_[i] for i in top_10_classes],
                digits=4
            )
            print(report)
        
        # Save models
        self.save_models(train_acc, test_acc)
        
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETED!")
        print("=" * 70)
        print(f"\n🎯 Final Test Accuracy: {test_acc*100:.2f}%")
        print(f"📊 Total Crops: {num_classes}")
        print("\n📍 Next: python scripts/test_model.py")
    
    def save_models(self, train_acc, test_acc):
        """Save all models and metadata"""
        print("\n💾 Saving models...")
        
        # Save Keras model as .h5
        self.model.save(self.models_dir / 'crop_model.h5')
        print("   ✅ Saved: crop_model.h5")
        
        # Save preprocessing objects
        joblib.dump(self.preprocessor.scaler, self.models_dir / 'scaler.pkl')
        joblib.dump(self.preprocessor.label_encoder, self.models_dir / 'label_encoder.pkl')
        print("   ✅ Saved: scaler.pkl, label_encoder.pkl")
        
        # Save metadata
        metadata = {
            'model_version': '2.0.0',
            'model_type': 'deep_learning',
            'training_date': pd.Timestamp.now().isoformat(),
            'feature_names': self.preprocessor.feature_names,
            'crops': self.preprocessor.label_encoder.classes_.tolist(),
            'num_crops': len(self.preprocessor.label_encoder.classes_),
            'accuracies': {
                'train': float(train_acc),
                'test': float(test_acc)
            }
        }
        
        with open(self.models_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("   ✅ Saved: metadata.json")


def main():
    trainer = CropRecommendationTrainer()
    trainer.train()


if __name__ == "__main__":
    main()
