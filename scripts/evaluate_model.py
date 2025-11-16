"""
Comprehensive model evaluation and performance analysis
"""
import joblib
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self):
        self.models = self.load_models()
        self.test_data = self.load_test_data()
        self.results = {}
    
    def load_models(self):
        """Load all trained models"""
        print("📂 Loading models...")
        
        models_dir = Path("models")
        
        if not models_dir.exists():
            raise FileNotFoundError("❌ Models directory not found! Train models first.")
        
        try:
            models = {
                'rf': joblib.load(models_dir / 'rf_model.pkl'),
                'xgb': joblib.load(models_dir / 'xgb_model.pkl'),
                'ensemble': joblib.load(models_dir / 'ensemble_model.pkl'),
                'scaler': joblib.load(models_dir / 'scaler.pkl'),
                'label_encoder': joblib.load(models_dir / 'label_encoder.pkl')
            }
            
            with open(models_dir / 'metadata.json', 'r') as f:
                models['metadata'] = json.load(f)
            
            print(f"✅ Loaded models - Version {models['metadata']['model_version']}")
            print(f"   Crops supported: {models['metadata']['num_crops']}")
            print(f"   Training accuracy: {models['metadata']['accuracies']['ensemble']*100:.2f}%")
            
            return models
        
        except Exception as e:
            raise FileNotFoundError(f"❌ Error loading models: {e}")
    
    def load_test_data(self):
        """Load test dataset"""
        print("\n📂 Loading test data...")
        
        data_path = Path("data/processed/crop_recommendation_clean.csv")
        
        if not data_path.exists():
            raise FileNotFoundError("❌ Test data not found!")
        
        df = pd.read_csv(data_path)
        
        # Get feature names from metadata
        feature_names = self.models['metadata']['feature_names']
        
        # Prepare features and labels
        X = df[feature_names]
        y = df['label']
        
        # Encode labels
        y_encoded = self.models['label_encoder'].transform(y)
        
        # Scale features
        X_scaled = self.models['scaler'].transform(X)
        
        print(f"✅ Loaded {len(df)} samples for evaluation")
        
        return {
            'X': X_scaled,
            'y': y_encoded,
            'y_labels': y,
            'features': X,
            'feature_names': feature_names
        }
    
    def evaluate_accuracy(self):
        """Evaluate accuracy metrics"""
        print("\n📊 Evaluating Accuracy Metrics...")
        
        results = {}
        
        for model_name in ['rf', 'xgb', 'ensemble']:
            model = self.models[model_name]
            
            # Predictions
            y_pred = model.predict(self.test_data['X'])
            
            # Metrics
            accuracy = accuracy_score(self.test_data['y'], y_pred)
            precision = precision_score(self.test_data['y'], y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.test_data['y'], y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.test_data['y'], y_pred, average='weighted', zero_division=0)
            
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            print(f"\n   {model_name.upper()} Model:")
            print(f"      Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"      Precision: {precision:.4f}")
            print(f"      Recall:    {recall:.4f}")
            print(f"      F1-Score:  {f1:.4f}")
        
        self.results['accuracy_metrics'] = results
        return results
    
    def evaluate_per_crop(self):
        """Per-crop performance analysis"""
        print("\n📊 Per-Crop Performance Analysis...")
        
        model = self.models['ensemble']
        y_pred = model.predict(self.test_data['X'])
        
        # Classification report
        report = classification_report(
            self.test_data['y'],
            y_pred,
            target_names=self.models['label_encoder'].classes_,
            output_dict=True,
            zero_division=0
        )
        
        # Convert to DataFrame
        df_report = pd.DataFrame(report).transpose()
        
        # Filter crops only
        crop_performance = df_report[df_report['support'] > 0]
        crop_performance = crop_performance[~crop_performance.index.isin(['accuracy', 'macro avg', 'weighted avg'])]
        crop_performance = crop_performance.sort_values('f1-score', ascending=False)
        
        print("\n   Top 10 Best Predicted Crops:")
        for i, (crop, row) in enumerate(crop_performance.head(10).iterrows(), 1):
            print(f"      {i:2d}. {crop:20s}: F1={row['f1-score']:.4f}, Precision={row['precision']:.4f}, Recall={row['recall']:.4f}")
        
        if len(crop_performance) > 20:
            print("\n   Bottom 10 Crops (Need Attention):")
            for i, (crop, row) in enumerate(crop_performance.tail(10).iterrows(), 1):
                print(f"      {i:2d}. {crop:20s}: F1={row['f1-score']:.4f}, Precision={row['precision']:.4f}, Recall={row['recall']:.4f}")
        
        self.results['per_crop_performance'] = crop_performance
        return crop_performance
    
    def cross_validation_analysis(self):
        """Cross-validation analysis"""
        print("\n📊 Cross-Validation Analysis...")
        
        for model_name in ['rf', 'xgb', 'ensemble']:
            model = self.models[model_name]
            
            # 5-fold cross-validation
            cv_scores = cross_val_score(
                model,
                self.test_data['X'],
                self.test_data['y'],
                cv=5,
                scoring='accuracy'
            )
            
            print(f"\n   {model_name.upper()} - 5-Fold CV:")
            print(f"      Mean Accuracy: {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
            print(f"      Std Deviation: {cv_scores.std():.4f}")
            print(f"      Min Accuracy:  {cv_scores.min():.4f}")
            print(f"      Max Accuracy:  {cv_scores.max():.4f}")
        
        return cv_scores
    
    def confusion_matrix_analysis(self):
        """Confusion matrix analysis"""
        print("\n📊 Confusion Matrix Analysis...")
        
        model = self.models['ensemble']
        y_pred = model.predict(self.test_data['X'])
        
        cm = confusion_matrix(self.test_data['y'], y_pred)
        
        # Calculate per-class accuracy
        crop_names = self.models['label_encoder'].classes_
        
        print("\n   Per-Class Accuracy (Top 15):")
        accuracies = []
        for i, crop in enumerate(crop_names):
            if i < len(cm):
                class_accuracy = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
                accuracies.append((crop, class_accuracy))
        
        # Sort by accuracy
        accuracies.sort(key=lambda x: x[1], reverse=True)
        
        for i, (crop, acc) in enumerate(accuracies[:15], 1):
            print(f"      {i:2d}. {crop:20s}: {acc:.4f} ({acc*100:.2f}%)")
        
        self.results['confusion_matrix'] = cm
        return cm
    
    def feature_importance_analysis(self):
        """Feature importance analysis"""
        print("\n📊 Feature Importance Analysis...")
        
        rf_model = self.models['rf']
        feature_names = self.test_data['feature_names']
        importances = rf_model.feature_importances_
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        
        print("\n   Feature Ranking:")
        for i, idx in enumerate(indices, 1):
            bar = '█' * int(importances[idx] * 50)
            print(f"      {i}. {feature_names[idx]:15s}: {importances[idx]:.4f} {bar}")
        
        self.results['feature_importance'] = dict(zip(feature_names, importances))
        return dict(zip(feature_names, importances))
    
    def model_comparison(self):
        """Compare all models"""
        print("\n📊 Model Comparison Summary...")
        
        comparison = []
        
        for model_name in ['rf', 'xgb', 'ensemble']:
            metrics = self.results['accuracy_metrics'][model_name]
            comparison.append({
                'Model': model_name.upper(),
                'Accuracy': f"{metrics['accuracy']*100:.2f}%",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}"
            })
        
        df_comparison = pd.DataFrame(comparison)
        print("\n" + df_comparison.to_string(index=False))
        
        return df_comparison
    
    def save_evaluation_report(self):
        """Save comprehensive evaluation report"""
        print("\n💾 Saving Evaluation Report...")
        
        report_path = Path("models/evaluation_report.json")
        
        # Prepare serializable report
        report = {
            'evaluation_date': pd.Timestamp.now().isoformat(),
            'model_version': self.models['metadata']['model_version'],
            'dataset_size': len(self.test_data['y']),
            'num_crops': len(self.models['label_encoder'].classes_),
            'accuracy_metrics': self.results['accuracy_metrics'],
            'feature_importance': self.results['feature_importance'],
            'crops_supported': self.models['label_encoder'].classes_.tolist()
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"   ✅ Report saved to {report_path}")
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline"""
        print("=" * 70)
        print("📊 COMPREHENSIVE MODEL EVALUATION")
        print("=" * 70)
        
        # Run all evaluations
        self.evaluate_accuracy()
        self.evaluate_per_crop()
        self.cross_validation_analysis()
        self.confusion_matrix_analysis()
        self.feature_importance_analysis()
        self.model_comparison()
        
        # Save report
        self.save_evaluation_report()
        
        print("\n" + "=" * 70)
        print("✅ EVALUATION COMPLETED!")
        print("=" * 70)
        
        # Summary
        ensemble_acc = self.results['accuracy_metrics']['ensemble']['accuracy']
        print(f"\n🎯 Final Ensemble Accuracy: {ensemble_acc*100:.2f}%")
        print(f"📊 Total Crops Evaluated: {len(self.models['label_encoder'].classes_)}")
        print(f"📁 Report saved: models/evaluation_report.json")

def main():
    evaluator = ModelEvaluator()
    evaluator.run_full_evaluation()

if __name__ == "__main__":
    main()
