"""
Comprehensive test for deep learning crop recommendation model (.h5 format)
Tests ~30 major Indian crops with realistic agricultural parameters
"""
import joblib
import numpy as np
import json
from pathlib import Path
import tensorflow as tf
from tensorflow import keras


def load_models():
    """Load all trained models"""
    print("📂 Loading models...")
    
    models_dir = Path("models")
    
    if not models_dir.exists():
        print("❌ Models directory not found!")
        return None
    
    try:
        # Load Keras model
        model = keras.models.load_model('models/crop_model.h5')
        scaler = joblib.load('models/scaler.pkl')
        label_encoder = joblib.load('models/label_encoder.pkl')
        
        with open('models/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        print("✅ Models loaded successfully!")
        print(f"   Model version: {metadata['model_version']}")
        print(f"   Crops supported: {metadata['num_crops']}")
        print(f"   Test accuracy: {metadata['accuracies']['test']*100:.2f}%")
        
        return {
            'model': model,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'metadata': metadata
        }
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None


def predict_crop(features, models):
    """Make crop prediction"""
    # Scale features
    features_scaled = models['scaler'].transform([features])
    
    # Get predictions
    proba = models['model'].predict(features_scaled, verbose=0)[0]
    
    # Get top 3
    top_indices = np.argsort(proba)[-3:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'crop': models['label_encoder'].classes_[idx],
            'confidence': proba[idx],
            'confidence_pct': proba[idx] * 100
        })
    
    return results


def run_comprehensive_tests():
    """Run comprehensive tests for ~30 major crops"""
    models = load_models()
    
    if models is None:
        return
    
    print("\n" + "=" * 80)
    print("🧪 COMPREHENSIVE CROP TESTING - 30 MAJOR INDIAN CROPS")
    print("=" * 80)
    
    # Comprehensive test cases with realistic Indian agricultural parameters
    # Format: [N, P, K, Temperature, Humidity, pH, Rainfall]
    test_cases = [
        # CEREALS
        {'name': 'Rice', 'features': [80, 40, 40, 25, 80, 6.5, 200], 'expected': 'Rice', 'category': 'Cereal'},
        {'name': 'Rice (High Yield)', 'features': [90, 45, 45, 27, 85, 6.8, 220], 'expected': 'Rice', 'category': 'Cereal'},
        {'name': 'Wheat', 'features': [120, 60, 40, 20, 50, 6.8, 70], 'expected': 'Wheat', 'category': 'Cereal'},
        {'name': 'Wheat (Winter)', 'features': [125, 65, 42, 18, 48, 7.0, 65], 'expected': 'Wheat', 'category': 'Cereal'},
        {'name': 'Maize', 'features': [90, 50, 35, 24, 65, 6.5, 75], 'expected': 'Maize', 'category': 'Cereal'},
        {'name': 'Barley', 'features': [110, 42, 22, 15, 45, 7.2, 42], 'expected': 'Barley', 'category': 'Cereal'},
        
        # MILLETS
        {'name': 'Bajra', 'features': [45, 22, 23, 32, 42, 7.5, 40], 'expected': 'Bajra', 'category': 'Millet'},
        {'name': 'Bajra (Dry)', 'features': [40, 20, 20, 30, 38, 7.8, 35], 'expected': 'Bajra', 'category': 'Millet'},
        {'name': 'Ragi', 'features': [35, 18, 20, 26, 55, 6.2, 25], 'expected': 'Ragi', 'category': 'Millet'},
        {'name': 'Jowar', 'features': [50, 25, 28, 28, 48, 6.8, 55], 'expected': 'Jowar', 'category': 'Millet'},
        
        # PULSES
        {'name': 'Chickpea', 'features': [30, 60, 40, 22, 55, 6.5, 55], 'expected': 'Chickpea', 'category': 'Pulse'},
        {'name': 'Lentil', 'features': [25, 55, 35, 20, 50, 6.8, 45], 'expected': 'Lentil', 'category': 'Pulse'},
        {'name': 'Pigeonpea', 'features': [20, 40, 30, 26, 60, 6.5, 70], 'expected': 'Pigeonpea', 'category': 'Pulse'},
        {'name': 'Mungbean', 'features': [22, 45, 32, 28, 65, 6.2, 60], 'expected': 'Mungbean', 'category': 'Pulse'},
        {'name': 'Blackgram', 'features': [24, 48, 35, 27, 68, 6.3, 65], 'expected': 'Blackgram', 'category': 'Pulse'},
        {'name': 'Kidneybeans', 'features': [28, 50, 38, 24, 70, 6.5, 80], 'expected': 'Kidneybeans', 'category': 'Pulse'},
        {'name': 'Mothbeans', 'features': [18, 35, 28, 30, 45, 7.0, 35], 'expected': 'Mothbeans', 'category': 'Pulse'},
        
        # CASH CROPS
        {'name': 'Cotton', 'features': [110, 45, 60, 26, 70, 7.2, 90], 'expected': 'Cotton', 'category': 'Cash Crop'},
        {'name': 'Cotton (High)', 'features': [120, 50, 65, 28, 75, 7.5, 95], 'expected': 'Cotton', 'category': 'Cash Crop'},
        {'name': 'Jute', 'features': [85, 35, 50, 28, 85, 6.2, 160], 'expected': 'Jute', 'category': 'Cash Crop'},
        {'name': 'Sugarcane', 'features': [300, 120, 130, 32, 82, 6.5, 200], 'expected': 'Sugarcane', 'category': 'Cash Crop'},
        {'name': 'Tobacco', 'features': [150, 75, 90, 28, 75, 5.5, 140], 'expected': 'Tobacco', 'category': 'Cash Crop'},
        
        # OILSEEDS
        {'name': 'Groundnut', 'features': [15, 75, 80, 30, 55, 6.2, 55], 'expected': 'Groundnut', 'category': 'Oilseed'},
        {'name': 'Soybean', 'features': [20, 55, 45, 28, 75, 7.0, 100], 'expected': 'Soybean', 'category': 'Oilseed'},
        {'name': 'Mustard', 'features': [95, 28, 12, 15, 48, 6.2, 32], 'expected': 'Mustard', 'category': 'Oilseed'},
        {'name': 'Sunflower', 'features': [65, 45, 50, 24, 60, 6.5, 70], 'expected': 'Sunflower', 'category': 'Oilseed'},
        
        # VEGETABLES
        {'name': 'Potato', 'features': [170, 85, 95, 18, 82, 5.5, 45], 'expected': 'Potato', 'category': 'Vegetable'},
        {'name': 'Potato (Spring)', 'features': [165, 80, 90, 20, 80, 5.8, 50], 'expected': 'Potato', 'category': 'Vegetable'},
        {'name': 'Tomato', 'features': [90, 70, 75, 24, 70, 6.5, 65], 'expected': 'Vegetables', 'category': 'Vegetable'},
        {'name': 'Onion', 'features': [85, 65, 70, 22, 65, 6.8, 55], 'expected': 'Vegetables', 'category': 'Vegetable'},
        
        # FRUITS/OTHERS
        {'name': 'Banana', 'features': [200, 90, 100, 28, 75, 6.5, 150], 'expected': 'Banana', 'category': 'Fruit'},
        {'name': 'Mango', 'features': [120, 60, 80, 30, 60, 6.5, 85], 'expected': 'Mango', 'category': 'Fruit'},
        {'name': 'Grapes', 'features': [75, 50, 65, 26, 55, 6.8, 50], 'expected': 'Grapes', 'category': 'Fruit'},
        {'name': 'Watermelon', 'features': [100, 70, 80, 28, 70, 6.5, 60], 'expected': 'Watermelon', 'category': 'Fruit'},
        {'name': 'Muskmelon', 'features': [95, 65, 75, 27, 68, 6.8, 55], 'expected': 'Muskmelon', 'category': 'Fruit'},
        {'name': 'Apple', 'features': [60, 40, 55, 18, 65, 6.2, 90], 'expected': 'Apple', 'category': 'Fruit'},
        {'name': 'Orange', 'features': [80, 50, 65, 25, 70, 6.5, 110], 'expected': 'Orange', 'category': 'Fruit'},
        {'name': 'Papaya', 'features': [110, 70, 85, 26, 75, 6.8, 120], 'expected': 'Papaya', 'category': 'Fruit'},
        {'name': 'Coconut', 'features': [90, 55, 70, 27, 80, 6.5, 180], 'expected': 'Coconut', 'category': 'Fruit'},
        {'name': 'Pomegranate', 'features': [70, 45, 60, 24, 55, 7.0, 65], 'expected': 'Pomegranate', 'category': 'Fruit'},
        {'name': 'Coffee', 'features': [100, 60, 80, 23, 75, 6.0, 160], 'expected': 'Coffee', 'category': 'Beverage'},
    ]
    
    # Group by category for organized output
    categories = {}
    for test in test_cases:
        cat = test.get('category', 'Other')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(test)
    
    # Statistics
    passed = 0
    total = len(test_cases)
    category_stats = {}
    
    # Feature names for display
    feature_names = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
    
    # Test each category
    for category, tests in categories.items():
        print(f"\n{'='*80}")
        print(f"📦 CATEGORY: {category.upper()}")
        print(f"{'='*80}")
        
        category_passed = 0
        category_total = len(tests)
        
        for i, test in enumerate(tests, 1):
            print(f"\n{'-'*80}")
            print(f"Test {i}/{category_total}: {test['name']}")
            print(f"{'-'*80}")
            
            print("Input Parameters:")
            for name, value in zip(feature_names, test['features']):
                unit = {'N': 'kg/ha', 'P': 'kg/ha', 'K': 'kg/ha', 'Temperature': '°C', 
                       'Humidity': '%', 'pH': '', 'Rainfall': 'mm'}[name]
                print(f"  {name:12s}: {value:6.1f} {unit}")
            
            results = predict_crop(test['features'], models)
            
            print(f"\nTop 3 Predictions:")
            for j, result in enumerate(results, 1):
                emoji = "✅" if result['crop'] == test.get('expected') else "  "
                confidence_bar = "█" * int(result['confidence_pct'] / 5)
                print(f"{emoji} {j}. {result['crop']:20s}: {result['confidence_pct']:6.2f}% {confidence_bar}")
            
            # Check if passed
            if results[0]['crop'] == test['expected']:
                print(f"\n✅ PASSED - Correctly predicted {results[0]['crop']}")
                passed += 1
                category_passed += 1
            else:
                print(f"\n⚠️  FAILED - Predicted {results[0]['crop']}, expected {test['expected']}")
        
        # Category summary
        category_stats[category] = {
            'passed': category_passed,
            'total': category_total,
            'accuracy': (category_passed / category_total) * 100
        }
        
        print(f"\n{'='*80}")
        print(f"📊 {category} Summary: {category_passed}/{category_total} passed ({category_stats[category]['accuracy']:.1f}%)")
        print(f"{'='*80}")
    
    # Final summary
    print(f"\n{'='*80}")
    print("🏆 FINAL TEST RESULTS")
    print(f"{'='*80}")
    
    overall_accuracy = (passed / total) * 100
    
    print(f"\n📈 Overall Performance:")
    print(f"   Total Tests:     {total}")
    print(f"   Passed:          {passed}")
    print(f"   Failed:          {total - passed}")
    print(f"   Accuracy:        {overall_accuracy:.2f}%")
    
    print(f"\n📊 Category-wise Performance:")
    for category, stats in category_stats.items():
        status = "✅" if stats['accuracy'] >= 80 else "⚠️ " if stats['accuracy'] >= 60 else "❌"
        print(f"   {status} {category:15s}: {stats['passed']:2d}/{stats['total']:2d} ({stats['accuracy']:5.1f}%)")
    
    print(f"\n{'='*80}")
    
    # Performance grading
    if overall_accuracy >= 90:
        grade = "🌟 EXCELLENT"
        comment = "Model performs exceptionally well!"
    elif overall_accuracy >= 80:
        grade = "✅ GOOD"
        comment = "Model performs well with minor improvements needed."
    elif overall_accuracy >= 70:
        grade = "⚠️  FAIR"
        comment = "Model needs improvement in several areas."
    else:
        grade = "❌ POOR"
        comment = "Model requires significant retraining."
    
    print(f"Grade: {grade}")
    print(f"Comment: {comment}")
    print(f"{'='*80}")
    
    # Model information
    print(f"\n📁 Model Information:")
    print(f"   Model Type:      {models['metadata']['model_type']}")
    print(f"   Model Version:   {models['metadata']['model_version']}")
    print(f"   Training Date:   {models['metadata']['training_date']}")
    print(f"   Supported Crops: {models['metadata']['num_crops']}")
    
    return {
        'total': total,
        'passed': passed,
        'failed': total - passed,
        'accuracy': overall_accuracy,
        'category_stats': category_stats
    }


def run_quick_test():
    """Run quick test with 6 crops"""
    models = load_models()
    
    if models is None:
        return
    
    print("\n" + "=" * 70)
    print("🧪 QUICK TEST MODE")
    print("=" * 70)
    
    test_cases = [
        {'name': 'Rice', 'features': [80, 40, 40, 25, 80, 6.5, 200], 'expected': 'Rice'},
        {'name': 'Wheat', 'features': [120, 60, 40, 20, 50, 6.8, 70], 'expected': 'Wheat'},
        {'name': 'Cotton', 'features': [110, 45, 60, 26, 70, 7.2, 90], 'expected': 'Cotton'},
        {'name': 'Soybean', 'features': [20, 55, 45, 28, 75, 7.0, 100], 'expected': 'Soybean'},
        {'name': 'Potato', 'features': [170, 85, 95, 18, 82, 5.5, 45], 'expected': 'Potato'},
        {'name': 'Bajra', 'features': [45, 22, 23, 32, 42, 7.5, 40], 'expected': 'Bajra'},
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}/{total}: {test['name']}")
        print(f"{'='*70}")
        
        feature_names = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
        print("Input:")
        for name, value in zip(feature_names, test['features']):
            print(f"  {name:12s}: {value}")
        
        results = predict_crop(test['features'], models)
        
        print(f"\nTop 3 Predictions:")
        for j, result in enumerate(results, 1):
            emoji = "✅" if result['crop'] == test.get('expected') else "  "
            print(f"{emoji} {j}. {result['crop']:15s}: {result['confidence_pct']:.2f}%")
        
        if results[0]['crop'] == test['expected']:
            print(f"\n✅ PASSED")
            passed += 1
        else:
            print(f"\n⚠️  Predicted {results[0]['crop']}, expected {test['expected']}")
    
    print("\n" + "=" * 70)
    print(f"✅ TESTS: {passed}/{total} passed ({passed/total*100:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 80)
    print("🌾 CROP RECOMMENDATION MODEL - TESTING SUITE")
    print("=" * 80)
    print("\nSelect testing mode:")
    print("  1. Comprehensive Test (~40 test cases for 30+ crops)")
    print("  2. Quick Test (6 major crops)")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nEnter choice (1/2) [default: 1]: ").strip() or "1"
    
    if choice == "2":
        run_quick_test()
    else:
        results = run_comprehensive_tests()
        
        # Save test results
        if results:
            import json
            with open('test_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Test results saved to: test_results.json")
