"""
Simple test script for Fake News Detection System
Tests the basic functionality without command line arguments
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

print("="*70)
print("FAKE NEWS DETECTION SYSTEM - SIMPLE TEST")
print("="*70)

# Test if models can be imported
print("\n1. Testing imports...")
try:
    from services.inference import MultiModelInference
    from services.orchestrator import PredictionOrchestrator, EnsembleMethod
    from services.explainer import PredictionExplainer
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test model initialization
print("\n2. Initializing models...")
try:
    inference = MultiModelInference(
        use_bert=True,
        use_roberta=True,
        use_tfidf=True
    )
    print(f"✓ Successfully loaded {len(inference.models)} model(s): {list(inference.models.keys())}")
except Exception as e:
    print(f"✗ Model initialization failed: {e}")
    print("\nNote: Make sure model files are in the correct locations:")
    print("  - models/bert/final_model/")
    print("  - models/roberta/final_model/")
    print("  - models/tf_idf/fake_news_classifier.joblib")
    print("  - models/tf_idf/tfidf_model.joblib")
    sys.exit(1)

# Test predictions
print("\n3. Testing predictions...")
test_texts = [
    "Scientists announce breakthrough in renewable energy research.",
    "SHOCKING!!! You WON'T BELIEVE what happens next!!!"
]

for i, text in enumerate(test_texts, 1):
    print(f"\n--- Test {i} ---")
    print(f"Text: {text}")
    
    try:
        # Get predictions
        predictions = inference.predict_all(text)
        
        print("\nPredictions:")
        for model_name, pred in predictions.items():
            print(f"  {model_name}: {pred.get('prediction', 'ERROR')} "
                  f"({pred.get('confidence', 0):.2%})")
        
        # Test ensemble
        orchestrator = PredictionOrchestrator(
            ensemble_method=EnsembleMethod.CONFIDENCE_WEIGHTED
        )
        ensemble_result = orchestrator.ensemble_predict(predictions)
        
        print(f"\nEnsemble: {ensemble_result.final_prediction} "
              f"(Confidence: {ensemble_result.confidence:.2%}, "
              f"Consensus: {ensemble_result.consensus_score:.2%})")
        
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*70)
print("TEST COMPLETED")
print("="*70)
print("\nTo use the full system:")
print("  python main.py --text 'Your news text here'")
print("  python main.py --interactive")
print("  python main.py --file article.txt")
print("\nFor examples:")
print("  python examples.py --example 1")
print("="*70)
