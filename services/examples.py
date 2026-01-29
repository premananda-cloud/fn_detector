"""
Example Usage Script for Fake News Detection System
Demonstrates various use cases and features
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from main import FakeNewsDetector
from services.inference import MultiModelInference
from services.orchestrator import PredictionOrchestrator, EnsembleMethod
from services.explainer import PredictionExplainer


def example_1_basic_usage():
    """Example 1: Basic usage with default settings"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Usage")
    print("="*70)
    
    # Initialize detector with all models
    detector = FakeNewsDetector(verbose=True)
    
    # Test text
    text = """
    BREAKING NEWS!!! Scientists have discovered a MIRACLE CURE for all diseases!
    You won't believe what they found! Anonymous sources confirm this SHOCKING
    revelation that doctors don't want you to know!!!
    """
    
    # Get prediction
    result = detector.predict(text, explain=True)
    
    # Print result
    detector.print_result(result, detailed=True)


def example_2_model_selection():
    """Example 2: Using specific models only"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Model Selection")
    print("="*70)
    
    # Use only transformer models (BERT and RoBERTa)
    detector = FakeNewsDetector(
        use_bert=True,
        use_roberta=True,
        use_tfidf=False,  # Disable TF-IDF
        verbose=True
    )
    
    text = "According to peer-reviewed research published in Nature, climate change impacts are accelerating."
    
    result = detector.predict(text, explain=True)
    detector.print_result(result)


def example_3_ensemble_methods():
    """Example 3: Comparing different ensemble methods"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Ensemble Methods Comparison")
    print("="*70)
    
    text = "New study reveals shocking health benefits of chocolate!"
    
    # Test all ensemble methods
    methods = [
        'majority_vote',
        'weighted_average',
        'confidence_weighted',
        'adaptive'
    ]
    
    for method in methods:
        print(f"\n--- Using {method.upper()} ---")
        
        detector = FakeNewsDetector(
            ensemble_method=method,
            verbose=False
        )
        
        result = detector.predict(text, explain=False)
        
        print(f"Prediction: {result['final_prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Consensus: {result['consensus_score']:.2%}")


def example_4_batch_processing():
    """Example 4: Batch processing multiple texts"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Batch Processing")
    print("="*70)
    
    detector = FakeNewsDetector(verbose=False)
    
    # Multiple texts to analyze
    texts = [
        "Scientists announce breakthrough in renewable energy technology.",
        "SHOCKING: Celebrity reveals secret to eternal youth!!!",
        "Government releases new economic data showing modest growth.",
        "You won't BELIEVE this ONE TRICK doctors HATE!!!",
        "Research study confirms effectiveness of new medical treatment."
    ]
    
    # Process batch
    results = detector.predict_batch(texts, explain=False)
    
    # Summary
    print("\nBatch Processing Results:")
    print("-" * 70)
    
    for i, (text, result) in enumerate(zip(texts, results), 1):
        pred = result['final_prediction']
        conf = result['confidence']
        emoji = "🚫" if pred == "FAKE" else "✅"
        
        print(f"{i}. {emoji} {pred} ({conf:.0%}) - {text[:50]}...")
    
    # Statistics
    fake_count = sum(1 for r in results if r['final_prediction'] == 'FAKE')
    real_count = sum(1 for r in results if r['final_prediction'] == 'REAL')
    
    print("\nSummary:")
    print(f"  Total: {len(results)}")
    print(f"  FAKE: {fake_count} ({fake_count/len(results)*100:.0f}%)")
    print(f"  REAL: {real_count} ({real_count/len(results)*100:.0f}%)")


def example_5_detailed_analysis():
    """Example 5: Detailed analysis with explanations"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Detailed Analysis with Explanations")
    print("="*70)
    
    detector = FakeNewsDetector(verbose=False)
    
    text = """
    A groundbreaking study published yesterday in the Journal of Medical Research
    suggests that a new treatment approach may help patients with chronic conditions.
    Dr. Sarah Johnson, lead researcher at the University Medical Center, stated that
    preliminary results show promise, though more research is needed. The study,
    which followed 500 participants over two years, demonstrated statistically
    significant improvements in patient outcomes compared to traditional methods.
    """
    
    result = detector.predict(text, explain=True)
    
    # Print full detailed result
    detector.print_result(result, detailed=True)
    
    # Print additional explanation details
    if 'explanation' in result:
        exp = result['explanation']
        
        print("\n--- Text Analysis Details ---")
        analysis = exp.get('text_analysis', {})
        print(f"Word count: {analysis.get('word_count', 0)}")
        print(f"Sentence count: {analysis.get('sentence_count', 0)}")
        print(f"All-caps words: {analysis.get('all_caps_words', 0)}")
        print(f"Exclamation marks: {analysis.get('exclamation_marks', 0)}")


def example_6_component_usage():
    """Example 6: Using individual components directly"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Direct Component Usage")
    print("="*70)
    
    text = "Breaking: Scientists discover aliens on Mars!"
    
    # 1. Inference
    print("\n1. Running inference...")
    inference = MultiModelInference(use_bert=True, use_roberta=True, use_tfidf=True)
    predictions = inference.predict_all(text)
    
    print("Individual predictions:")
    for model, pred in predictions.items():
        print(f"  {model}: {pred['prediction']} ({pred['confidence']:.2%})")
    
    # 2. Orchestration
    print("\n2. Creating ensemble prediction...")
    orchestrator = PredictionOrchestrator(
        ensemble_method=EnsembleMethod.CONFIDENCE_WEIGHTED
    )
    ensemble_result = orchestrator.ensemble_predict(predictions)
    
    print(f"Ensemble: {ensemble_result.final_prediction} ({ensemble_result.confidence:.2%})")
    
    # 3. Explanation
    print("\n3. Generating explanation...")
    explainer = PredictionExplainer()
    explanation = explainer.explain_ensemble(text, predictions, ensemble_result)
    
    print("Key findings:")
    for finding in explanation.get('key_findings', []):
        print(f"  • {finding}")


def example_7_risk_assessment():
    """Example 7: Risk level assessment"""
    print("\n" + "="*70)
    print("EXAMPLE 7: Risk Level Assessment")
    print("="*70)
    
    detector = FakeNewsDetector(verbose=False)
    
    # Test different types of content
    test_cases = [
        ("High confidence fake", "SHOCKING SECRET that doctors DON'T want you to know!!!"),
        ("High confidence real", "Government report shows GDP growth of 2.3% this quarter."),
        ("Ambiguous content", "Some people say this might be true, but others disagree."),
        ("Sensational real news", "Major earthquake strikes coastal region, thousands evacuated.")
    ]
    
    print("\nRisk Assessment Results:")
    print("-" * 70)
    
    for label, text in test_cases:
        result = detector.predict(text, explain=False)
        
        print(f"\n{label}:")
        print(f"  Text: {text[:60]}...")
        print(f"  Prediction: {result['final_prediction']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Risk Level: {result['risk_level']}")


def example_8_custom_weights():
    """Example 8: Custom model weights"""
    print("\n" + "="*70)
    print("EXAMPLE 8: Custom Model Weights")
    print("="*70)
    
    text = "New research suggests potential health benefits."
    
    # Default weights
    print("\n--- With Default Weights ---")
    detector1 = FakeNewsDetector(
        ensemble_method='weighted_average',
        verbose=False
    )
    result1 = detector1.predict(text, explain=False)
    print(f"Prediction: {result1['final_prediction']} ({result1['confidence']:.2%})")
    
    # Custom weights (favor RoBERTa)
    print("\n--- With Custom Weights (RoBERTa-heavy) ---")
    from services.orchestrator import PredictionOrchestrator
    from services.inference import MultiModelInference
    
    inference = MultiModelInference()
    predictions = inference.predict_all(text)
    
    custom_orchestrator = PredictionOrchestrator(
        ensemble_method=EnsembleMethod.WEIGHTED_AVERAGE,
        model_weights={'BERT': 0.2, 'RoBERTa': 0.6, 'TF-IDF': 0.2}
    )
    
    result2 = custom_orchestrator.ensemble_predict(predictions)
    print(f"Prediction: {result2.final_prediction} ({result2.confidence:.2%})")


def run_all_examples():
    """Run all examples"""
    examples = [
        ("Basic Usage", example_1_basic_usage),
        ("Model Selection", example_2_model_selection),
        ("Ensemble Methods", example_3_ensemble_methods),
        ("Batch Processing", example_4_batch_processing),
        ("Detailed Analysis", example_5_detailed_analysis),
        ("Component Usage", example_6_component_usage),
        ("Risk Assessment", example_7_risk_assessment),
        ("Custom Weights", example_8_custom_weights),
    ]
    
    for name, func in examples:
        try:
            func()
            print("\n✓ Example completed successfully\n")
        except Exception as e:
            print(f"\n✗ Example failed: {e}\n")
            import traceback
            traceback.print_exc()


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fake News Detection Examples')
    parser.add_argument('--example', type=int, help='Run specific example (1-8)')
    parser.add_argument('--all', action='store_true', help='Run all examples')
    
    args = parser.parse_args()
    
    if args.all:
        run_all_examples()
    elif args.example:
        examples = [
            example_1_basic_usage,
            example_2_model_selection,
            example_3_ensemble_methods,
            example_4_batch_processing,
            example_5_detailed_analysis,
            example_6_component_usage,
            example_7_risk_assessment,
            example_8_custom_weights,
        ]
        
        if 1 <= args.example <= len(examples):
            examples[args.example - 1]()
        else:
            print(f"Invalid example number. Choose 1-{len(examples)}")
    else:
        print("Fake News Detection System - Examples")
        print("\nUsage:")
        print("  python examples.py --example 1    # Run example 1")
        print("  python examples.py --all          # Run all examples")
        print("\nAvailable examples:")
        print("  1. Basic Usage")
        print("  2. Model Selection")
        print("  3. Ensemble Methods")
        print("  4. Batch Processing")
        print("  5. Detailed Analysis")
        print("  6. Component Usage")
        print("  7. Risk Assessment")
        print("  8. Custom Weights")


if __name__ == "__main__":
    main()
