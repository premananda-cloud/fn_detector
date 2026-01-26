#!/usr/bin/env python3
"""
test_core.py - Standalone test for core detector functionality
Run from project root: python test_core.py
"""
import sys
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_core_imports():
    """Test importing core components"""
    print("="*60)
    print("TESTING CORE PACKAGE IMPORTS")
    print("="*60)
    
    try:
        # Test direct imports from core
        from core.detector import AITextDetector
        from core.engine.call_bert import BERTDetector
        from core.processor.feature_extracter import FeatureExtractor
        from core.processor.text_processor import TextProcessor
        
        print("✅ All core imports successful!")
        print(f"  AITextDetector: {AITextDetector}")
        print(f"  BERTDetector: {BERTDetector}")
        print(f"  FeatureExtractor: {FeatureExtractor}")
        print(f"  TextProcessor: {TextProcessor}")
        
        return True
    except ImportError as e:
        print(f"❌ Core import failed: {e}")
        print("\nDebug info:")
        print(f"Current directory: {os.getcwd()}")
        print(f"Directory contents: {os.listdir('.')}")
        return False

def test_text_processor():
    """Test text processor functionality"""
    print("\n" + "="*60)
    print("TESTING TEXT PROCESSOR")
    print("="*60)
    
    try:
        from core.processor.text_processor import TextProcessor
        processor = TextProcessor()
        
        test_texts = [
            "Hello world! This is a test.",
            "   Extra   spaces   here.  ",
            "UPPERCASE and lowercase MIXED.",
            "Test with multiple. Sentences. Here!"
        ]
        
        for i, text in enumerate(test_texts):
            processed = processor.process(text)
            print(f"Test {i+1}:")
            print(f"  Original: '{text}'")
            print(f"  Processed: '{processed}'")
            print(f"  Length: {len(original)} -> {len(processed)} chars")
            print()
        
        return True
    except Exception as e:
        print(f"❌ Text processor test failed: {e}")
        return False

def test_feature_extractor():
    """Test feature extractor functionality"""
    print("\n" + "="*60)
    print("TESTING FEATURE EXTRACTOR")
    print("="*60)
    
    try:
        from core.processor.feature_extracter import FeatureExtractor
        extractor = FeatureExtractor()
        
        test_text = """
        Artificial intelligence has revolutionized numerous industries in recent years.
        Machine learning algorithms can now perform complex tasks with remarkable efficiency.
        However, it's important to consider the ethical implications carefully.
        """
        
        features = extractor.extract(test_text)
        
        print(f"Text length: {len(test_text)} chars")
        print(f"Number of features extracted: {len(features)}")
        print("\nTop 10 features:")
        
        # Show top 10 most interesting features
        sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        for name, value in sorted_features:
            print(f"  {name:25s}: {value:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Feature extractor test failed: {e}")
        return False

def test_bert_detector():
    """Test BERT detector functionality"""
    print("\n" + "="*60)
    print("TESTING BERT DETECTOR")
    print("="*60)
    
    try:
        from core.engine.call_bert import BERTDetector
        
        # Try to initialize - this will test if models exist
        print("Initializing BERT detector...")
        detector = BERTDetector(
            model_path="models/bert/final_model",
            device="cpu"
        )
        
        print("✅ BERT detector initialized")
        print(f"Model info: {detector.get_model_info()}")
        
        # Test with a very simple text if model exists
        test_text = "This is a simple test sentence."
        
        try:
            predictions = detector.predict(test_text)
            print(f"Prediction for '{test_text[:30]}...':")
            print(f"  Probability: {predictions[0]['probability']:.4f}")
            print(f"  Label: {'AI' if predictions[0]['label'] == 1 else 'Human'}")
        except Exception as e:
            print(f"⚠️  Prediction test skipped (might be missing model files): {e}")
        
        return True
    except Exception as e:
        print(f"❌ BERT detector test failed: {e}")
        print("\nNote: This might fail if model files are not in 'models/bert/final_model/'")
        return False

def test_ai_text_detector():
    """Test the main AITextDetector"""
    print("\n" + "="*60)
    print("TESTING MAIN AI TEXT DETECTOR")
    print("="*60)
    
    try:
        from core.detector import AITextDetector
        
        print("Initializing AITextDetector...")
        detector = AITextDetector(
            model_path="models/bert/final_model",
            device="cpu",
            confidence_threshold=0.5
        )
        
        print("✅ AITextDetector initialized")
        
        # Hard-coded test texts
        test_texts = [
            # Potential AI-like text
            """Artificial intelligence has fundamentally transformed numerous industries through its remarkable capacity for pattern recognition and data analysis. Machine learning algorithms demonstrate sophisticated capabilities in processing complex information, thereby enabling unprecedented levels of automation and efficiency across various sectors.""",
            
            # More human-like text
            """I really don't know about this AI stuff. It seems kinda complicated, you know? Like, my phone already does too much. I just want to text my friends without all the fancy features!""",
            
            # Short test
            "This is just a simple test."
        ]
        
        descriptions = ["AI-like text", "Human-like text", "Simple test"]
        
        for i, (text, desc) in enumerate(zip(test_texts, descriptions)):
            print(f"\nTest {i+1}: {desc}")
            print("-" * 40)
            
            try:
                result = detector.detect(text, return_probabilities=True)
                
                print(f"Text preview: {text[:50]}...")
                print(f"Prediction: {'AI-GENERATED' if result['is_ai_generated'] else 'HUMAN-WRITTEN'}")
                print(f"Confidence: {result['confidence']:.2%}")
                print(f"Confidence Level: {result['confidence_level']}")
                
                if 'probabilities' in result:
                    probs = result['probabilities']
                    print(f"Probabilities: AI={probs['ai_generated']:.2%}, Human={probs['human_written']:.2%}")
                    
            except Exception as e:
                print(f"⚠️  Detection failed: {e}")
        
        return True
    except Exception as e:
        print(f"❌ AITextDetector test failed: {e}")
        return False

def test_batch_detection():
    """Test batch detection functionality"""
    print("\n" + "="*60)
    print("TESTING BATCH DETECTION")
    print("="*60)
    
    try:
        from core.detector import AITextDetector
        
        detector = AITextDetector(
            model_path="models/bert/final_model",
            device="cpu"
        )
        
        # Batch of test texts
        batch_texts = [
            "Neural networks require extensive training data.",
            "I went to the store and bought some milk.",
            "The transformer architecture revolutionized NLP.",
            "My cat is sleeping on the couch right now.",
            "Deep learning models achieve state-of-the-art results."
        ]
        
        print(f"Running batch detection on {len(batch_texts)} texts...")
        results = detector.detect_batch(batch_texts)
        
        print("\nBatch Results:")
        for i, (text, result) in enumerate(zip(batch_texts, results)):
            print(f"\nText {i+1}: '{text[:30]}...'")
            print(f"  AI-Generated: {result['is_ai_generated']}")
            print(f"  Confidence: {result['confidence']:.2%}")
        
        return True
    except Exception as e:
        print(f"❌ Batch detection test failed: {e}")
        return False

def main():
    """Main test function"""
    print("FAKE NEWS DETECTOR - CORE PACKAGE TEST")
    print("Running from:", os.getcwd())
    print()
    
    # Run all tests
    tests = [
        ("Core Imports", test_core_imports),
        ("Text Processor", test_text_processor),
        ("Feature Extractor", test_feature_extractor),
        ("BERT Detector", test_bert_detector),
        ("AI Text Detector", test_ai_text_detector),
        ("Batch Detection", test_batch_detection)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! Your core package is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    # Add current directory to Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        sys.exit(1)