# main.py
"""
AI Fake News Detector - Main Entry Point
"""

import sys
import os
import logging

# ADD THIS - Same as test.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'services'))

# Now import
from services import DetectionOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Simple command line interface"""
    
    # Initialize orchestrator
    logger.info("Initializing AI Text Detector...")
    try:
        orchestrator = DetectionOrchestrator(
            model_path="models/bert/final_model",
            device="cpu",
            confidence_threshold=0.5
        )
        
        # Warm up
        logger.info("Warming up model...")
        orchestrator.warm_up()
        
        print("\n" + "=" * 60)
        print("AI FAKE NEWS DETECTOR")
        print("=" * 60)
        
        # Interactive mode
        while True:
            print("\nEnter text to analyze (or 'quit' to exit):")
            user_input = input("> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                print("Please enter some text.")
                continue
            
            # Analyze text
            logger.info("Analyzing text...")
            try:
                result = orchestrator.analyze_text(user_input)
                
                print("\n" + "-" * 40)
                print(f"RESULT: {'AI-GENERATED' if result['is_ai_generated'] else 'HUMAN-WRITTEN'}")
                print(f"Confidence: {result['confidence']:.2%}")
                print(f"Level: {result.get('confidence_level', 'Unknown')}")
                print("-" * 40)
                
                if 'explanation' in result:
                    print(f"\nExplanation:")
                    print(result['explanation']['interpretation'])
                    
                    # Show key indicators
                    if result['explanation']['key_indicators']:
                        print(f"\nKey Indicators:")
                        for indicator in result['explanation']['key_indicators']:
                            print(f"  • {indicator['name']}: {indicator['suggests']}")
                
                # Show suggestions if available
                if 'explanation' in result and 'suggestions' in result['explanation']:
                    print(f"\nSuggestions:")
                    for suggestion in result['explanation']['suggestions']:
                        print(f"  {suggestion}")
                
            except Exception as e:
                logger.error(f"Error during analysis: {e}")
                print("Analysis failed. Please try again.")
        
        # Show final statistics
        stats = orchestrator.get_session_statistics()
        print(f"\nSession Summary:")
        print(f"  Texts analyzed: {stats['total_detections']}")
        print(f"  Cache hit rate: {stats['inference_statistics']['cache_hit_rate']:.2%}")
        
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()