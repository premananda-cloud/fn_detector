#!/usr/bin/env python3
"""
Simple Interactive Demo for Fake News Detection System
Run this without any arguments for a quick demo
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from main import FakeNewsDetector


def print_banner():
    """Print welcome banner"""
    print("\n" + "="*70)
    print(" "*15 + "FAKE NEWS DETECTION SYSTEM")
    print(" "*20 + "Interactive Demo")
    print("="*70)


def demo_mode():
    """Run a quick demo with sample texts"""
    print_banner()
    print("\n🚀 Starting Quick Demo...\n")
    
    # Initialize detector
    print("Initializing detector...")
    try:
        detector = FakeNewsDetector(verbose=False)
        print("✓ Detector initialized successfully!\n")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        print("\nPlease ensure model files are in the correct locations:")
        print("  - models/bert/final_model/")
        print("  - models/roberta/final_model/")
        print("  - models/tf_idf/")
        return
    
    # Sample texts
    samples = [
        {
            'label': 'Legitimate News',
            'text': 'According to a study published in Nature journal, researchers at MIT have developed a new solar cell technology that improves efficiency by 15 percent. The peer-reviewed research was conducted over three years with detailed data analysis.'
        },
        {
            'label': 'Suspicious Content',
            'text': 'SHOCKING!!! Scientists HATE this ONE WEIRD TRICK! You WON\'T BELIEVE what happens next! Anonymous sources reveal SECRET that changes EVERYTHING!!!'
        }
    ]
    
    for i, sample in enumerate(samples, 1):
        print(f"\n{'='*70}")
        print(f"SAMPLE {i}: {sample['label']}")
        print(f"{'='*70}")
        print(f"\nText: {sample['text'][:100]}...")
        
        # Get prediction
        result = detector.predict(sample['text'], explain=False)
        
        # Display result
        pred = result['final_prediction']
        conf = result['confidence']
        risk = result['risk_level']
        
        emoji = "🚫" if pred == "FAKE" else "✅" if pred == "REAL" else "❓"
        
        print(f"\n{emoji} PREDICTION: {pred}")
        print(f"📊 Confidence: {conf:.1%}")
        print(f"⚠️  Risk Level: {risk}")
        
        # Show individual models
        print("\nIndividual Model Votes:")
        for model_name, pred_data in result['individual_predictions'].items():
            model_pred = pred_data.get('prediction', 'ERROR')
            model_conf = pred_data.get('confidence', 0)
            print(f"  • {model_name}: {model_pred} ({model_conf:.1%})")


def interactive_mode():
    """Run interactive mode"""
    print_banner()
    print("\n💬 Interactive Mode")
    print("Enter text to analyze, or 'demo' for sample analysis")
    print("Type 'quit' or 'exit' to quit\n")
    print("="*70)
    
    # Initialize detector
    try:
        detector = FakeNewsDetector(verbose=False)
        print("\n✓ System ready!\n")
    except Exception as e:
        print(f"\n✗ Initialization failed: {e}\n")
        return
    
    while True:
        try:
            # Get input
            user_input = input("\nEnter text (or command): ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'demo':
                demo_mode()
                continue
            
            if user_input.lower() in ['help', 'h', '?']:
                print("\nCommands:")
                print("  demo  - Run demo with sample texts")
                print("  help  - Show this help")
                print("  quit  - Exit program")
                print("\nOr enter any text to analyze it")
                continue
            
            # Analyze text
            print("\n🔍 Analyzing...")
            result = detector.predict(user_input, explain=False)
            
            # Display result
            print("\n" + "-"*70)
            pred = result['final_prediction']
            conf = result['confidence']
            risk = result['risk_level']
            
            emoji = "🚫" if pred == "FAKE" else "✅" if pred == "REAL" else "❓"
            
            print(f"{emoji} PREDICTION: {pred}")
            print(f"📊 Confidence: {conf:.1%}")
            print(f"⚠️  Risk Level: {risk}")
            print(f"🤝 Consensus: {result['consensus_score']:.1%}")
            
            # Show model agreement
            predictions_list = [p.get('prediction') for p in result['individual_predictions'].values()]
            fake_votes = predictions_list.count('FAKE')
            real_votes = predictions_list.count('REAL')
            
            print(f"\nModel Votes: {fake_votes} FAKE, {real_votes} REAL")
            print("-"*70)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Fake News Detection - Simple Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py              # Interactive mode
  python demo.py --demo       # Quick demo with samples
  python demo.py --help       # Show this help

For full features, use main.py:
  python main.py --text "Your text here"
  python main.py --interactive
        """
    )
    
    parser.add_argument('--demo', '-d', action='store_true',
                       help='Run quick demo mode')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run interactive mode (default)')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_mode()
    else:
        # Default to interactive
        interactive_mode()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
