"""
Main entry point for Fake News Detection System
Integrates inference, orchestration, and explanation services
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, List

# Add services to path
sys.path.append(str(Path(__file__).parent / 'services'))

from inference import MultiModelInference
from orchestrator import PredictionOrchestrator, EnsembleMethod
from explainer import PredictionExplainer


class FakeNewsDetector:
    """
    Main class for fake news detection system
    Integrates all components: inference, orchestration, and explanation
    """
    
    def __init__(self, 
                 use_bert: bool = True,
                 use_roberta: bool = True,
                 use_tfidf: bool = True,
                 ensemble_method: str = 'confidence_weighted',
                 verbose: bool = True):
        """
        Initialize fake news detector
        
        Args:
            use_bert: Whether to use BERT model
            use_roberta: Whether to use RoBERTa model
            use_tfidf: Whether to use TF-IDF model
            ensemble_method: Ensemble method to use
            verbose: Print detailed output
        """
        self.verbose = verbose
        
        if self.verbose:
            print("="*70)
            print("Initializing Fake News Detection System")
            print("="*70)
        
        # Initialize inference system
        try:
            self.inference = MultiModelInference(
                use_bert=use_bert,
                use_roberta=use_roberta,
                use_tfidf=use_tfidf
            )
        except Exception as e:
            print(f"Error initializing inference system: {e}")
            raise
        
        # Initialize orchestrator
        try:
            ensemble_method_enum = EnsembleMethod[ensemble_method.upper()]
        except KeyError:
            print(f"Invalid ensemble method: {ensemble_method}")
            print(f"Valid methods: {[e.value for e in EnsembleMethod]}")
            ensemble_method_enum = EnsembleMethod.CONFIDENCE_WEIGHTED
        
        self.orchestrator = PredictionOrchestrator(ensemble_method=ensemble_method_enum)
        
        # Initialize explainer
        self.explainer = PredictionExplainer()
        
        if self.verbose:
            print("\n✓ System initialized successfully!")
            print(f"✓ Loaded models: {list(self.inference.models.keys())}")
            print(f"✓ Ensemble method: {ensemble_method_enum.value}")
    
    def predict(self, text: str, explain: bool = True) -> dict:
        """
        Predict whether text is fake news
        
        Args:
            text: Text to analyze
            explain: Whether to include explanations
            
        Returns:
            Dictionary with prediction results
        """
        if not text or not text.strip():
            return {
                'error': 'Empty text provided',
                'final_prediction': 'ERROR'
            }
        
        # Get predictions from all models
        if self.verbose:
            print("\n" + "="*70)
            print("Running predictions...")
            print("="*70)
        
        predictions = self.inference.predict_all(text)
        
        # Get ensemble prediction
        ensemble_result = self.orchestrator.ensemble_predict(predictions)
        
        # Prepare result
        result = {
            'text': text[:200] + "..." if len(text) > 200 else text,
            'final_prediction': ensemble_result.final_prediction,
            'confidence': ensemble_result.confidence,
            'consensus_score': ensemble_result.consensus_score,
            'risk_level': ensemble_result.risk_level,
            'ensemble_method': ensemble_result.method_used,
            'individual_predictions': predictions
        }
        
        # Add explanation if requested
        if explain:
            explanation = self.explainer.explain_ensemble(
                text,
                predictions,
                ensemble_result
            )
            result['explanation'] = explanation
        
        return result
    
    def predict_batch(self, texts: List[str], explain: bool = False) -> List[dict]:
        """
        Predict on multiple texts
        
        Args:
            texts: List of texts to analyze
            explain: Whether to include explanations
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i, text in enumerate(texts, 1):
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"Processing text {i}/{len(texts)}")
                print(f"{'='*70}")
            
            result = self.predict(text, explain=explain)
            results.append(result)
        
        return results
    
    def print_result(self, result: dict, detailed: bool = True):
        """
        Print formatted result
        
        Args:
            result: Prediction result dictionary
            detailed: Whether to print detailed information
        """
        if 'error' in result:
            print(f"\n❌ Error: {result['error']}")
            return
        
        print("\n" + "="*70)
        print("FAKE NEWS DETECTION RESULT")
        print("="*70)
        
        # Main result
        pred = result['final_prediction']
        emoji = "🚫" if pred == "FAKE" else "✅" if pred == "REAL" else "❓"
        
        print(f"\n{emoji} PREDICTION: {pred}")
        print(f"📊 Confidence: {result['confidence']:.2%}")
        print(f"🤝 Consensus: {result['consensus_score']:.2%}")
        print(f"⚠️  Risk Level: {result['risk_level']}")
        
        if detailed:
            # Individual model predictions
            print("\n" + "-"*70)
            print("INDIVIDUAL MODEL PREDICTIONS")
            print("-"*70)
            
            for model_name, pred in result['individual_predictions'].items():
                print(f"\n{model_name}:")
                print(f"  Prediction: {pred.get('prediction', 'N/A')}")
                print(f"  Confidence: {pred.get('confidence', 0):.2%}")
                
                if 'probabilities' in pred:
                    probs = pred['probabilities']
                    print(f"  Probabilities: REAL={probs['real']:.2%}, FAKE={probs['fake']:.2%}")
        
        # Explanation
        if 'explanation' in result and detailed:
            print("\n" + "-"*70)
            print("KEY FINDINGS")
            print("-"*70)
            for finding in result['explanation'].get('key_findings', []):
                print(f"  • {finding}")
        
        # Recommendation
        print("\n" + "-"*70)
        print("RECOMMENDATION")
        print("-"*70)
        print(self._generate_recommendation(result))
        
        print("="*70 + "\n")
    
    def _generate_recommendation(self, result: dict) -> str:
        """Generate recommendation based on result"""
        pred = result['final_prediction']
        conf = result['confidence']
        risk = result['risk_level']
        
        if pred == "FAKE":
            if risk == "HIGH":
                return "⛔ This content is likely FAKE NEWS. Do NOT share or trust this information."
            elif risk == "MEDIUM":
                return "⚠️  This content may be FAKE. Verify from trusted sources before sharing."
            else:
                return "❓ Unclear classification. Further investigation recommended."
        elif pred == "REAL":
            if risk == "LOW":
                return "✅ This content appears to be LEGITIMATE. High confidence from multiple models."
            elif risk == "MEDIUM":
                return "✓ This content is likely REAL, but verify important claims independently."
            else:
                return "❓ Low confidence prediction. Cross-check with other sources."
        else:
            return "❓ Unable to classify. Models disagree. Manual review required."
    
    def analyze_from_file(self, filepath: str, explain: bool = True) -> dict:
        """
        Analyze text from a file
        
        Args:
            filepath: Path to text file
            explain: Whether to include explanations
            
        Returns:
            Prediction result
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            return self.predict(text, explain=explain)
        except Exception as e:
            return {
                'error': f'Failed to read file: {e}',
                'final_prediction': 'ERROR'
            }


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description='Fake News Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect fake news from text
  python main.py --text "Breaking news: Scientists discover..."
  
  # Analyze from file
  python main.py --file article.txt
  
  # Use specific models
  python main.py --text "News text..." --no-tfidf
  
  # Choose ensemble method
  python main.py --text "News..." --ensemble majority_vote
  
  # Batch mode
  python main.py --batch texts.txt
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--text', '-t', type=str, help='Text to analyze')
    input_group.add_argument('--file', '-f', type=str, help='File containing text to analyze')
    input_group.add_argument('--batch', '-b', type=str, help='File with multiple texts (one per line)')
    input_group.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    
    # Model selection
    parser.add_argument('--no-bert', action='store_true', help='Disable BERT model')
    parser.add_argument('--no-roberta', action='store_true', help='Disable RoBERTa model')
    parser.add_argument('--no-tfidf', action='store_true', help='Disable TF-IDF model')
    
    # Ensemble method
    parser.add_argument('--ensemble', '-e', type=str, 
                       default='confidence_weighted',
                       choices=[e.value for e in EnsembleMethod],
                       help='Ensemble method to use')
    
    # Output options
    parser.add_argument('--simple', '-s', action='store_true', help='Simple output (less detailed)')
    parser.add_argument('--no-explain', action='store_true', help='Disable explanations')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode (minimal output)')
    
    args = parser.parse_args()
    
    # Initialize detector
    try:
        detector = FakeNewsDetector(
            use_bert=not args.no_bert,
            use_roberta=not args.no_roberta,
            use_tfidf=not args.no_tfidf,
            ensemble_method=args.ensemble,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"Failed to initialize detector: {e}")
        return 1
    
    # Process input
    if args.text:
        # Single text prediction
        result = detector.predict(args.text, explain=not args.no_explain)
        detector.print_result(result, detailed=not args.simple)
    
    elif args.file:
        # File prediction
        result = detector.analyze_from_file(args.file, explain=not args.no_explain)
        detector.print_result(result, detailed=not args.simple)
    
    elif args.batch:
        # Batch prediction
        try:
            with open(args.batch, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            results = detector.predict_batch(texts, explain=not args.no_explain)
            
            # Print summary
            fake_count = sum(1 for r in results if r['final_prediction'] == 'FAKE')
            real_count = sum(1 for r in results if r['final_prediction'] == 'REAL')
            
            print("\n" + "="*70)
            print("BATCH PROCESSING SUMMARY")
            print("="*70)
            print(f"Total texts: {len(results)}")
            print(f"FAKE: {fake_count} ({fake_count/len(results)*100:.1f}%)")
            print(f"REAL: {real_count} ({real_count/len(results)*100:.1f}%)")
            print("="*70)
            
            # Print individual results if not quiet
            if not args.quiet:
                for i, result in enumerate(results, 1):
                    print(f"\n--- Text {i} ---")
                    detector.print_result(result, detailed=not args.simple)
        
        except Exception as e:
            print(f"Batch processing failed: {e}")
            return 1
    
    elif args.interactive:
        # Interactive mode
        print("\n" + "="*70)
        print("INTERACTIVE MODE")
        print("="*70)
        print("Enter text to analyze (type 'quit' to exit)")
        print("Type 'help' for options")
        print("="*70 + "\n")
        
        while True:
            try:
                text = input("\nEnter text: ").strip()
                
                if text.lower() == 'quit':
                    print("Goodbye!")
                    break
                
                if text.lower() == 'help':
                    print("\nCommands:")
                    print("  quit  - Exit interactive mode")
                    print("  help  - Show this help")
                    print("\nOr enter any text to analyze")
                    continue
                
                if not text:
                    print("Please enter some text")
                    continue
                
                result = detector.predict(text, explain=not args.no_explain)
                detector.print_result(result, detailed=not args.simple)
            
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
