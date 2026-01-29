"""
Explainer Service for Fake News Detection
Provides interpretability and explanations for model predictions
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter
import re


class PredictionExplainer:
    """
    Provides explanations for model predictions
    """
    
    def __init__(self):
        """Initialize explainer"""
        # Common fake news indicators
        self.fake_indicators = {
            'clickbait': [
                'you won\'t believe', 'shocking', 'unbelievable', 'miracle',
                'scientists hate', 'doctors hate', 'secret', 'they don\'t want you to know',
                'breaking', 'urgent', 'alert'
            ],
            'emotional': [
                '!!!', 'outrageous', 'disgusting', 'terrifying', 'devastating',
                'horrific', 'amazing', 'incredible'
            ],
            'unreliable_sources': [
                'anonymous source', 'insider says', 'rumor has it', 'allegedly',
                'sources claim', 'it is reported'
            ],
            'sensational': [
                'all caps', 'excessive punctuation', 'conspiracy', 'cover-up',
                'mainstream media', 'wake up'
            ]
        }
        
        # Credibility indicators
        self.credibility_indicators = {
            'source_attribution': [
                'according to', 'study shows', 'research indicates',
                'published in', 'peer-reviewed', 'data from'
            ],
            'balanced_language': [
                'however', 'although', 'on the other hand', 'while',
                'experts say', 'evidence suggests'
            ],
            'specific_details': [
                'date', 'location', 'statistics', 'names', 'quotes'
            ]
        }
    
    def explain_prediction(self, 
                          text: str, 
                          prediction: str, 
                          confidence: float,
                          model_name: str = "Unknown") -> Dict:
        """
        Generate explanation for a prediction
        
        Args:
            text: Input text
            prediction: Model prediction (FAKE/REAL)
            confidence: Prediction confidence
            model_name: Name of the model
            
        Returns:
            Dictionary with explanation components
        """
        explanation = {
            'model': model_name,
            'prediction': prediction,
            'confidence': confidence,
            'text_features': self._analyze_text_features(text),
            'indicators_found': self._find_indicators(text),
            'readability': self._assess_readability(text),
            'sentiment_analysis': self._analyze_sentiment(text),
            'summary': ""
        }
        
        # Generate summary
        explanation['summary'] = self._generate_explanation_summary(explanation)
        
        return explanation
    
    def _analyze_text_features(self, text: str) -> Dict:
        """Analyze basic text features"""
        words = text.split()
        sentences = text.split('.')
        
        # Count caps
        caps_count = sum(1 for char in text if char.isupper())
        caps_ratio = caps_count / len(text) if text else 0
        
        # Count exclamations and questions
        exclamations = text.count('!')
        questions = text.count('?')
        
        # Count all caps words
        all_caps_words = sum(1 for word in words if word.isupper() and len(word) > 1)
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'caps_ratio': caps_ratio,
            'exclamation_marks': exclamations,
            'question_marks': questions,
            'all_caps_words': all_caps_words
        }
    
    def _find_indicators(self, text: str) -> Dict:
        """Find fake news and credibility indicators in text"""
        text_lower = text.lower()
        
        found_indicators = {
            'fake_indicators': {
                'clickbait': [],
                'emotional': [],
                'unreliable_sources': [],
                'sensational': []
            },
            'credibility_indicators': {
                'source_attribution': [],
                'balanced_language': [],
                'specific_details': []
            }
        }
        
        # Check fake indicators
        for category, phrases in self.fake_indicators.items():
            for phrase in phrases:
                if phrase in text_lower:
                    found_indicators['fake_indicators'][category].append(phrase)
        
        # Check credibility indicators
        for category, phrases in self.credibility_indicators.items():
            for phrase in phrases:
                if phrase in text_lower:
                    found_indicators['credibility_indicators'][category].append(phrase)
        
        # Check for URLs (often credibility indicator)
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        if urls:
            found_indicators['credibility_indicators']['specific_details'].extend(['URL found'] * len(urls))
        
        return found_indicators
    
    def _assess_readability(self, text: str) -> Dict:
        """Assess text readability"""
        words = text.split()
        sentences = text.split('.')
        
        if not words or not sentences:
            return {'score': 0, 'level': 'Unable to assess'}
        
        # Simple readability metrics
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = np.mean([len(w) for w in words])
        
        # Rough complexity score (0-100)
        complexity = min(100, (avg_sentence_length * 2 + avg_word_length * 5))
        
        if complexity < 30:
            level = "Very Easy"
        elif complexity < 50:
            level = "Easy"
        elif complexity < 70:
            level = "Moderate"
        else:
            level = "Complex"
        
        return {
            'complexity_score': complexity,
            'level': level,
            'avg_sentence_length': avg_sentence_length,
            'avg_word_length': avg_word_length
        }
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Basic sentiment analysis"""
        # Simple sentiment words (this is basic, consider using a proper sentiment library)
        positive_words = [
            'good', 'great', 'excellent', 'positive', 'fortunate', 'correct', 'superior',
            'amazing', 'wonderful', 'fantastic'
        ]
        negative_words = [
            'bad', 'terrible', 'awful', 'negative', 'unfortunate', 'wrong', 'inferior',
            'horrible', 'disgusting', 'evil'
        ]
        
        text_lower = text.lower()
        words = text_lower.split()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = "Positive"
        elif negative_count > positive_count:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        return {
            'sentiment': sentiment,
            'positive_words': positive_count,
            'negative_words': negative_count,
            'sentiment_ratio': (positive_count - negative_count) / len(words) if words else 0
        }
    
    def _generate_explanation_summary(self, explanation: Dict) -> str:
        """Generate human-readable summary of explanation"""
        lines = []
        
        pred = explanation['prediction']
        conf = explanation['confidence']
        
        lines.append(f"The {explanation['model']} model predicts this is {pred} news with {conf:.1%} confidence.")
        
        # Text features
        features = explanation['text_features']
        if features['all_caps_words'] > 3:
            lines.append(f"⚠️  The text contains {features['all_caps_words']} all-caps words, which is common in sensational content.")
        
        if features['exclamation_marks'] > 5:
            lines.append(f"⚠️  Excessive punctuation detected ({features['exclamation_marks']} exclamation marks).")
        
        # Indicators
        fake_count = sum(len(v) for v in explanation['indicators_found']['fake_indicators'].values())
        cred_count = sum(len(v) for v in explanation['indicators_found']['credibility_indicators'].values())
        
        if fake_count > 0:
            lines.append(f"⚠️  Found {fake_count} fake news indicators (clickbait phrases, emotional language, etc.).")
        
        if cred_count > 0:
            lines.append(f"✓ Found {cred_count} credibility indicators (source attribution, balanced language, etc.).")
        
        # Readability
        readability = explanation['readability']
        if readability['level'] == "Very Easy":
            lines.append(f"📖 Text readability: {readability['level']} - may be oversimplified.")
        
        # Sentiment
        sentiment = explanation['sentiment_analysis']
        if abs(sentiment['sentiment_ratio']) > 0.05:
            lines.append(f"😊 Text sentiment: {sentiment['sentiment']} - heavily emotional content.")
        
        return " ".join(lines)
    
    def explain_ensemble(self, 
                        text: str,
                        individual_predictions: Dict[str, Dict],
                        ensemble_result) -> Dict:
        """
        Provide explanation for ensemble prediction
        
        Args:
            text: Input text
            individual_predictions: Predictions from each model
            ensemble_result: EnsembleResult object
            
        Returns:
            Dictionary with comprehensive explanation
        """
        explanation = {
            'ensemble': {
                'prediction': ensemble_result.final_prediction,
                'confidence': ensemble_result.confidence,
                'consensus': ensemble_result.consensus_score,
                'method': ensemble_result.method_used
            },
            'model_explanations': {},
            'agreement_analysis': self._analyze_model_agreement(individual_predictions),
            'text_analysis': self._analyze_text_features(text),
            'key_findings': []
        }
        
        # Get explanation for each model
        for model_name, pred in individual_predictions.items():
            if pred.get('prediction') not in ['ERROR', None]:
                explanation['model_explanations'][model_name] = self.explain_prediction(
                    text, 
                    pred['prediction'], 
                    pred['confidence'],
                    model_name
                )
        
        # Generate key findings
        explanation['key_findings'] = self._generate_key_findings(
            text, 
            individual_predictions, 
            ensemble_result
        )
        
        return explanation
    
    def _analyze_model_agreement(self, predictions: Dict[str, Dict]) -> Dict:
        """Analyze agreement between models"""
        valid_preds = {k: v for k, v in predictions.items() if v.get('prediction') not in ['ERROR', None]}
        
        if not valid_preds:
            return {'agreement': 'N/A', 'details': 'No valid predictions'}
        
        pred_counts = Counter([p['prediction'] for p in valid_preds.values()])
        
        agreement_level = "Perfect" if len(pred_counts) == 1 else "Partial" if len(pred_counts) == 2 else "No"
        
        return {
            'agreement': agreement_level,
            'prediction_distribution': dict(pred_counts),
            'num_models': len(valid_preds),
            'confidence_variance': np.var([p['confidence'] for p in valid_preds.values()])
        }
    
    def _generate_key_findings(self, 
                               text: str,
                               predictions: Dict[str, Dict],
                               ensemble_result) -> List[str]:
        """Generate key findings from analysis"""
        findings = []
        
        # Model agreement
        all_preds = [p['prediction'] for p in predictions.values() if p.get('prediction') not in ['ERROR', None]]
        if len(set(all_preds)) == 1:
            findings.append(f"✓ All models agree: {all_preds[0]}")
        else:
            fake_count = all_preds.count('FAKE')
            real_count = all_preds.count('REAL')
            findings.append(f"⚠️  Models disagree: {fake_count} predict FAKE, {real_count} predict REAL")
        
        # Confidence analysis
        confidences = [p['confidence'] for p in predictions.values() if 'confidence' in p]
        if confidences:
            avg_conf = np.mean(confidences)
            if avg_conf > 0.8:
                findings.append(f"✓ High average confidence: {avg_conf:.1%}")
            elif avg_conf < 0.6:
                findings.append(f"⚠️  Low average confidence: {avg_conf:.1%} - results uncertain")
        
        # Text features
        features = self._analyze_text_features(text)
        if features['all_caps_words'] > 5:
            findings.append("⚠️  Excessive use of all-caps words detected")
        if features['exclamation_marks'] > 10:
            findings.append("⚠️  Excessive exclamation marks detected")
        
        # Indicators
        indicators = self._find_indicators(text)
        fake_indicators = sum(len(v) for v in indicators['fake_indicators'].values())
        if fake_indicators > 3:
            findings.append(f"⚠️  Multiple fake news indicators found ({fake_indicators} total)")
        
        return findings
    
    def format_detailed_explanation(self, explanation: Dict) -> str:
        """Format detailed explanation as readable text"""
        lines = []
        lines.append("="*70)
        lines.append("DETAILED EXPLANATION")
        lines.append("="*70)
        
        # Ensemble results
        if 'ensemble' in explanation:
            ens = explanation['ensemble']
            lines.append(f"\n🎯 Ensemble Prediction: {ens['prediction']}")
            lines.append(f"   Confidence: {ens['confidence']:.2%}")
            lines.append(f"   Consensus: {ens['consensus']:.2%}")
            lines.append(f"   Method: {ens['method']}")
        
        # Agreement analysis
        if 'agreement_analysis' in explanation:
            agree = explanation['agreement_analysis']
            lines.append(f"\n🤝 Model Agreement: {agree['agreement']}")
            lines.append(f"   Distribution: {agree['prediction_distribution']}")
        
        # Key findings
        if 'key_findings' in explanation:
            lines.append("\n🔍 Key Findings:")
            for finding in explanation['key_findings']:
                lines.append(f"   • {finding}")
        
        # Text analysis
        if 'text_analysis' in explanation:
            analysis = explanation['text_analysis']
            lines.append(f"\n📝 Text Analysis:")
            lines.append(f"   Word count: {analysis['word_count']}")
            lines.append(f"   Sentence count: {analysis['sentence_count']}")
            lines.append(f"   All-caps words: {analysis['all_caps_words']}")
            lines.append(f"   Exclamation marks: {analysis['exclamation_marks']}")
        
        lines.append("="*70)
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Test explainer
    print("Testing Prediction Explainer\n")
    
    explainer = PredictionExplainer()
    
    # Test text
    test_text = """
    BREAKING!!! You WON'T BELIEVE what scientists just discovered! 
    This SHOCKING revelation will change EVERYTHING! Anonymous sources 
    claim that this miracle cure has been HIDDEN from the public!!!
    """
    
    # Test explanation
    explanation = explainer.explain_prediction(
        test_text,
        "FAKE",
        0.92,
        "Test Model"
    )
    
    print("Explanation:")
    print(explanation['summary'])
    print("\nIndicators found:")
    print(f"Fake indicators: {sum(len(v) for v in explanation['indicators_found']['fake_indicators'].values())}")
    print(f"Credibility indicators: {sum(len(v) for v in explanation['indicators_found']['credibility_indicators'].values())}")
