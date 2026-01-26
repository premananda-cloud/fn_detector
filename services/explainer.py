"""
Explainer Service
Provides interpretability and explanations for AI text detection predictions
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict

from core.detector import AITextDetector
from core.processor.feature_extrater import FeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExplainerService:
    """
    Service for explaining AI text detection predictions
    Provides feature importance, attention visualization, and human-readable explanations
    """
    
    def __init__(
        self,
        model_path: str = "models/bert/final_model",
        device: str = "cpu"
    ):
        """
        Initialize explainer service
        
        Args:
            model_path: Path to the trained model
            device: Device for inference
        """
        self.model_path = model_path
        self.device = device
        
        logger.info("Initializing explainer service...")
        
        # Initialize detector and feature extractor
        self.detector = AITextDetector(
            model_path=model_path,
            device=device
        )
        self.feature_extractor = FeatureExtractor()
        
        # Feature importance thresholds for explanations
        self.feature_thresholds = {
            'high_lexical_diversity': 0.75,
            'low_lexical_diversity': 0.4,
            'high_transition_words': 0.05,
            'high_avg_sentence_length': 25,
            'low_sentence_variance': 5,
            'high_complex_words': 0.25,
            'low_contraction_ratio': 0.01,
            'high_function_words': 0.45
        }
        
        logger.info("✓ Explainer service initialized")
    
    def explain_prediction(
        self,
        text: str,
        include_features: bool = True,
        include_attention: bool = True,
        include_suggestions: bool = True
    ) -> Dict:
        """
        Generate comprehensive explanation for a prediction
        
        Args:
            text: Input text
            include_features: Include feature analysis
            include_attention: Include attention weights
            include_suggestions: Include writing suggestions
            
        Returns:
            Explanation dictionary
        """
        logger.info("Generating explanation...")
        
        explanation = {}
        
        # Get prediction
        result = self.detector.detect(
            text,
            return_probabilities=True,
            return_features=True
        )
        
        explanation['prediction'] = {
            'is_ai_generated': result['is_ai_generated'],
            'confidence': result['confidence'],
            'confidence_level': result['confidence_level']
        }
        
        # Feature-based explanation
        if include_features:
            feature_explanation = self._explain_features(result.get('features', {}))
            explanation['feature_analysis'] = feature_explanation
        
        # Attention-based explanation
        if include_attention:
            attention_explanation = self._explain_attention(text)
            explanation['attention_analysis'] = attention_explanation
        
        # Overall interpretation
        explanation['interpretation'] = self._generate_interpretation(
            result,
            result.get('features', {})
        )
        
        # Suggestions for improvement
        if include_suggestions:
            suggestions = self._generate_suggestions(
                result,
                result.get('features', {})
            )
            explanation['suggestions'] = suggestions
        
        # Key indicators
        explanation['key_indicators'] = self._identify_key_indicators(
            result.get('features', {})
        )
        
        return explanation
    
    def _explain_features(self, features: Dict) -> Dict:
        """
        Explain feature contributions to the prediction
        
        Args:
            features: Extracted features
            
        Returns:
            Feature explanation
        """
        if not features:
            return {}
        
        # Get feature importance summary
        summary = self.feature_extractor.get_feature_importance_summary(features)
        
        # Categorize features
        categorized = {
            'lexical_features': {},
            'syntactic_features': {},
            'stylistic_features': {},
            'readability_features': {},
            'coherence_features': {}
        }
        
        # Lexical
        categorized['lexical_features'] = {
            'lexical_diversity': features.get('lexical_diversity', 0),
            'unique_word_ratio': features.get('unique_word_ratio', 0),
            'stopword_ratio': features.get('stopword_ratio', 0),
            'function_word_ratio': features.get('function_word_ratio', 0)
        }
        
        # Syntactic
        categorized['syntactic_features'] = {
            'avg_sentence_length': features.get('avg_sentence_length', 0),
            'sentence_variance': features.get('std_sentence_length', 0),
            'transition_word_ratio': features.get('transition_word_ratio', 0),
            'punctuation_density': features.get('punctuation_density', 0)
        }
        
        # Stylistic
        categorized['stylistic_features'] = {
            'contraction_ratio': features.get('contraction_ratio', 0),
            'long_word_ratio': features.get('long_word_ratio', 0),
            'uppercase_ratio': features.get('uppercase_ratio', 0)
        }
        
        # Readability
        categorized['readability_features'] = {
            'flesch_reading_ease': features.get('flesch_reading_ease', 0),
            'flesch_kincaid_grade': features.get('flesch_kincaid_grade', 0),
            'complex_word_ratio': features.get('complex_word_ratio', 0)
        }
        
        # Coherence
        categorized['coherence_features'] = {
            'bigram_repetition': features.get('bigram_repetition', 0),
            'unique_bigram_ratio': features.get('unique_bigram_ratio', 0),
            'word_entropy': features.get('word_entropy', 0),
            'burstiness': features.get('burstiness', 0)
        }
        
        return {
            'categories': categorized,
            'summary_scores': summary,
            'notable_features': self._identify_notable_features(features)
        }
    
    def _identify_notable_features(self, features: Dict) -> List[Dict]:
        """
        Identify features that stand out
        
        Args:
            features: Feature dictionary
            
        Returns:
            List of notable features with explanations
        """
        notable = []
        
        # Check each feature against thresholds
        if features.get('lexical_diversity', 0) > self.feature_thresholds['high_lexical_diversity']:
            notable.append({
                'feature': 'lexical_diversity',
                'value': features['lexical_diversity'],
                'direction': 'high',
                'interpretation': 'Very high vocabulary diversity (more human-like)'
            })
        
        if features.get('lexical_diversity', 1) < self.feature_thresholds['low_lexical_diversity']:
            notable.append({
                'feature': 'lexical_diversity',
                'value': features['lexical_diversity'],
                'direction': 'low',
                'interpretation': 'Low vocabulary diversity (more AI-like)'
            })
        
        if features.get('transition_word_ratio', 0) > self.feature_thresholds['high_transition_words']:
            notable.append({
                'feature': 'transition_word_ratio',
                'value': features['transition_word_ratio'],
                'direction': 'high',
                'interpretation': 'High use of transition words (common in AI text)'
            })
        
        if features.get('avg_sentence_length', 0) > self.feature_thresholds['high_avg_sentence_length']:
            notable.append({
                'feature': 'avg_sentence_length',
                'value': features['avg_sentence_length'],
                'direction': 'high',
                'interpretation': 'Long average sentence length'
            })
        
        if features.get('std_sentence_length', 10) < self.feature_thresholds['low_sentence_variance']:
            notable.append({
                'feature': 'sentence_variance',
                'value': features['std_sentence_length'],
                'direction': 'low',
                'interpretation': 'Very consistent sentence lengths (more AI-like)'
            })
        
        if features.get('complex_word_ratio', 0) > self.feature_thresholds['high_complex_words']:
            notable.append({
                'feature': 'complex_word_ratio',
                'value': features['complex_word_ratio'],
                'direction': 'high',
                'interpretation': 'High proportion of complex words'
            })
        
        if features.get('contraction_ratio', 0) < self.feature_thresholds['low_contraction_ratio']:
            notable.append({
                'feature': 'contraction_ratio',
                'value': features['contraction_ratio'],
                'direction': 'low',
                'interpretation': 'Few contractions (more formal, AI-like)'
            })
        
        return notable
    
    def _explain_attention(self, text: str) -> Dict:
        """
        Explain model attention patterns
        
        Args:
            text: Input text
            
        Returns:
            Attention explanation
        """
        try:
            # Get attention weights from BERT detector
            attention_result = self.detector.bert_detector.predict_with_attention(text)
            
            return {
                'important_tokens': attention_result.get('important_tokens', []),
                'attention_summary': self._summarize_attention(
                    attention_result.get('important_tokens', [])
                )
            }
        except Exception as e:
            logger.warning(f"Could not generate attention explanation: {e}")
            return {
                'important_tokens': [],
                'attention_summary': "Attention analysis not available"
            }
    
    def _summarize_attention(self, important_tokens: List[Dict]) -> str:
        """
        Create human-readable summary of attention patterns
        
        Args:
            important_tokens: List of important tokens with weights
            
        Returns:
            Summary string
        """
        if not important_tokens:
            return "No significant attention patterns detected"
        
        # Get top 3 tokens
        top_tokens = important_tokens[:3]
        token_words = [t['token'] for t in top_tokens if t['token'] not in ['[CLS]', '[SEP]', '[PAD]']]
        
        if token_words:
            return f"Model focused most on: {', '.join(token_words)}"
        else:
            return "Model attention distributed broadly"
    
    def _generate_interpretation(self, result: Dict, features: Dict) -> str:
        """
        Generate human-readable interpretation
        
        Args:
            result: Prediction result
            features: Extracted features
            
        Returns:
            Interpretation string
        """
        confidence = result['confidence']
        is_ai = result['is_ai_generated']
        
        # Base interpretation
        if is_ai:
            if confidence > 0.9:
                base = "This text is very likely AI-generated."
            elif confidence > 0.75:
                base = "This text is likely AI-generated."
            else:
                base = "This text shows some characteristics of AI-generated content."
        else:
            if confidence < 0.1:
                base = "This text is very likely human-written."
            elif confidence < 0.25:
                base = "This text is likely human-written."
            else:
                base = "This text shows some characteristics of human writing."
        
        # Add feature-based reasoning
        reasons = []
        
        if features.get('lexical_diversity', 0) < 0.4:
            reasons.append("low vocabulary diversity")
        
        if features.get('std_sentence_length', 10) < 5:
            reasons.append("very consistent sentence structure")
        
        if features.get('transition_word_ratio', 0) > 0.05:
            reasons.append("frequent use of transition words")
        
        if features.get('contraction_ratio', 0) < 0.01:
            reasons.append("formal tone with few contractions")
        
        if features.get('complex_word_ratio', 0) > 0.25:
            reasons.append("high proportion of complex vocabulary")
        
        if reasons:
            interpretation = f"{base} Key indicators include: {', '.join(reasons)}."
        else:
            interpretation = base
        
        return interpretation
    
    def _generate_suggestions(self, result: Dict, features: Dict) -> List[str]:
        """
        Generate suggestions for making text more/less AI-like
        
        Args:
            result: Prediction result
            features: Extracted features
            
        Returns:
            List of suggestions
        """
        suggestions = []
        is_ai = result['is_ai_generated']
        
        if is_ai:
            # Suggestions to make text more human-like
            suggestions.append("To make the text appear more human:")
            
            if features.get('lexical_diversity', 0) < 0.5:
                suggestions.append("• Vary your vocabulary more - use synonyms and different word choices")
            
            if features.get('std_sentence_length', 10) < 5:
                suggestions.append("• Vary sentence lengths - mix short, punchy sentences with longer ones")
            
            if features.get('contraction_ratio', 0) < 0.01:
                suggestions.append("• Use more contractions (e.g., 'don't' instead of 'do not') for a casual tone")
            
            if features.get('transition_word_ratio', 0) > 0.05:
                suggestions.append("• Reduce formal transition words - write more naturally")
            
            suggestions.append("• Add personal touches, opinions, or rhetorical questions")
            suggestions.append("• Include occasional minor imperfections or colloquialisms")
        else:
            # Text is human-like
            suggestions.append("The text already appears human-written.")
            suggestions.append("To maintain authenticity:")
            suggestions.append("• Keep varying sentence structure and vocabulary")
            suggestions.append("• Continue using natural language patterns")
        
        return suggestions
    
    def _identify_key_indicators(self, features: Dict) -> List[Dict]:
        """
        Identify the most important indicators from features
        
        Args:
            features: Feature dictionary
            
        Returns:
            List of key indicator dictionaries
        """
        indicators = []
        
        # Lexical diversity indicator
        lexical_div = features.get('lexical_diversity', 0)
        if lexical_div < 0.4:
            indicators.append({
                'name': 'Vocabulary Repetition',
                'value': lexical_div,
                'impact': 'high',
                'suggests': 'AI-generated'
            })
        elif lexical_div > 0.75:
            indicators.append({
                'name': 'Vocabulary Diversity',
                'value': lexical_div,
                'impact': 'high',
                'suggests': 'Human-written'
            })
        
        # Sentence consistency indicator
        sentence_std = features.get('std_sentence_length', 10)
        if sentence_std < 5:
            indicators.append({
                'name': 'Sentence Uniformity',
                'value': sentence_std,
                'impact': 'medium',
                'suggests': 'AI-generated'
            })
        
        # Transition words indicator
        transition_ratio = features.get('transition_word_ratio', 0)
        if transition_ratio > 0.05:
            indicators.append({
                'name': 'Formal Transitions',
                'value': transition_ratio,
                'impact': 'medium',
                'suggests': 'AI-generated'
            })
        
        # Contraction usage indicator
        contraction_ratio = features.get('contraction_ratio', 0)
        if contraction_ratio < 0.01:
            indicators.append({
                'name': 'Formal Language',
                'value': contraction_ratio,
                'impact': 'low',
                'suggests': 'AI-generated'
            })
        elif contraction_ratio > 0.05:
            indicators.append({
                'name': 'Casual Language',
                'value': contraction_ratio,
                'impact': 'low',
                'suggests': 'Human-written'
            })
        
        return indicators
    
    def compare_texts(self, text1: str, text2: str) -> Dict:
        """
        Compare two texts and their predictions
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Comparison dictionary
        """
        logger.info("Comparing two texts...")
        
        # Get explanations for both
        explanation1 = self.explain_prediction(text1, include_attention=False)
        explanation2 = self.explain_prediction(text2, include_attention=False)
        
        # Extract features for comparison
        features1 = explanation1['feature_analysis']['categories']
        features2 = explanation2['feature_analysis']['categories']
        
        return {
            'text1': {
                'prediction': explanation1['prediction'],
                'key_indicators': explanation1['key_indicators']
            },
            'text2': {
                'prediction': explanation2['prediction'],
                'key_indicators': explanation2['key_indicators']
            },
            'comparison': self._compare_feature_categories(features1, features2),
            'verdict': self._generate_comparison_verdict(explanation1, explanation2)
        }
    
    def _compare_feature_categories(self, features1: Dict, features2: Dict) -> Dict:
        """Compare feature categories between two texts"""
        comparison = {}
        
        for category in features1.keys():
            if category in features2:
                cat_comparison = {}
                for feature in features1[category].keys():
                    if feature in features2[category]:
                        val1 = features1[category][feature]
                        val2 = features2[category][feature]
                        cat_comparison[feature] = {
                            'text1': val1,
                            'text2': val2,
                            'difference': abs(val1 - val2),
                            'similar': abs(val1 - val2) < 0.1
                        }
                comparison[category] = cat_comparison
        
        return comparison
    
    def _generate_comparison_verdict(self, exp1: Dict, exp2: Dict) -> str:
        """Generate verdict from comparison"""
        pred1 = exp1['prediction']['is_ai_generated']
        pred2 = exp2['prediction']['is_ai_generated']
        conf1 = exp1['prediction']['confidence']
        conf2 = exp2['prediction']['confidence']
        
        if pred1 == pred2:
            return f"Both texts are classified as {'AI-generated' if pred1 else 'human-written'}"
        else:
            return f"Text 1 is {'AI-generated' if pred1 else 'human-written'} (confidence: {conf1:.2%}), while Text 2 is {'AI-generated' if pred2 else 'human-written'} (confidence: {conf2:.2%})"


if __name__ == "__main__":
    # Example usage
    explainer = ExplainerService(
        model_path="models/bert/final_model",
        device="cpu"
    )
    
    sample_text = """
    Artificial intelligence has revolutionized numerous industries in recent years.
    Machine learning algorithms can now perform tasks with remarkable efficiency.
    However, it is important to consider the ethical implications of these technologies.
    """
    
    explanation = explainer.explain_prediction(sample_text)
    
    print("Prediction Explanation:")
    print("=" * 60)
    print(f"Prediction: {'AI-generated' if explanation['prediction']['is_ai_generated'] else 'Human-written'}")
    print(f"Confidence: {explanation['prediction']['confidence']:.2%}")
    print(f"\nInterpretation:\n{explanation['interpretation']}")
    print(f"\nKey Indicators:")
    for indicator in explanation['key_indicators']:
        print(f"  - {indicator['name']}: {indicator['suggests']}")
