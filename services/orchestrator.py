"""
Orchestrator Service for Fake News Detection
Handles ensemble predictions and result interpretation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass


class EnsembleMethod(Enum):
    """Ensemble methods for combining model predictions"""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    UNANIMOUS = "unanimous"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    ADAPTIVE = "adaptive"


@dataclass
class EnsembleResult:
    """Result from ensemble prediction"""
    final_prediction: str
    confidence: float
    individual_predictions: Dict[str, Dict]
    consensus_score: float
    method_used: str
    explanation: str
    risk_level: str


class PredictionOrchestrator:
    """
    Orchestrates predictions from multiple models and creates ensemble predictions
    """
    
    def __init__(self, 
                 ensemble_method: EnsembleMethod = EnsembleMethod.CONFIDENCE_WEIGHTED,
                 model_weights: Optional[Dict[str, float]] = None):
        """
        Initialize orchestrator
        
        Args:
            ensemble_method: Method to use for ensemble predictions
            model_weights: Optional weights for each model (for weighted methods)
        """
        self.ensemble_method = ensemble_method
        
        # Default model weights (can be adjusted based on validation performance)
        self.model_weights = model_weights or {
            'BERT': 0.35,
            'RoBERTa': 0.40,
            'TF-IDF': 0.25
        }
        
        # Normalize weights
        total_weight = sum(self.model_weights.values())
        self.model_weights = {k: v/total_weight for k, v in self.model_weights.items()}
    
    def ensemble_predict(self, predictions: Dict[str, Dict]) -> EnsembleResult:
        """
        Create ensemble prediction from multiple model predictions
        
        Args:
            predictions: Dictionary of predictions from each model
            
        Returns:
            EnsembleResult object with final prediction and metadata
        """
        if not predictions:
            raise ValueError("No predictions provided")
        
        # Filter out error predictions
        valid_predictions = {
            k: v for k, v in predictions.items() 
            if v.get('prediction') not in ['ERROR', None]
        }
        
        if not valid_predictions:
            return EnsembleResult(
                final_prediction="ERROR",
                confidence=0.0,
                individual_predictions=predictions,
                consensus_score=0.0,
                method_used=self.ensemble_method.value,
                explanation="All models failed to make predictions",
                risk_level="UNKNOWN"
            )
        
        # Choose ensemble method
        if self.ensemble_method == EnsembleMethod.MAJORITY_VOTE:
            result = self._majority_vote(valid_predictions)
        elif self.ensemble_method == EnsembleMethod.WEIGHTED_AVERAGE:
            result = self._weighted_average(valid_predictions)
        elif self.ensemble_method == EnsembleMethod.UNANIMOUS:
            result = self._unanimous(valid_predictions)
        elif self.ensemble_method == EnsembleMethod.CONFIDENCE_WEIGHTED:
            result = self._confidence_weighted(valid_predictions)
        elif self.ensemble_method == EnsembleMethod.ADAPTIVE:
            result = self._adaptive(valid_predictions)
        else:
            result = self._confidence_weighted(valid_predictions)
        
        # Add original predictions
        result.individual_predictions = predictions
        
        return result
    
    def _majority_vote(self, predictions: Dict[str, Dict]) -> EnsembleResult:
        """Simple majority voting"""
        votes = [pred['prediction'] for pred in predictions.values()]
        fake_votes = votes.count('FAKE')
        real_votes = votes.count('REAL')
        
        final_prediction = 'FAKE' if fake_votes > real_votes else 'REAL'
        
        # Calculate consensus score
        total_votes = len(votes)
        winning_votes = max(fake_votes, real_votes)
        consensus_score = winning_votes / total_votes
        
        # Average confidence
        avg_confidence = np.mean([pred['confidence'] for pred in predictions.values()])
        
        explanation = f"{winning_votes}/{total_votes} models predicted {final_prediction}"
        
        return EnsembleResult(
            final_prediction=final_prediction,
            confidence=avg_confidence,
            individual_predictions={},
            consensus_score=consensus_score,
            method_used="majority_vote",
            explanation=explanation,
            risk_level=self._calculate_risk_level(final_prediction, consensus_score)
        )
    
    def _weighted_average(self, predictions: Dict[str, Dict]) -> EnsembleResult:
        """Weighted average of model predictions"""
        weighted_fake_prob = 0.0
        weighted_real_prob = 0.0
        total_weight = 0.0
        
        for model_name, pred in predictions.items():
            weight = self.model_weights.get(model_name, 1.0)
            probs = pred.get('probabilities', {})
            
            weighted_fake_prob += probs.get('fake', 0.0) * weight
            weighted_real_prob += probs.get('real', 0.0) * weight
            total_weight += weight
        
        # Normalize
        weighted_fake_prob /= total_weight
        weighted_real_prob /= total_weight
        
        final_prediction = 'FAKE' if weighted_fake_prob > weighted_real_prob else 'REAL'
        confidence = max(weighted_fake_prob, weighted_real_prob)
        
        # Consensus score based on probability difference
        consensus_score = abs(weighted_fake_prob - weighted_real_prob)
        
        explanation = f"Weighted probabilities: FAKE={weighted_fake_prob:.3f}, REAL={weighted_real_prob:.3f}"
        
        return EnsembleResult(
            final_prediction=final_prediction,
            confidence=confidence,
            individual_predictions={},
            consensus_score=consensus_score,
            method_used="weighted_average",
            explanation=explanation,
            risk_level=self._calculate_risk_level(final_prediction, confidence)
        )
    
    def _unanimous(self, predictions: Dict[str, Dict]) -> EnsembleResult:
        """Require unanimous agreement from all models"""
        all_predictions = [pred['prediction'] for pred in predictions.values()]
        
        if len(set(all_predictions)) == 1:
            # All models agree
            final_prediction = all_predictions[0]
            consensus_score = 1.0
            avg_confidence = np.mean([pred['confidence'] for pred in predictions.values()])
            explanation = f"All {len(predictions)} models unanimously predict {final_prediction}"
        else:
            # No unanimous agreement
            final_prediction = "UNCERTAIN"
            consensus_score = 0.0
            avg_confidence = 0.5
            fake_count = all_predictions.count('FAKE')
            real_count = all_predictions.count('REAL')
            explanation = f"No consensus: {fake_count} FAKE, {real_count} REAL votes"
        
        return EnsembleResult(
            final_prediction=final_prediction,
            confidence=avg_confidence,
            individual_predictions={},
            consensus_score=consensus_score,
            method_used="unanimous",
            explanation=explanation,
            risk_level=self._calculate_risk_level(final_prediction, consensus_score)
        )
    
    def _confidence_weighted(self, predictions: Dict[str, Dict]) -> EnsembleResult:
        """Weight predictions by their confidence scores"""
        weighted_fake_prob = 0.0
        weighted_real_prob = 0.0
        total_confidence = 0.0
        
        for model_name, pred in predictions.items():
            confidence = pred['confidence']
            probs = pred.get('probabilities', {})
            
            weighted_fake_prob += probs.get('fake', 0.0) * confidence
            weighted_real_prob += probs.get('real', 0.0) * confidence
            total_confidence += confidence
        
        if total_confidence > 0:
            weighted_fake_prob /= total_confidence
            weighted_real_prob /= total_confidence
        
        final_prediction = 'FAKE' if weighted_fake_prob > weighted_real_prob else 'REAL'
        confidence = max(weighted_fake_prob, weighted_real_prob)
        
        # Consensus score
        consensus_score = abs(weighted_fake_prob - weighted_real_prob)
        
        explanation = f"Confidence-weighted probabilities: FAKE={weighted_fake_prob:.3f}, REAL={weighted_real_prob:.3f}"
        
        return EnsembleResult(
            final_prediction=final_prediction,
            confidence=confidence,
            individual_predictions={},
            consensus_score=consensus_score,
            method_used="confidence_weighted",
            explanation=explanation,
            risk_level=self._calculate_risk_level(final_prediction, confidence)
        )
    
    def _adaptive(self, predictions: Dict[str, Dict]) -> EnsembleResult:
        """Adaptive method: choose strategy based on agreement level"""
        all_predictions = [pred['prediction'] for pred in predictions.values()]
        unique_predictions = set(all_predictions)
        
        if len(unique_predictions) == 1:
            # Perfect agreement - use simple average
            return self._majority_vote(predictions)
        else:
            # Disagreement - use confidence weighting
            return self._confidence_weighted(predictions)
    
    def _calculate_risk_level(self, prediction: str, score: float) -> str:
        """
        Calculate risk level based on prediction and confidence/consensus
        
        Args:
            prediction: Final prediction (FAKE/REAL)
            score: Confidence or consensus score
            
        Returns:
            Risk level string
        """
        if prediction == "ERROR" or prediction == "UNCERTAIN":
            return "UNKNOWN"
        
        if prediction == "FAKE":
            if score >= 0.8:
                return "HIGH"  # High confidence fake
            elif score >= 0.6:
                return "MEDIUM"
            else:
                return "LOW"  # Low confidence, needs review
        else:  # REAL
            if score >= 0.8:
                return "LOW"  # High confidence real
            elif score >= 0.6:
                return "MEDIUM"
            else:
                return "HIGH"  # Low confidence on real, might be fake
        
    def interpret_results(self, result: EnsembleResult, verbose: bool = True) -> str:
        """
        Generate human-readable interpretation of results
        
        Args:
            result: EnsembleResult object
            verbose: Include detailed explanation
            
        Returns:
            Formatted interpretation string
        """
        lines = []
        lines.append("="*70)
        lines.append("FAKE NEWS DETECTION RESULTS")
        lines.append("="*70)
        
        # Main result
        lines.append(f"\n🎯 FINAL PREDICTION: {result.final_prediction}")
        lines.append(f"📊 Confidence: {result.confidence:.2%}")
        lines.append(f"🤝 Consensus Score: {result.consensus_score:.2%}")
        lines.append(f"⚠️  Risk Level: {result.risk_level}")
        lines.append(f"🔧 Method Used: {result.method_used}")
        
        # Individual model predictions
        if verbose and result.individual_predictions:
            lines.append("\n" + "-"*70)
            lines.append("INDIVIDUAL MODEL PREDICTIONS")
            lines.append("-"*70)
            
            for model_name, pred in result.individual_predictions.items():
                lines.append(f"\n{model_name}:")
                lines.append(f"  Prediction: {pred.get('prediction', 'N/A')}")
                lines.append(f"  Confidence: {pred.get('confidence', 0):.2%}")
                
                if 'probabilities' in pred:
                    probs = pred['probabilities']
                    lines.append(f"  Probabilities:")
                    lines.append(f"    - REAL: {probs.get('real', 0):.2%}")
                    lines.append(f"    - FAKE: {probs.get('fake', 0):.2%}")
        
        # Explanation
        lines.append("\n" + "-"*70)
        lines.append("EXPLANATION")
        lines.append("-"*70)
        lines.append(result.explanation)
        
        # Recommendation
        lines.append("\n" + "-"*70)
        lines.append("RECOMMENDATION")
        lines.append("-"*70)
        lines.append(self._generate_recommendation(result))
        
        lines.append("="*70)
        
        return "\n".join(lines)
    
    def _generate_recommendation(self, result: EnsembleResult) -> str:
        """Generate recommendation based on results"""
        if result.final_prediction == "FAKE":
            if result.risk_level == "HIGH":
                return "⛔ This content is likely FAKE NEWS. High confidence across models. Do not share."
            elif result.risk_level == "MEDIUM":
                return "⚠️  This content is possibly FAKE. Exercise caution and verify from trusted sources."
            else:
                return "❓ Low confidence prediction. Further verification recommended before sharing."
        
        elif result.final_prediction == "REAL":
            if result.risk_level == "LOW":
                return "✅ This content appears to be REAL. High confidence across models."
            elif result.risk_level == "MEDIUM":
                return "✓ This content is likely REAL, but verify important claims."
            else:
                return "❓ Low confidence prediction. Cross-check with other sources before trusting."
        
        else:
            return "❓ Unable to determine authenticity. Models disagree. Manual review needed."
    
    def batch_ensemble(self, batch_predictions: List[Dict[str, Dict]]) -> List[EnsembleResult]:
        """
        Process ensemble predictions for a batch of inputs
        
        Args:
            batch_predictions: List of prediction dictionaries
            
        Returns:
            List of EnsembleResult objects
        """
        results = []
        for predictions in batch_predictions:
            results.append(self.ensemble_predict(predictions))
        return results
    
    def get_summary_statistics(self, results: List[EnsembleResult]) -> Dict:
        """
        Calculate summary statistics for a batch of results
        
        Args:
            results: List of EnsembleResult objects
            
        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {}
        
        fake_count = sum(1 for r in results if r.final_prediction == "FAKE")
        real_count = sum(1 for r in results if r.final_prediction == "REAL")
        uncertain_count = sum(1 for r in results if r.final_prediction == "UNCERTAIN")
        
        avg_confidence = np.mean([r.confidence for r in results])
        avg_consensus = np.mean([r.consensus_score for r in results])
        
        risk_distribution = {
            "HIGH": sum(1 for r in results if r.risk_level == "HIGH"),
            "MEDIUM": sum(1 for r in results if r.risk_level == "MEDIUM"),
            "LOW": sum(1 for r in results if r.risk_level == "LOW"),
            "UNKNOWN": sum(1 for r in results if r.risk_level == "UNKNOWN")
        }
        
        return {
            "total_predictions": len(results),
            "fake_count": fake_count,
            "real_count": real_count,
            "uncertain_count": uncertain_count,
            "fake_percentage": fake_count / len(results) * 100,
            "real_percentage": real_count / len(results) * 100,
            "average_confidence": avg_confidence,
            "average_consensus": avg_consensus,
            "risk_distribution": risk_distribution
        }


if __name__ == "__main__":
    # Test orchestrator
    print("Testing Prediction Orchestrator\n")
    
    # Mock predictions for testing
    test_predictions = {
        'BERT': {
            'prediction': 'FAKE',
            'confidence': 0.89,
            'probabilities': {'real': 0.11, 'fake': 0.89}
        },
        'RoBERTa': {
            'prediction': 'FAKE',
            'confidence': 0.92,
            'probabilities': {'real': 0.08, 'fake': 0.92}
        },
        'TF-IDF': {
            'prediction': 'REAL',
            'confidence': 0.65,
            'probabilities': {'real': 0.65, 'fake': 0.35}
        }
    }
    
    # Test different ensemble methods
    for method in EnsembleMethod:
        print(f"\n{'='*70}")
        print(f"Testing {method.value}")
        print(f"{'='*70}")
        
        orchestrator = PredictionOrchestrator(ensemble_method=method)
        result = orchestrator.ensemble_predict(test_predictions)
        
        print(orchestrator.interpret_results(result, verbose=False))
