"""
Orchestrator Service
Coordinates all services and manages the complete detection workflow
"""

import logging
from typing import Dict, List, Union, Optional
from datetime import datetime
import json

from inference import InferenceService
from explainer import ExplainerService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DetectionOrchestrator:
    """
    Main orchestrator for AI text detection
    Coordinates inference, explanation, and reporting services
    """
    
    def __init__(
        self,
        model_path: str = "models/bert/final_model",
        device: str = "cpu",
        confidence_threshold: float = 0.5,
        batch_size: int = 8,
        enable_cache: bool = True
    ):
        """
        Initialize detection orchestrator
        
        Args:
            model_path: Path to the trained model
            device: Device for inference
            confidence_threshold: Classification threshold
            batch_size: Batch size for processing
            enable_cache: Enable result caching
        """
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        
        logger.info("Initializing Detection Orchestrator...")
        
        # Initialize services
        self.inference_service = InferenceService(
            model_path=model_path,
            device=device,
            confidence_threshold=confidence_threshold,
            batch_size=batch_size,
            enable_cache=enable_cache
        )
        
        self.explainer_service = ExplainerService(
            model_path=model_path,
            device=device
        )
        
        # Session tracking
        self.session_id = self._generate_session_id()
        self.detection_history = []
        
        logger.info("✓ Detection Orchestrator initialized")
    
    def detect(
        self,
        text: Union[str, List[str]],
        explain: bool = False,
        detailed: bool = False
    ) -> Union[Dict, List[Dict]]:
        """
        Main detection method
        
        Args:
            text: Single text or list of texts
            explain: Include explanation
            detailed: Include detailed analysis
            
        Returns:
            Detection result(s)
        """
        is_single = isinstance(text, str)
        
        # Run inference
        if is_single:
            result = self._detect_single(text, explain, detailed)
            self._add_to_history(text, result)
            return result
        else:
            results = self._detect_batch(text, explain, detailed)
            for t, r in zip(text, results):
                self._add_to_history(t, r)
            return results
    
    def _detect_single(
        self,
        text: str,
        explain: bool = False,
        detailed: bool = False
    ) -> Dict:
        """
        Detect single text
        
        Args:
            text: Input text
            explain: Include explanation
            detailed: Include detailed analysis
            
        Returns:
            Detection result
        """
        logger.info("Processing single text...")
        
        # Get prediction
        result = self.inference_service.predict_single(
            text,
            return_probabilities=True,
            return_features=detailed
        )
        
        # Add explanation if requested
        if explain:
            explanation = self.explainer_service.explain_prediction(
                text,
                include_features=detailed,
                include_attention=detailed,
                include_suggestions=True
            )
            result['explanation'] = explanation
        
        # Add session info
        result['session_id'] = self.session_id
        
        return result
    
    def _detect_batch(
        self,
        texts: List[str],
        explain: bool = False,
        detailed: bool = False
    ) -> List[Dict]:
        """
        Detect multiple texts
        
        Args:
            texts: List of texts
            explain: Include explanations
            detailed: Include detailed analysis
            
        Returns:
            List of detection results
        """
        logger.info(f"Processing batch of {len(texts)} texts...")
        
        # Get predictions
        results = self.inference_service.predict_batch(
            texts,
            return_probabilities=True,
            return_features=detailed
        )
        
        # Add explanations if requested
        if explain:
            for i, (text, result) in enumerate(zip(texts, results)):
                logger.debug(f"Generating explanation for text {i+1}/{len(texts)}")
                explanation = self.explainer_service.explain_prediction(
                    text,
                    include_features=detailed,
                    include_attention=False,  # Skip attention for batch
                    include_suggestions=True
                )
                result['explanation'] = explanation
        
        # Add session info
        for result in results:
            result['session_id'] = self.session_id
        
        return results
    
    def analyze_text(self, text: str) -> Dict:
        """
        Comprehensive text analysis with full explanation
        
        Args:
            text: Input text
            
        Returns:
            Comprehensive analysis result
        """
        logger.info("Performing comprehensive text analysis...")
        
        # Get detailed detection with explanation
        result = self.detect(text, explain=True, detailed=True)
        
        # Add comprehensive analysis
        result['analysis_type'] = 'comprehensive'
        result['analysis_timestamp'] = datetime.now().isoformat()
        
        return result
    
    def compare_texts(self, text1: str, text2: str) -> Dict:
        """
        Compare two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Comparison result
        """
        logger.info("Comparing two texts...")
        
        # Get individual detections
        result1 = self.detect(text1, explain=True, detailed=True)
        result2 = self.detect(text2, explain=True, detailed=True)
        
        # Get comparison from explainer
        comparison = self.explainer_service.compare_texts(text1, text2)
        
        return {
            'text1_result': result1,
            'text2_result': result2,
            'comparison': comparison,
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat()
        }
    
    def batch_analysis(
        self,
        texts: List[str],
        generate_report: bool = True
    ) -> Dict:
        """
        Batch analysis with optional report generation
        
        Args:
            texts: List of texts to analyze
            generate_report: Generate summary report
            
        Returns:
            Batch analysis result
        """
        logger.info(f"Performing batch analysis on {len(texts)} texts...")
        
        # Run batch detection
        results = self.detect(texts, explain=False, detailed=False)
        
        batch_result = {
            'total_texts': len(texts),
            'results': results,
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat()
        }
        
        # Generate report if requested
        if generate_report:
            report = self._generate_batch_report(results)
            batch_result['report'] = report
        
        return batch_result
    
    def _generate_batch_report(self, results: List[Dict]) -> Dict:
        """
        Generate summary report for batch analysis
        
        Args:
            results: List of detection results
            
        Returns:
            Report dictionary
        """
        total = len(results)
        ai_generated = sum(1 for r in results if r['is_ai_generated'])
        human_written = total - ai_generated
        
        # Confidence statistics
        confidences = [r['confidence'] for r in results]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Confidence level distribution
        confidence_levels = {}
        for r in results:
            level = r.get('confidence_level', 'Unknown')
            confidence_levels[level] = confidence_levels.get(level, 0) + 1
        
        return {
            'summary': {
                'total_texts': total,
                'ai_generated_count': ai_generated,
                'human_written_count': human_written,
                'ai_generated_percentage': (ai_generated / total * 100) if total > 0 else 0,
                'human_written_percentage': (human_written / total * 100) if total > 0 else 0
            },
            'confidence_statistics': {
                'average_confidence': avg_confidence,
                'min_confidence': min(confidences) if confidences else 0,
                'max_confidence': max(confidences) if confidences else 0,
                'confidence_level_distribution': confidence_levels
            },
            'recommendations': self._generate_batch_recommendations(results)
        }
    
    def _generate_batch_recommendations(self, results: List[Dict]) -> List[str]:
        """Generate recommendations based on batch results"""
        recommendations = []
        
        # Check for borderline cases
        borderline = sum(
            1 for r in results 
            if abs(r['confidence'] - self.confidence_threshold) < 0.1
        )
        
        if borderline > 0:
            recommendations.append(
                f"{borderline} text(s) have borderline confidence - manual review recommended"
            )
        
        # Check for very high confidence AI texts
        high_conf_ai = sum(
            1 for r in results 
            if r['is_ai_generated'] and r['confidence'] > 0.9
        )
        
        if high_conf_ai > len(results) * 0.5:
            recommendations.append(
                "Majority of texts show strong AI-generation signals"
            )
        
        # Check for consistency
        ai_count = sum(1 for r in results if r['is_ai_generated'])
        if ai_count == len(results) or ai_count == 0:
            recommendations.append(
                "All texts classified the same way - consider reviewing detection parameters"
            )
        
        return recommendations
    
    def get_session_statistics(self) -> Dict:
        """
        Get statistics for current session
        
        Returns:
            Session statistics
        """
        inference_stats = self.inference_service.get_statistics()
        
        # Aggregate history
        total_detections = len(self.detection_history)
        ai_count = sum(
            1 for entry in self.detection_history 
            if entry['result'].get('is_ai_generated', False)
        )
        
        return {
            'session_id': self.session_id,
            'total_detections': total_detections,
            'ai_generated_count': ai_count,
            'human_written_count': total_detections - ai_count,
            'inference_statistics': inference_stats,
            'detection_history_size': len(self.detection_history)
        }
    
    def get_detection_history(
        self,
        limit: Optional[int] = None,
        filter_ai_only: bool = False,
        filter_human_only: bool = False
    ) -> List[Dict]:
        """
        Get detection history
        
        Args:
            limit: Maximum number of results
            filter_ai_only: Only AI-generated texts
            filter_human_only: Only human-written texts
            
        Returns:
            List of historical detections
        """
        history = self.detection_history.copy()
        
        # Apply filters
        if filter_ai_only:
            history = [
                entry for entry in history 
                if entry['result'].get('is_ai_generated', False)
            ]
        elif filter_human_only:
            history = [
                entry for entry in history 
                if not entry['result'].get('is_ai_generated', True)
            ]
        
        # Apply limit
        if limit:
            history = history[-limit:]
        
        return history
    
    def clear_history(self):
        """Clear detection history"""
        self.detection_history.clear()
        logger.info("Detection history cleared")
    
    def clear_cache(self):
        """Clear inference cache"""
        self.inference_service.clear_cache()
        logger.info("Inference cache cleared")
    
    def update_threshold(self, new_threshold: float):
        """
        Update confidence threshold
        
        Args:
            new_threshold: New threshold value
        """
        self.inference_service.update_threshold(new_threshold)
        self.confidence_threshold = new_threshold
        logger.info(f"Confidence threshold updated to {new_threshold}")
    
    def export_results(
        self,
        results: Union[Dict, List[Dict]],
        format: str = 'json'
    ) -> str:
        """
        Export results to specified format
        
        Args:
            results: Result(s) to export
            format: Export format ('json', 'csv', 'txt')
            
        Returns:
            Formatted string
        """
        if format == 'json':
            return json.dumps(results, indent=2)
        elif format == 'txt':
            return self._format_as_text(results)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _format_as_text(self, results: Union[Dict, List[Dict]]) -> str:
        """Format results as readable text"""
        if isinstance(results, dict):
            results = [results]
        
        output = []
        output.append("AI Text Detection Results")
        output.append("=" * 60)
        output.append("")
        
        for i, result in enumerate(results, 1):
            output.append(f"Text {i}:")
            output.append(f"  Classification: {'AI-generated' if result['is_ai_generated'] else 'Human-written'}")
            output.append(f"  Confidence: {result['confidence']:.2%}")
            output.append(f"  Confidence Level: {result.get('confidence_level', 'Unknown')}")
            
            if 'explanation' in result:
                output.append(f"  Interpretation: {result['explanation']['interpretation']}")
            
            output.append("")
        
        return "\n".join(output)
    
    def _add_to_history(self, text: str, result: Dict):
        """Add detection to history"""
        self.detection_history.append({
            'text': text[:100] + '...' if len(text) > 100 else text,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep history size manageable
        if len(self.detection_history) > 1000:
            self.detection_history = self.detection_history[-1000:]
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def warm_up(self):
        """Warm up the detection system"""
        logger.info("Warming up detection system...")
        self.inference_service.warm_up()
        logger.info("✓ System ready")


if __name__ == "__main__":
    # Example usage
    orchestrator = DetectionOrchestrator(
        model_path="models/bert/final_model",
        device="cpu",
        confidence_threshold=0.5
    )
    
    # Warm up
    orchestrator.warm_up()
    
    # Single detection with explanation
    text = """
    Artificial intelligence has fundamentally transformed the landscape of modern technology.
    Machine learning algorithms demonstrate remarkable capabilities in pattern recognition.
    However, ethical considerations remain paramount in the development of these systems.
    """
    
    result = orchestrator.analyze_text(text)
    
    print("Comprehensive Analysis:")
    print("=" * 60)
    print(f"Classification: {'AI-generated' if result['is_ai_generated'] else 'Human-written'}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nInterpretation:")
    print(result['explanation']['interpretation'])
    
    # Batch analysis
    texts = [
        "This is a simple test sentence.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models require extensive training data."
    ]
    
    batch_result = orchestrator.batch_analysis(texts, generate_report=True)
    
    print("\n" + "=" * 60)
    print("Batch Analysis Report:")
    print(f"Total Texts: {batch_result['report']['summary']['total_texts']}")
    print(f"AI-Generated: {batch_result['report']['summary']['ai_generated_count']}")
    print(f"Human-Written: {batch_result['report']['summary']['human_written_count']}")
    
    # Get statistics
    stats = orchestrator.get_session_statistics()
    print("\n" + "=" * 60)
    print("Session Statistics:")
    print(f"Total Detections: {stats['total_detections']}")
    print(f"Cache Hit Rate: {stats['inference_statistics']['cache_hit_rate']:.2%}")
