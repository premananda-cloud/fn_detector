"""
Core Detector Module
Orchestrates the AI-generated text detection pipeline
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Union, Tuple
import numpy as np

# Add project root to path if needed
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.processor.text_processor import TextProcessor
from core.processor.feature_extracter import FeatureExtractor
from core.engine.call_bert import BERTDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AITextDetector:
    """
    Main detector class that coordinates text processing, feature extraction,
    and model inference for AI-generated text detection.
    """
    
    def __init__(
        self,
        model_path: str = "models/bert/final_model",
        device: str = "cpu",
        confidence_threshold: float = 0.5,
        batch_size: int = 8
    ):
        """
        Initialize the AI Text Detector
        
        Args:
            model_path: Path to the trained BERT model
            device: Device to run inference on ('cpu' or 'cuda')
            confidence_threshold: Threshold for classification (0-1)
            batch_size: Batch size for processing multiple texts
        """
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        
        # Initialize components
        logger.info("Initializing AI Text Detector...")
        self._initialize_components()
        logger.info("Detector initialization complete")
    
    def _initialize_components(self):
        """Initialize all detector components"""
        try:
            # Initialize text processor
            self.text_processor = TextProcessor()
            logger.info("✓ Text processor initialized")
            
            # Initialize feature extractor
            self.feature_extractor = FeatureExtractor()
            logger.info("✓ Feature extractor initialized")
            
            # Initialize BERT detector
            self.bert_detector = BERTDetector(
                model_path=self.model_path,
                device=self.device
            )
            logger.info("✓ BERT detector initialized")
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise
    
    def detect(
        self,
        text: Union[str, List[str]],
        return_probabilities: bool = True,
        return_features: bool = False
    ) -> Dict:
        """
        Detect if text is AI-generated
        
        Args:
            text: Single text string or list of texts
            return_probabilities: Whether to return probability scores
            return_features: Whether to return extracted features
            
        Returns:
            Dictionary containing detection results
        """
        # Handle single text vs batch
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        # Validate inputs
        if not texts or any(not t.strip() for t in texts):
            raise ValueError("Empty or invalid text provided")
        
        logger.info(f"Processing {len(texts)} text(s)...")
        
        # Process texts
        processed_texts = self._process_texts(texts)
        
        # Extract features (optional, for ensemble models)
        features = None
        if return_features:
            features = self._extract_features(processed_texts)
        
        # Run BERT detection
        predictions = self._run_detection(processed_texts)
        
        # Format results
        results = self._format_results(
            predictions,
            texts,
            return_probabilities,
            features if return_features else None
        )
        
        # Return single result if single input
        return results[0] if is_single else results
    
    def _process_texts(self, texts: List[str]) -> List[str]:
        """
        Preprocess texts for detection
        
        Args:
            texts: List of raw text strings
            
        Returns:
            List of processed text strings
        """
        processed = []
        for text in texts:
            try:
                processed_text = self.text_processor.process(text)
                processed.append(processed_text)
            except Exception as e:
                logger.warning(f"Error processing text: {str(e)}")
                processed.append(text)  # Use original if processing fails
        
        return processed
    
    def _extract_features(self, texts: List[str]) -> List[Dict]:
        """
        Extract linguistic features from texts
        
        Args:
            texts: List of processed texts
            
        Returns:
            List of feature dictionaries
        """
        features = []
        for text in texts:
            try:
                text_features = self.feature_extractor.extract(text)
                features.append(text_features)
            except Exception as e:
                logger.warning(f"Error extracting features: {str(e)}")
                features.append({})
        
        return features
    
    def _run_detection(self, texts: List[str]) -> List[Dict]:
        """
        Run BERT model detection
        
        Args:
            texts: List of processed texts
            
        Returns:
            List of prediction dictionaries
        """
        try:
            predictions = self.bert_detector.predict(
                texts,
                batch_size=self.batch_size
            )
            return predictions
        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            raise
    
    def _format_results(
        self,
        predictions: List[Dict],
        original_texts: List[str],
        return_probabilities: bool,
        features: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Format detection results
        
        Args:
            predictions: Model predictions
            original_texts: Original input texts
            return_probabilities: Whether to include probabilities
            features: Extracted features (optional)
            
        Returns:
            List of formatted result dictionaries
        """
        results = []
        
        for i, pred in enumerate(predictions):
            result = {
                'text': original_texts[i][:100] + '...' if len(original_texts[i]) > 100 else original_texts[i],
                'is_ai_generated': pred['probability'] >= self.confidence_threshold,
                'confidence': pred['probability']
            }
            
            if return_probabilities:
                result['probabilities'] = {
                    'ai_generated': pred['probability'],
                    'human_written': 1 - pred['probability']
                }
            
            # Add confidence level interpretation
            confidence_level = self._get_confidence_level(pred['probability'])
            result['confidence_level'] = confidence_level
            
            # Add features if requested
            if features and i < len(features):
                result['features'] = features[i]
            
            results.append(result)
        
        return results
    
    def _get_confidence_level(self, probability: float) -> str:
        """
        Interpret confidence level from probability
        
        Args:
            probability: Prediction probability
            
        Returns:
            Confidence level string
        """
        if probability >= 0.9 or probability <= 0.1:
            return "Very High"
        elif probability >= 0.75 or probability <= 0.25:
            return "High"
        elif probability >= 0.6 or probability <= 0.4:
            return "Moderate"
        else:
            return "Low"
    
    def detect_batch(
        self,
        texts: List[str],
        return_probabilities: bool = True
    ) -> List[Dict]:
        """
        Batch detection for multiple texts
        
        Args:
            texts: List of texts to analyze
            return_probabilities: Whether to return probabilities
            
        Returns:
            List of detection results
        """
        return self.detect(
            texts,
            return_probabilities=return_probabilities,
            return_features=False
        )
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_path': self.model_path,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'batch_size': self.batch_size,
            'model_info': self.bert_detector.get_model_info()
        }
    
    def update_threshold(self, new_threshold: float):
        """
        Update confidence threshold
        
        Args:
            new_threshold: New threshold value (0-1)
        """
        if not 0 <= new_threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        
        self.confidence_threshold = new_threshold
        logger.info(f"Confidence threshold updated to {new_threshold}")


if __name__ == "__main__":
    # Example usage
    detector = AITextDetector(
        model_path="models/bert/final_model",
        device="cpu",
        confidence_threshold=0.5
    )
    
    # Single text detection
    sample_text = """
    Artificial intelligence has revolutionized numerous industries in recent years.
    Machine learning algorithms can now perform tasks that were once thought to require
    human intelligence, such as image recognition and natural language processing.
    """
    
    result = detector.detect(sample_text)
    print("Detection Result:")
    print(f"Is AI Generated: {result['is_ai_generated']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Confidence Level: {result['confidence_level']}")