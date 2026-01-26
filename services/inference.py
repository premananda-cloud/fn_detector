"""
Inference Service
Handles model inference, batch predictions, and result formatting
"""

import os
import sys
import logging
from typing import Dict, List, Union, Optional, Tuple
import time
from datetime import datetime
import numpy as np

# Add project root to path if needed
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.detector import AITextDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceService:
    """
    Service for running inference on texts to detect AI generation
    Handles single and batch predictions with caching and optimization
    """
    
    def __init__(
        self,
        model_path: str = "models/bert/final_model",
        device: str = "cpu",
        confidence_threshold: float = 0.5,
        batch_size: int = 8,
        enable_cache: bool = True,
        max_cache_size: int = 1000
    ):
        """
        Initialize inference service
        
        Args:
            model_path: Path to the trained model
            device: Device for inference ('cpu' or 'cuda')
            confidence_threshold: Classification threshold
            batch_size: Batch size for processing
            enable_cache: Enable result caching
            max_cache_size: Maximum cache size
        """
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.enable_cache = enable_cache
        self.max_cache_size = max_cache_size
        
        # Initialize detector
        logger.info("Initializing inference service...")
        self.detector = AITextDetector(
            model_path=model_path,
            device=device,
            confidence_threshold=confidence_threshold,
            batch_size=batch_size
        )
        
        # Cache for storing results
        self.cache = {} if enable_cache else None
        
        # Statistics
        self.stats = {
            'total_inferences': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_texts_processed': 0,
            'avg_inference_time': 0,
            'inference_times': []
        }
        
        logger.info("✓ Inference service initialized")
    
    def predict_single(
        self,
        text: str,
        return_probabilities: bool = True,
        return_features: bool = False,
        use_cache: bool = True
    ) -> Dict:
        """
        Predict if a single text is AI-generated
        
        Args:
            text: Input text
            return_probabilities: Include probability scores
            return_features: Include extracted features
            use_cache: Use cached results if available
            
        Returns:
            Prediction result dictionary
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Check cache
        if use_cache and self.enable_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in self.cache:
                self.stats['cache_hits'] += 1
                logger.debug("Cache hit for text")
                return self.cache[cache_key]
            self.stats['cache_misses'] += 1
        
        # Run inference
        start_time = time.time()
        
        result = self.detector.detect(
            text=text,
            return_probabilities=return_probabilities,
            return_features=return_features
        )
        
        inference_time = time.time() - start_time
        
        # Add metadata
        result['inference_time'] = inference_time
        result['timestamp'] = datetime.now().isoformat()
        result['model_path'] = self.model_path
        
        # Update statistics
        self._update_stats(inference_time)
        
        # Cache result
        if use_cache and self.enable_cache:
            self._cache_result(text, result)
        
        return result
    
    def predict_batch(
        self,
        texts: List[str],
        return_probabilities: bool = True,
        return_features: bool = False,
        show_progress: bool = False
    ) -> List[Dict]:
        """
        Predict for multiple texts
        
        Args:
            texts: List of input texts
            return_probabilities: Include probability scores
            return_features: Include extracted features
            show_progress: Show progress during processing
            
        Returns:
            List of prediction results
        """
        if not texts:
            raise ValueError("Text list cannot be empty")
        
        logger.info(f"Processing batch of {len(texts)} texts")
        
        start_time = time.time()
        
        # Check cache for each text
        results = []
        texts_to_process = []
        text_indices = []
        
        for i, text in enumerate(texts):
            if self.enable_cache:
                cache_key = self._get_cache_key(text)
                if cache_key in self.cache:
                    results.append((i, self.cache[cache_key]))
                    self.stats['cache_hits'] += 1
                    continue
            
            texts_to_process.append(text)
            text_indices.append(i)
            self.stats['cache_misses'] += 1
        
        # Process uncached texts
        if texts_to_process:
            new_results = self.detector.detect_batch(
                texts_to_process,
                return_probabilities=return_probabilities
            )
            
            # Add metadata and cache
            for i, (text_idx, result) in enumerate(zip(text_indices, new_results)):
                result['inference_time'] = (time.time() - start_time) / len(texts_to_process)
                result['timestamp'] = datetime.now().isoformat()
                result['model_path'] = self.model_path
                
                results.append((text_idx, result))
                
                # Cache result
                if self.enable_cache:
                    self._cache_result(texts_to_process[i], result)
        
        # Sort by original index
        results.sort(key=lambda x: x[0])
        final_results = [r[1] for r in results]
        
        # Update statistics
        total_time = time.time() - start_time
        self.stats['total_texts_processed'] += len(texts)
        
        logger.info(f"Batch processing completed in {total_time:.2f}s")
        
        return final_results
    
    def predict_with_confidence_analysis(
        self,
        text: str
    ) -> Dict:
        """
        Predict with detailed confidence analysis
        
        Args:
            text: Input text
            
        Returns:
            Detailed prediction result with confidence breakdown
        """
        result = self.predict_single(
            text,
            return_probabilities=True,
            return_features=True
        )
        
        # Add confidence analysis
        confidence_analysis = self._analyze_confidence(result)
        result['confidence_analysis'] = confidence_analysis
        
        return result
    
    def _analyze_confidence(self, result: Dict) -> Dict:
        """
        Analyze prediction confidence
        
        Args:
            result: Prediction result
            
        Returns:
            Confidence analysis
        """
        probability = result['confidence']
        
        # Determine confidence level
        if probability >= 0.9 or probability <= 0.1:
            level = "Very High"
            certainty = "Highly Certain"
        elif probability >= 0.75 or probability <= 0.25:
            level = "High"
            certainty = "Certain"
        elif probability >= 0.6 or probability <= 0.4:
            level = "Moderate"
            certainty = "Moderately Certain"
        else:
            level = "Low"
            certainty = "Uncertain"
        
        # Distance from threshold
        distance_from_threshold = abs(probability - self.confidence_threshold)
        
        return {
            'level': level,
            'certainty': certainty,
            'distance_from_threshold': distance_from_threshold,
            'requires_review': distance_from_threshold < 0.1,
            'recommendation': self._get_recommendation(probability, distance_from_threshold)
        }
    
    def _get_recommendation(self, probability: float, distance: float) -> str:
        """Get recommendation based on prediction"""
        if distance < 0.1:
            return "Borderline case - manual review recommended"
        elif probability >= 0.8:
            return "High confidence AI-generated text"
        elif probability <= 0.2:
            return "High confidence human-written text"
        elif probability >= 0.6:
            return "Likely AI-generated"
        elif probability <= 0.4:
            return "Likely human-written"
        else:
            return "Uncertain - additional analysis recommended"
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        # Use hash of text as key
        return str(hash(text))
    
    def _cache_result(self, text: str, result: Dict):
        """Cache inference result"""
        if not self.enable_cache:
            return
        
        cache_key = self._get_cache_key(text)
        
        # Manage cache size
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = result
    
    def _update_stats(self, inference_time: float):
        """Update inference statistics"""
        self.stats['total_inferences'] += 1
        self.stats['inference_times'].append(inference_time)
        
        # Keep only last 1000 times for average
        if len(self.stats['inference_times']) > 1000:
            self.stats['inference_times'] = self.stats['inference_times'][-1000:]
        
        self.stats['avg_inference_time'] = np.mean(self.stats['inference_times'])
    
    def get_statistics(self) -> Dict:
        """
        Get inference statistics
        
        Returns:
            Statistics dictionary
        """
        cache_hit_rate = (
            self.stats['cache_hits'] / 
            (self.stats['cache_hits'] + self.stats['cache_misses'])
            if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0
            else 0
        )
        
        return {
            'total_inferences': self.stats['total_inferences'],
            'total_texts_processed': self.stats['total_texts_processed'],
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self.cache) if self.cache else 0,
            'avg_inference_time': self.stats['avg_inference_time'],
            'model_info': self.detector.get_model_info()
        }
    
    def clear_cache(self):
        """Clear the result cache"""
        if self.cache:
            self.cache.clear()
            logger.info("Cache cleared")
    
    def update_threshold(self, new_threshold: float):
        """
        Update confidence threshold
        
        Args:
            new_threshold: New threshold value
        """
        self.detector.update_threshold(new_threshold)
        self.confidence_threshold = new_threshold
        # Clear cache as results may change
        self.clear_cache()
        logger.info(f"Threshold updated to {new_threshold}, cache cleared")
    
    def warm_up(self, num_samples: int = 5):
        """
        Warm up the model with sample predictions
        
        Args:
            num_samples: Number of warm-up samples
        """
        logger.info("Warming up model...")
        sample_text = "This is a sample text for warming up the model."
        
        for _ in range(num_samples):
            self.predict_single(sample_text, use_cache=False)
        
        logger.info("✓ Model warm-up complete")
    
    def benchmark(self, texts: List[str]) -> Dict:
        """
        Benchmark inference performance
        
        Args:
            texts: Test texts
            
        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking with {len(texts)} texts...")
        
        # Single inference benchmark
        start_time = time.time()
        for text in texts:
            self.predict_single(text, use_cache=False)
        single_time = time.time() - start_time
        
        # Batch inference benchmark
        start_time = time.time()
        self.predict_batch(texts)
        batch_time = time.time() - start_time
        
        return {
            'num_texts': len(texts),
            'single_inference_total_time': single_time,
            'single_inference_avg_time': single_time / len(texts),
            'batch_inference_total_time': batch_time,
            'batch_inference_avg_time': batch_time / len(texts),
            'speedup_factor': single_time / batch_time if batch_time > 0 else 0
        }


if __name__ == "__main__":
    # Example usage
    service = InferenceService(
        model_path="models/bert/final_model",
        device="cpu",
        confidence_threshold=0.5
    )
    
    # Warm up
    service.warm_up()
    
    # Single prediction
    text = "Artificial intelligence has revolutionized many industries."
    result = service.predict_single(text)
    
    print("Prediction Result:")
    print(f"Is AI Generated: {result['is_ai_generated']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Inference Time: {result['inference_time']:.4f}s")