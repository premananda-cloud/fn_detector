"""
Core detection engine package.
Contains all components for AI text detection: detectors, engines, and processors.
"""

from .detector import AITextDetector
from .engine.call_bert import BERTDetector
from .processor.feature_extracter import FeatureExtractor
from .processor.text_processor import TextProcessor

__version__ = "1.0.0"
__all__ = [
    'AITextDetector',
    'BERTDetector', 
    'FeatureExtractor',
    'TextProcessor'
]