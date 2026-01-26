"""
Text processing subpackage.
Contains text processors and feature extractors.
"""

# Import from feature_extracter.py (not feature_extrater.py)
try:
    from .feature_extracter import FeatureExtractor
    from .text_processor import TextProcessor
except ImportError:
    # Fallback in case of import issues
    class FeatureExtractor:
        """Placeholder if actual class can't be imported"""
        pass
    
    class TextProcessor:
        """Placeholder if actual class can't be imported"""
        pass

__all__ = ['FeatureExtractor', 'TextProcessor']