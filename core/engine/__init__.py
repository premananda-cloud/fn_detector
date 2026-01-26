"""
Model engine subpackage.
Contains model loading and inference engines.
"""

from .call_bert import BERTDetector

__all__ = ['BERTDetector']