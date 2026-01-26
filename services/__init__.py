"""
Services Module
Provides inference, explanation, and orchestration services for AI text detection
"""

from inference import InferenceService
from explainer import ExplainerService
from orchestrator import DetectionOrchestrator

__all__ = [
    'InferenceService',
    'ExplainerService',
    'DetectionOrchestrator'
]

__version__ = '1.0.0'
