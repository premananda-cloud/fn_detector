"""
Services package for Fake News Detection System
"""

from .inference import (
    MultiModelInference,
    BERTInference,
    RoBERTaInference,
    TFIDFInference
)

from .orchestrator import (
    PredictionOrchestrator,
    EnsembleMethod,
    EnsembleResult
)

from .explainer import (
    PredictionExplainer
)

__all__ = [
    'MultiModelInference',
    'BERTInference',
    'RoBERTaInference',
    'TFIDFInference',
    'PredictionOrchestrator',
    'EnsembleMethod',
    'EnsembleResult',
    'PredictionExplainer'
]

__version__ = '1.0.0'
