"""
Inference Service for Fake News Detection
Handles predictions from BERT, RoBERTa, and TF-IDF models
"""

import torch
import joblib
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    RobertaTokenizer, 
    RobertaForSequenceClassification
)
import warnings
warnings.filterwarnings('ignore')


class ModelInference:
    """Base class for model inference"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def predict(self, text: str) -> Dict:
        """
        Make prediction on input text
        Returns: Dictionary with prediction, confidence, and probabilities
        """
        raise NotImplementedError("Subclasses must implement predict method")


class BERTInference(ModelInference):
    """BERT model inference"""
    
    def __init__(self, model_path: str = "models/bert/final_model"):
        super().__init__(model_path)
        self.load_model()
    
    def load_model(self):
        """Load BERT model and tokenizer"""
        try:
            print(f"Loading BERT model from {self.model_path}...")
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=2
            )
            self.model.to(self.device)
            self.model.eval()
            print("BERT model loaded successfully!")
        except Exception as e:
            print(f"Error loading BERT model: {e}")
            raise
    
    def predict(self, text: str, max_length: int = 512) -> Dict:
        """
        Predict using BERT model
        
        Args:
            text: Input text to classify
            max_length: Maximum sequence length
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                padding="max_length",
                truncation=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction].item()
            
            return {
                'model': 'BERT',
                'prediction': 'FAKE' if prediction == 1 else 'REAL',
                'prediction_label': prediction,
                'confidence': confidence,
                'probabilities': {
                    'real': probabilities[0][0].item(),
                    'fake': probabilities[0][1].item()
                }
            }
        except Exception as e:
            print(f"Error in BERT prediction: {e}")
            return {
                'model': 'BERT',
                'prediction': 'ERROR',
                'confidence': 0.0,
                'error': str(e)
            }


class RoBERTaInference(ModelInference):
    """RoBERTa model inference"""
    
    def __init__(self, model_path: str = "models/roberta/final_model"):
        super().__init__(model_path)
        self.load_model()
    
    def load_model(self):
        """Load RoBERTa model and tokenizer"""
        try:
            print(f"Loading RoBERTa model from {self.model_path}...")
            self.tokenizer = RobertaTokenizer.from_pretrained(self.model_path)
            self.model = RobertaForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=2
            )
            self.model.to(self.device)
            self.model.eval()
            print("RoBERTa model loaded successfully!")
        except Exception as e:
            print(f"Error loading RoBERTa model: {e}")
            raise
    
    def predict(self, text: str, max_length: int = 512) -> Dict:
        """
        Predict using RoBERTa model
        
        Args:
            text: Input text to classify
            max_length: Maximum sequence length
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                padding="max_length",
                truncation=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction].item()
            
            return {
                'model': 'RoBERTa',
                'prediction': 'FAKE' if prediction == 1 else 'REAL',
                'prediction_label': prediction,
                'confidence': confidence,
                'probabilities': {
                    'real': probabilities[0][0].item(),
                    'fake': probabilities[0][1].item()
                }
            }
        except Exception as e:
            print(f"Error in RoBERTa prediction: {e}")
            return {
                'model': 'RoBERTa',
                'prediction': 'ERROR',
                'confidence': 0.0,
                'error': str(e)
            }


class TFIDFInference(ModelInference):
    """TF-IDF + ML model inference"""
    
    def __init__(self, 
                 model_path: str = "models/tf_idf/fake_news_classifier.joblib",
                 vectorizer_path: str = "models/tf_idf/tfidf_model.joblib"):
        super().__init__(model_path)
        self.vectorizer_path = Path(vectorizer_path)
        self.vectorizer = None
        self.load_model()
    
    def load_model(self):
        """Load TF-IDF vectorizer and classifier model"""
        try:
            print(f"Loading TF-IDF model from {self.model_path}...")
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
            print("TF-IDF model loaded successfully!")
        except Exception as e:
            print(f"Error loading TF-IDF model: {e}")
            raise
    
    def predict(self, text: str) -> Dict:
        """
        Predict using TF-IDF + ML model
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Vectorize input
            text_vectorized = self.vectorizer.transform([text])
            
            # Get prediction
            prediction = self.model.predict(text_vectorized)[0]
            
            # Get probabilities if available
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(text_vectorized)[0]
                confidence = probabilities[prediction]
            else:
                # For models without probability estimates
                probabilities = [0.5, 0.5]
                confidence = 0.5
            
            return {
                'model': 'TF-IDF',
                'prediction': 'FAKE' if prediction == 1 else 'REAL',
                'prediction_label': int(prediction),
                'confidence': float(confidence),
                'probabilities': {
                    'real': float(probabilities[0]),
                    'fake': float(probabilities[1])
                }
            }
        except Exception as e:
            print(f"Error in TF-IDF prediction: {e}")
            return {
                'model': 'TF-IDF',
                'prediction': 'ERROR',
                'confidence': 0.0,
                'error': str(e)
            }


class MultiModelInference:
    """Handles inference from all three models"""
    
    def __init__(self, 
                 use_bert: bool = True,
                 use_roberta: bool = True,
                 use_tfidf: bool = True):
        """
        Initialize multi-model inference
        
        Args:
            use_bert: Whether to use BERT model
            use_roberta: Whether to use RoBERTa model
            use_tfidf: Whether to use TF-IDF model
        """
        self.models = {}
        
        # Load models based on flags
        if use_bert:
            try:
                self.models['BERT'] = BERTInference()
            except Exception as e:
                print(f"Could not load BERT model: {e}")
        
        if use_roberta:
            try:
                self.models['RoBERTa'] = RoBERTaInference()
            except Exception as e:
                print(f"Could not load RoBERTa model: {e}")
        
        if use_tfidf:
            try:
                self.models['TF-IDF'] = TFIDFInference()
            except Exception as e:
                print(f"Could not load TF-IDF model: {e}")
        
        if not self.models:
            raise ValueError("No models could be loaded!")
        
        print(f"\nSuccessfully loaded {len(self.models)} model(s): {list(self.models.keys())}")
    
    def predict_all(self, text: str) -> Dict[str, Dict]:
        """
        Get predictions from all loaded models
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with predictions from each model
        """
        predictions = {}
        
        for model_name, model in self.models.items():
            print(f"\nRunning {model_name} prediction...")
            predictions[model_name] = model.predict(text)
        
        return predictions
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Dict]]:
        """
        Get predictions for multiple texts
        
        Args:
            texts: List of input texts to classify
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i, text in enumerate(texts):
            print(f"\n{'='*50}")
            print(f"Processing text {i+1}/{len(texts)}")
            print(f"{'='*50}")
            results.append(self.predict_all(text))
        
        return results


if __name__ == "__main__":
    # Test the inference system
    print("="*70)
    print("Testing Multi-Model Inference System")
    print("="*70)
    
    # Initialize inference system
    try:
        inference = MultiModelInference(
            use_bert=True,
            use_roberta=True,
            use_tfidf=True
        )
    except Exception as e:
        print(f"Error initializing inference system: {e}")
        exit(1)
    
    # Test texts
    test_texts = [
        "Scientists discover groundbreaking cure for cancer using AI technology.",
        "BREAKING: Aliens have landed in New York City, government confirms!"
    ]
    
    # Run predictions
    for i, text in enumerate(test_texts, 1):
        print(f"\n{'='*70}")
        print(f"Test Case {i}")
        print(f"{'='*70}")
        print(f"Text: {text[:100]}...")
        
        predictions = inference.predict_all(text)
        
        print("\nPredictions:")
        for model_name, result in predictions.items():
            print(f"\n{model_name}:")
            print(f"  Prediction: {result.get('prediction', 'N/A')}")
            print(f"  Confidence: {result.get('confidence', 0):.4f}")
            if 'probabilities' in result:
                print(f"  Real: {result['probabilities']['real']:.4f}")
                print(f"  Fake: {result['probabilities']['fake']:.4f}")
