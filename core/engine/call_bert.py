"""
BERT Model Interface
Handles loading and inference with fine-tuned BERT models
"""

import os
import logging
from typing import List, Dict, Optional, Union
import torch
import torch.nn.functional as F
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BERTDetector:
    """
    BERT-based AI text detector
    Handles model loading, tokenization, and inference
    """
    
    def __init__(
        self,
        model_path: str = "models/bert/final_model",
        device: str = None,
        max_length: int = 512,
        use_amp: bool = False
    ):
        """
        Initialize BERT detector
        
        Args:
            model_path: Path to the fine-tuned BERT model
            device: Device to use ('cpu', 'cuda', or None for auto-detect)
            max_length: Maximum sequence length for tokenization
            use_amp: Use automatic mixed precision (for GPU)
        """
        self.model_path = model_path
        self.max_length = max_length
        self.use_amp = use_amp
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self._load_model()
        self._load_tokenizer()
        
        # Model info
        self.model_name = self._get_model_name()
        
    def _load_model(self):
        """Load the fine-tuned BERT model"""
        try:
            logger.info(f"Loading model from {self.model_path}...")
            
            # Try loading with AutoModel (more flexible)
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_path,
                    num_labels=2  # Binary classification
                )
            except Exception as e:
                logger.warning(f"AutoModel loading failed: {e}")
                # Fallback to BertForSequenceClassification
                self.model = BertForSequenceClassification.from_pretrained(
                    self.model_path,
                    num_labels=2
                )
            
            # Move model to device
            self.model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            logger.info("✓ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model from {self.model_path}: {str(e)}")
    
    def _load_tokenizer(self):
        """Load the tokenizer"""
        try:
            # Try to load tokenizer from model path
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            except Exception:
                # Fallback to bert-base-uncased
                logger.warning("Loading default bert-base-uncased tokenizer")
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            
            logger.info("✓ Tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading tokenizer: {str(e)}")
            raise RuntimeError(f"Failed to load tokenizer: {str(e)}")
    
    def _get_model_name(self) -> str:
        """Get the base model name"""
        try:
            config = self.model.config
            return config._name_or_path
        except:
            return "bert-base-uncased"
    
    def predict(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 8
    ) -> List[Dict]:
        """
        Predict whether texts are AI-generated
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for inference
            
        Returns:
            List of prediction dictionaries
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        predictions = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_predictions = self._predict_batch(batch_texts)
            predictions.extend(batch_predictions)
        
        return predictions
    
    def _predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict on a batch of texts
        
        Args:
            texts: List of texts
            
        Returns:
            List of prediction dictionaries
        """
        try:
            # Tokenize
            encodings = self.tokenizer(
                texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # Inference
            with torch.no_grad():
                if self.use_amp and self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
            
            # Get logits and probabilities
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            
            # Get predictions
            predictions = []
            for i, prob in enumerate(probabilities):
                pred_dict = {
                    'probability': prob[1].item(),  # Probability of AI-generated (class 1)
                    'logits': logits[i].cpu().numpy().tolist(),
                    'label': int(torch.argmax(prob).item())
                }
                predictions.append(pred_dict)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def predict_with_attention(
        self,
        text: str
    ) -> Dict:
        """
        Predict with attention weights for interpretability
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with prediction and attention weights
        """
        try:
            # Tokenize
            encodings = self.tokenizer(
                text,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # Get tokens for interpretation
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            
            # Inference with attention output
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True
                )
            
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            attentions = outputs.attentions  # Tuple of attention tensors
            
            # Get last layer attention (average across heads)
            last_layer_attention = attentions[-1][0].mean(dim=0)  # Average across heads
            
            # Get attention for [CLS] token (used for classification)
            cls_attention = last_layer_attention[0].cpu().numpy()
            
            return {
                'probability': probabilities[0][1].item(),
                'label': int(torch.argmax(probabilities[0]).item()),
                'tokens': tokens,
                'attention_weights': cls_attention.tolist(),
                'important_tokens': self._get_important_tokens(tokens, cls_attention)
            }
            
        except Exception as e:
            logger.error(f"Error getting attention: {str(e)}")
            raise
    
    def _get_important_tokens(
        self,
        tokens: List[str],
        attention_weights: np.ndarray,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Get most important tokens based on attention weights
        
        Args:
            tokens: List of tokens
            attention_weights: Attention weights
            top_k: Number of top tokens to return
            
        Returns:
            List of token-weight pairs
        """
        # Filter out special tokens
        filtered_indices = [
            i for i, token in enumerate(tokens)
            if token not in ['[CLS]', '[SEP]', '[PAD]']
        ]
        
        # Get top-k indices
        filtered_weights = attention_weights[filtered_indices]
        top_indices = np.argsort(filtered_weights)[-top_k:][::-1]
        
        important_tokens = [
            {
                'token': tokens[filtered_indices[idx]],
                'weight': float(filtered_weights[idx]),
                'position': filtered_indices[idx]
            }
            for idx in top_indices
        ]
        
        return important_tokens
    
    def get_embeddings(
        self,
        texts: Union[str, List[str]]
    ) -> np.ndarray:
        """
        Get BERT embeddings for texts
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            encodings = self.tokenizer(
                texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict:
        """
        Get model information
        
        Returns:
            Dictionary with model details
        """
        return {
            'model_name': self.model_name,
            'model_path': self.model_path,
            'device': str(self.device),
            'max_length': self.max_length,
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'num_trainable_parameters': sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
        }
    
    def save_model(self, save_path: str):
        """
        Save the model and tokenizer
        
        Args:
            save_path: Path to save the model
        """
        try:
            os.makedirs(save_path, exist_ok=True)
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            logger.info(f"Model saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise


if __name__ == "__main__":
    # Example usage
    detector = BERTDetector(
        model_path="models/bert/final_model",
        device="cpu"
    )
    
    # Test prediction
    sample_texts = [
        "This is a sample text to test the detector.",
        "Machine learning models can classify text with high accuracy."
    ]
    
    predictions = detector.predict(sample_texts)
    
    for text, pred in zip(sample_texts, predictions):
        print(f"\nText: {text}")
        print(f"AI Probability: {pred['probability']:.4f}")
        print(f"Label: {'AI' if pred['label'] == 1 else 'Human'}")
    
    # Test with attention
    result = detector.predict_with_attention(sample_texts[0])
    print("\nTop important tokens:")
    for token_info in result['important_tokens'][:5]:
        print(f"  {token_info['token']}: {token_info['weight']:.4f}")