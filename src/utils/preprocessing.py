"""
Text preprocessing utilities for sentiment analysis.
"""

import re
import string
import numpy as np
from typing import List, Union
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
import logging

from ..config.config import ModelConfig, PredictionConfig

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Handle text preprocessing for sentiment analysis."""
    
    def __init__(self):
        self.config = ModelConfig()
        self.prediction_config = PredictionConfig()
        self.word_index = None
        self._load_word_index()
    
    def _load_word_index(self) -> None:
        """Load the word index from IMDb dataset."""
        try:
            self.word_index = imdb.get_word_index()
            logger.info("Word index loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading word index: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing HTML tags, converting to lowercase, and removing punctuation."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def text_to_sequences(self, text: str) -> List[int]:
        """Convert text to sequence of integers using IMDb word index."""
        if self.word_index is None:
            raise ValueError("Word index not loaded")
        
        cleaned_text = self.clean_text(text)
        tokenized_text = []
        
        for word in cleaned_text.split():
            # IMDb dataset uses word_index + 3 for unknown words
            # 0: padding, 1: start of sequence, 2: unknown word
            word_idx = self.word_index.get(word, 2) + 3
            tokenized_text.append(word_idx)
        
        return tokenized_text
    
    def pad_sequences(self, sequences: Union[List[int], List[List[int]]], maxlen: int = None) -> np.ndarray:
        """Pad sequences to the same length."""
        if maxlen is None:
            maxlen = self.config.MAX_LENGTH
        
        # Handle single sequence
        if isinstance(sequences[0], int):
            sequences = [sequences]
        
        return pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')
    
    def preprocess_text(self, text: str) -> np.ndarray:
        """Full preprocessing pipeline for a single text."""
        try:
            # Convert text to sequence
            sequence = self.text_to_sequences(text)
            
            # Pad sequence
            padded_sequence = self.pad_sequences(sequence)
            
            return padded_sequence
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            raise
    
    def preprocess_batch(self, texts: List[str]) -> np.ndarray:
        """Preprocess a batch of texts."""
        try:
            sequences = [self.text_to_sequences(text) for text in texts]
            padded_sequences = self.pad_sequences(sequences)
            
            return padded_sequences
            
        except Exception as e:
            logger.error(f"Error preprocessing batch: {e}")
            raise
    
    def decode_sequence(self, sequence: List[int]) -> str:
        """Decode a sequence back to text (for debugging purposes)."""
        if self.word_index is None:
            raise ValueError("Word index not loaded")
        
        # Create reverse word index
        reverse_word_index = {v + 3: k for k, v in self.word_index.items()}
        reverse_word_index[0] = '<PAD>'
        reverse_word_index[1] = '<START>'
        reverse_word_index[2] = '<UNK>'
        
        return ' '.join([reverse_word_index.get(i, '<UNK>') for i in sequence if i != 0])
    
    def get_vocabulary_size(self) -> int:
        """Get the size of the vocabulary."""
        return self.config.NUM_WORDS
    
    def get_max_length(self) -> int:
        """Get the maximum sequence length."""
        return self.config.MAX_LENGTH

class SentimentAnalyzer:
    """Analyze sentiment predictions."""
    
    def __init__(self):
        self.config = PredictionConfig()
    
    def interpret_prediction(self, prediction_prob: float) -> dict:
        """Interpret model prediction probability."""
        sentiment = "Positive" if prediction_prob >= self.config.POSITIVE_THRESHOLD else "Negative"
        confidence = prediction_prob if sentiment == "Positive" else 1 - prediction_prob
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'probability': prediction_prob,
            'threshold': self.config.POSITIVE_THRESHOLD
        }
    
    def get_sentiment_label(self, prediction_prob: float) -> str:
        """Get sentiment label from prediction probability."""
        return "Positive" if prediction_prob >= self.config.POSITIVE_THRESHOLD else "Negative"
    
    def format_prediction_output(self, text: str, prediction_prob: float) -> str:
        """Format prediction output for display."""
        result = self.interpret_prediction(prediction_prob)
        
        output = f"""
Review: '{text}'
Predicted Probability of Positive Sentiment: {result['probability']:.4f}
Predicted Sentiment: {result['sentiment']}
Confidence: {result['confidence']:.4f}
"""
        return output.strip()