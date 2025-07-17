"""
Data loading and GloVe embedding utilities.
"""

import os
import zipfile
import urllib.request
import numpy as np
from pathlib import Path
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import Tuple, Dict, Optional
import logging

from ..config.config import ModelConfig, EmbeddingConfig, create_directories

logger = logging.getLogger(__name__)

class DataLoader:
    """Handle data loading for sentiment analysis."""
    
    def __init__(self):
        self.config = ModelConfig()
        self.embedding_config = EmbeddingConfig()
        create_directories()
    
    def download_glove_embeddings(self) -> None:
        """Download GloVe embeddings if not already present."""
        if self.embedding_config.GLOVE_FILE.exists():
            logger.info("GloVe embeddings already exist. Skipping download.")
            return
        
        logger.info("Downloading GloVe embeddings...")
        
        try:
            # Download the zip file
            urllib.request.urlretrieve(
                self.embedding_config.GLOVE_URL,
                self.embedding_config.GLOVE_ZIP_PATH
            )
            
            # Extract the zip file
            with zipfile.ZipFile(self.embedding_config.GLOVE_ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(self.embedding_config.GLOVE_EXTRACT_DIR)
            
            # Clean up zip file
            self.embedding_config.GLOVE_ZIP_PATH.unlink()
            
            logger.info("GloVe embeddings downloaded and extracted successfully.")
            
        except Exception as e:
            logger.error(f"Error downloading GloVe embeddings: {e}")
            raise
    
    def load_imdb_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Load and preprocess IMDb dataset."""
        logger.info(f"Loading IMDb dataset with {self.config.NUM_WORDS} words and maxlen {self.config.MAX_LENGTH}...")
        
        try:
            (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=self.config.NUM_WORDS)
            
            logger.info(f"Training samples: {len(x_train)}")
            logger.info(f"Test samples: {len(x_test)}")
            
            # Pad sequences
            logger.info("Padding sequences...")
            x_train = pad_sequences(x_train, maxlen=self.config.MAX_LENGTH, padding='post', truncating='post')
            x_test = pad_sequences(x_test, maxlen=self.config.MAX_LENGTH, padding='post', truncating='post')
            
            logger.info("Sequences padded successfully.")
            
            return (x_train, y_train), (x_test, y_test)
            
        except Exception as e:
            logger.error(f"Error loading IMDb data: {e}")
            raise
    
    def load_glove_embeddings(self) -> Dict[str, np.ndarray]:
        """Load GloVe embeddings from file."""
        if not self.embedding_config.GLOVE_FILE.exists():
            logger.warning("GloVe file not found. Attempting to download...")
            self.download_glove_embeddings()
        
        logger.info(f"Loading GloVe embeddings from {self.embedding_config.GLOVE_FILE}...")
        
        embeddings_index = {}
        
        try:
            with open(self.embedding_config.GLOVE_FILE, encoding='utf8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
            
            logger.info(f"Found {len(embeddings_index)} word vectors in GloVe.")
            return embeddings_index
            
        except Exception as e:
            logger.error(f"Error loading GloVe embeddings: {e}")
            raise
    
    def create_embedding_matrix(self, embeddings_index: Dict[str, np.ndarray]) -> np.ndarray:
        """Create embedding matrix for the model."""
        logger.info("Creating embedding matrix...")
        
        try:
            embedding_matrix = np.zeros((self.config.NUM_WORDS, self.config.EMBEDDING_DIM))
            word_index = imdb.get_word_index()
            
            for word, i in word_index.items():
                if i < self.config.NUM_WORDS:
                    embedding_vector = embeddings_index.get(word)
                    if embedding_vector is not None:
                        embedding_matrix[i] = embedding_vector
            
            logger.info("Embedding matrix created successfully.")
            return embedding_matrix
            
        except Exception as e:
            logger.error(f"Error creating embedding matrix: {e}")
            raise
    
    def get_word_index(self) -> Dict[str, int]:
        """Get the word index from IMDb dataset."""
        return imdb.get_word_index()
    
    def prepare_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Prepare all data for training."""
        logger.info("Preparing data for training...")
        
        # Load IMDb data
        (x_train, y_train), (x_test, y_test) = self.load_imdb_data()
        
        # Load GloVe embeddings
        embeddings_index = self.load_glove_embeddings()
        
        # Create embedding matrix
        embedding_matrix = self.create_embedding_matrix(embeddings_index)
        
        logger.info("Data preparation complete.")
        
        return (x_train, y_train), (x_test, y_test), embedding_matrix