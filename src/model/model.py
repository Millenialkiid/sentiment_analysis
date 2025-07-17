"""
Neural network model architecture for sentiment analysis.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from typing import Optional, Tuple
import logging

from ..config.config import ModelConfig, TrainingConfig

logger = logging.getLogger(__name__)

class SentimentModel:
    """Bidirectional LSTM model for sentiment analysis."""
    
    def __init__(self, embedding_matrix: Optional[np.ndarray] = None):
        self.config = ModelConfig()
        self.training_config = TrainingConfig()
        self.model = None
        self.embedding_matrix = embedding_matrix
        self.history = None
    
    def build_model(self) -> Sequential:
        """Build the Bidirectional LSTM model with GloVe embeddings."""
        logger.info("Building Bidirectional LSTM model...")
        
        model = Sequential()
        
        # Embedding layer
        if self.embedding_matrix is not None:
            model.add(Embedding(
                input_dim=self.config.NUM_WORDS,
                output_dim=self.config.EMBEDDING_DIM,
                weights=[self.embedding_matrix],
                input_length=self.config.MAX_LENGTH,
                trainable=True,
                name='embedding'
            ))
            logger.info("Using pre-trained GloVe embeddings")
        else:
            model.add(Embedding(
                input_dim=self.config.NUM_WORDS,
                output_dim=self.config.EMBEDDING_DIM,
                input_length=self.config.MAX_LENGTH,
                trainable=True,
                name='embedding'
            ))
            logger.info("Using trainable embeddings")
        
        # Bidirectional LSTM layer
        model.add(Bidirectional(
            LSTM(
                units=self.config.LSTM_UNITS,
                dropout=self.config.DROPOUT_RATE,
                recurrent_dropout=self.config.RECURRENT_DROPOUT_RATE,
                return_sequences=False
            ),
            name='bidirectional_lstm'
        ))
        
        # Dropout layer
        model.add(Dropout(self.config.DENSE_DROPOUT_RATE, name='dropout'))
        
        # Dense output layer
        model.add(Dense(units=1, activation='sigmoid', name='output'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        logger.info("Model built successfully")
        
        return model
    
    def get_model_summary(self) -> str:
        """Get model summary as string."""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        import io
        import sys
        
        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        self.model.summary()
        
        sys.stdout = old_stdout
        summary = buffer.getvalue()
        
        return summary
    
    def get_callbacks(self) -> list:
        """Get training callbacks."""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=str(self.training_config.SAVE_WEIGHTS_PATH),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=0.0001,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray, 
              x_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              verbose: int = 1) -> tf.keras.callbacks.History:
        """Train the model."""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        logger.info("Starting model training...")
        
        # Prepare validation data
        if x_val is not None and y_val is not None:
            validation_data = (x_val, y_val)
        else:
            validation_data = None
        
        # Train the model
        self.history = self.model.fit(
            x_train, y_train,
            batch_size=self.config.BATCH_SIZE,
            epochs=self.config.EPOCHS,
            validation_data=validation_data,
            validation_split=self.config.VALIDATION_SPLIT if validation_data is None else None,
            callbacks=self.get_callbacks(),
            verbose=verbose
        )
        
        logger.info("Model training completed")
        return self.history
    
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        """Evaluate the model on test data."""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        logger.info("Evaluating model on test data...")
        
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        
        logger.info(f"Test Loss: {loss:.4f}")
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        
        return loss, accuracy
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions on input data."""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        return self.model.predict(x)
    
    def save_model(self, filepath: Optional[str] = None) -> None:
        """Save the entire model."""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        if filepath is None:
            filepath = str(self.training_config.SAVE_MODEL_PATH)
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def save_weights(self, filepath: Optional[str] = None) -> None:
        """Save model weights only."""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        if filepath is None:
            filepath = str(self.training_config.SAVE_WEIGHTS_PATH)
        
        self.model.save_weights(filepath)
        logger.info(f"Model weights saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a complete model."""
        self.model = tf.keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def load_weights(self, filepath: str) -> None:
        """Load model weights."""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        self.model.load_weights(filepath)
        logger.info(f"Model weights loaded from {filepath}")
    
    def get_training_history(self) -> Optional[tf.keras.callbacks.History]:
        """Get training history."""
        return self.history