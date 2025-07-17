"""
Training pipeline for sentiment analysis model.
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from typing import Tuple, Dict, Any
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from ..config.config import TrainingConfig, setup_logging
from ..data.data_loader import DataLoader
from ..model.model import SentimentModel
from ..utils.preprocessing import TextPreprocessor, SentimentAnalyzer

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handle the complete training pipeline."""
    
    def __init__(self, log_level=logging.INFO):
        self.config = TrainingConfig()
        self.data_loader = DataLoader()
        self.model = None
        self.preprocessor = TextPreprocessor()
        self.analyzer = SentimentAnalyzer()
        
        # Setup logging
        setup_logging(log_level)
        
        # Set random seed for reproducibility
        tf.random.set_seed(self.config.RANDOM_SEED)
        np.random.seed(self.config.RANDOM_SEED)
    
    def prepare_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Prepare all data for training."""
        logger.info("Preparing data...")
        return self.data_loader.prepare_data()
    
    def build_and_train_model(self, 
                             train_data: Tuple[np.ndarray, np.ndarray],
                             test_data: Tuple[np.ndarray, np.ndarray],
                             embedding_matrix: np.ndarray) -> SentimentModel:
        """Build and train the model."""
        x_train, y_train = train_data
        x_test, y_test = test_data
        
        logger.info("Building model...")
        self.model = SentimentModel(embedding_matrix)
        self.model.build_model()
        
        # Print model summary
        logger.info("Model Summary:")
        logger.info(self.model.get_model_summary())
        
        # Train the model
        logger.info("Starting training...")
        history = self.model.train(x_train, y_train)
        
        # Evaluate on test data
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test)
        
        return self.model
    
    def evaluate_model(self, test_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        """Evaluate the trained model and generate detailed metrics."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call build_and_train_model() first.")
        
        x_test, y_test = test_data
        
        logger.info("Evaluating model...")
        
        # Get predictions
        y_pred_proba = self.model.predict(x_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test)
        
        # Generate classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }
        
        # Log results
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info("Classification Report:")
        logger.info(classification_report(y_test, y_pred))
        logger.info("Confusion Matrix:")
        logger.info(conf_matrix)
        
        return results
    
    def plot_training_history(self, save_path: str = None) -> None:
        """Plot training history."""
        if self.model is None or self.model.history is None:
            logger.warning("No training history available.")
            return
        
        history = self.model.history.history
        
        plt.figure(figsize=(12, 4))
        
        # Plot training & validation accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot training & validation loss
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Training history plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_confusion_matrix(self, conf_matrix: np.ndarray, save_path: str = None) -> None:
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Confusion matrix plot saved to {save_path}")
        else:
            plt.show()
    
    def test_predictions(self, sample_texts: list = None) -> None:
        """Test model predictions on sample texts."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call build_and_train_model() first.")
        
        if sample_texts is None:
            sample_texts = [
                "This movie was absolutely fantastic! I loved every minute of it. The acting was superb and the story was captivating.",
                "This film was a complete disaster. The plot was boring, the characters were annoying, and I fell asleep halfway through.",
                "The movie had some good moments, but it was also quite slow at times. Overall, it was just okay.",
                "Highly recommend this masterpiece! A true cinematic achievement that will stay with you.",
                "Waste of time and money. Avoid at all costs. Nothing redeeming about this production."
            ]
        
        logger.info("Testing predictions on sample texts...")
        
        for text in sample_texts:
            # Preprocess text
            processed_input = self.preprocessor.preprocess_text(text)
            
            # Get prediction
            prediction_proba = self.model.predict(processed_input)[0][0]
            
            # Format output
            output = self.analyzer.format_prediction_output(text, prediction_proba)
            logger.info(output)
            print(output)
            print("-" * 50)
    
    def save_model(self, save_weights_only: bool = False) -> None:
        """Save the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call build_and_train_model() first.")
        
        if save_weights_only:
            self.model.save_weights()
        else:
            self.model.save_model()
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete training pipeline."""
        logger.info("Starting complete training pipeline...")
        
        # Prepare data
        (x_train, y_train), (x_test, y_test), embedding_matrix = self.prepare_data()
        
        # Build and train model
        self.model = self.build_and_train_model(
            (x_train, y_train), 
            (x_test, y_test), 
            embedding_matrix
        )
        
        # Evaluate model
        results = self.evaluate_model((x_test, y_test))
        
        # Test predictions
        self.test_predictions()
        
        # Save model
        self.save_model()
        
        logger.info("Training pipeline completed successfully!")
        
        return results