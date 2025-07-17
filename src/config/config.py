"""
Configuration settings for the sentiment analysis project.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
GLOVE_DIR = DATA_DIR / "glove"

# Model hyperparameters
class ModelConfig:
    NUM_WORDS = 10000
    MAX_LENGTH = 256
    EMBEDDING_DIM = 100
    LSTM_UNITS = 128
    DROPOUT_RATE = 0.3
    RECURRENT_DROPOUT_RATE = 0.3
    DENSE_DROPOUT_RATE = 0.4
    BATCH_SIZE = 64
    EPOCHS = 10
    VALIDATION_SPLIT = 0.2

# GloVe embeddings
class EmbeddingConfig:
    GLOVE_URL = "http://nlp.stanford.edu/data/glove.6B.zip"
    GLOVE_ZIP_PATH = DATA_DIR / "glove.6B.zip"
    GLOVE_EXTRACT_DIR = GLOVE_DIR
    GLOVE_FILE = GLOVE_EXTRACT_DIR / "glove.6B.100d.txt"

# Training configuration
class TrainingConfig:
    RANDOM_SEED = 42
    SAVE_MODEL_PATH = MODELS_DIR / "sentiment_model.h5"
    SAVE_WEIGHTS_PATH = MODELS_DIR / "sentiment_weights.h5"
    TENSORBOARD_LOG_DIR = PROJECT_ROOT / "logs"

# Prediction configuration
class PredictionConfig:
    POSITIVE_THRESHOLD = 0.5
    
# Create necessary directories
def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        DATA_DIR,
        MODELS_DIR,
        GLOVE_DIR,
        TrainingConfig.TENSORBOARD_LOG_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Environment-specific settings
def get_environment():
    """Get the current environment (development, production, etc.)."""
    return os.getenv('ENVIRONMENT', 'development')

# Logging configuration
import logging

def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('sentiment_analysis.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)