import os
import logging
from src.model.model import SentimentModel
from src.utils.preprocessing import TextPreprocessor, SentimentAnalyzer
from src.config.config import TrainingConfig

# Global vars to hold the loaded model and preprocessor
_m = None
_p = None

def load_model_and_preprocessor():
    """
    Loads the model and preprocessor into memory.
    """
    global _m, _p
    
    # Return if already loaded
    if _m and _p:
        return _m, _p

    try:
        logging.info("Loading model and preprocessor...")
        
        # Get model path from your config
        cfg = TrainingConfig()
        path = str(cfg.SAVE_MODEL_PATH)

        if not os.path.exists(path):
            logging.error(f"Model file not found at {path}")
            raise FileNotFoundError(f"Model file not found at {path}. Please run train.py first.")

        # Initialize the model (this builds the architecture)
        _m = SentimentModel()
        _m.build_model()
        
        # Load the *weights* into the existing model
        # This fixes the "No model config found" error
        _m.model.load_weights(path)
        
        # Initialize preprocessor
        _p = TextPreprocessor()
        
        logging.info("Model and preprocessor loaded successfully.")
        return _m, _p

    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise e

def predict_sentiment(txt: str) -> dict:
    """
    Predicts sentiment for a single text review.
    """
    try:
        # Ensure model is loaded
        m, p = load_model_and_preprocessor()
        
        proc_txt = p.preprocess_text(txt)
        prob = m.predict(proc_txt)[0][0]
        
        # Use your SentimentAnalyzer
        a = SentimentAnalyzer()
        res = a.interpret_prediction(prob)
            
        return res

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return {
            "sentiment": "Error",
            "confidence": 0.0,
            "probability": 0.0,
            "error": str(e)
        }

