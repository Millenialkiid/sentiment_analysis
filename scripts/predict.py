#!/usr/bin/env python3
"""
Prediction script for sentiment analysis model.
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model.model import SentimentModel
from src.utils.preprocessing import TextPreprocessor, SentimentAnalyzer
from src.config.config import TrainingConfig

def load_model(model_path: str = None) -> SentimentModel:
    """Load a trained model."""
    if model_path is None:
        config = TrainingConfig()
        model_path = str(config.SAVE_MODEL_PATH)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    model = SentimentModel()
    model.load_model(model_path)
    return model

def predict_sentiment(text: str, model: SentimentModel) -> dict:
    """Predict sentiment for a single text."""
    preprocessor = TextPreprocessor()
    analyzer = SentimentAnalyzer()
    
    # Preprocess text
    processed_input = preprocessor.preprocess_text(text)
    
    # Get prediction
    prediction_proba = model.predict(processed_input)[0][0]
    
    # Analyze result
    result = analyzer.interpret_prediction(prediction_proba)
    
    return result

def interactive_mode(model: SentimentModel):
    """Interactive prediction mode."""
    print("=== Interactive Sentiment Analysis ===")
    print("Enter text to analyze sentiment (type 'quit' to exit):")
    print("-" * 50)
    
    while True:
        try:
            text = input("\nEnter text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not text:
                print("Please enter some text.")
                continue
            
            result = predict_sentiment(text, model)
            
            print(f"\nText: '{text}'")
            print(f"Sentiment: {result['sentiment']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Probability: {result['probability']:.4f}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def batch_predict(texts: list, model: SentimentModel) -> list:
    """Predict sentiment for multiple texts."""
    results = []
    
    for text in texts:
        try:
            result = predict_sentiment(text, model)
            result['text'] = text
            results.append(result)
        except Exception as e:
            print(f"Error processing text '{text}': {e}")
    
    return results

def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Predict sentiment using trained model')
    parser.add_argument('--model-path', type=str, help='Path to trained model')
    parser.add_argument('--text', type=str, help='Single text to analyze')
    parser.add_argument('--file', type=str, help='File containing texts to analyze (one per line)')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    try:
        # Load model
        print("Loading model...")
        model = load_model(args.model_path)
        print("Model loaded successfully!")
        
        if args.interactive:
            interactive_mode(model)
        
        elif args.text:
            result = predict_sentiment(args.text, model)
            print(f"\nText: '{args.text}'")
            print(f"Sentiment: {result['sentiment']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Probability: {result['probability']:.4f}")
        
        elif args.file:
            if not os.path.exists(args.file):
                print(f"File not found: {args.file}")
                sys.exit(1)
            
            with open(args.file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            print(f"Analyzing {len(texts)} texts...")
            results = batch_predict(texts, model)
            
            print("\nResults:")
            print("-" * 80)
            for i, result in enumerate(results, 1):
                print(f"{i}. Text: '{result['text']}'")
                print(f"   Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.4f})")
                print()
        
        else:
            # Default sample predictions
            sample_texts = [
                "This movie was absolutely fantastic! I loved every minute of it.",
                "This film was a complete disaster. The plot was boring and annoying.",
                "The movie had some good moments, but it was also quite slow at times.",
                "Highly recommend this masterpiece! A true cinematic achievement.",
                "Waste of time and money. Avoid at all costs."
            ]
            
            print("Running sample predictions...")
            results = batch_predict(sample_texts, model)
            
            print("\nSample Results:")
            print("-" * 80)
            for i, result in enumerate(results, 1):
                print(f"{i}. Text: '{result['text']}'")
                print(f"   Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.4f})")
                print()
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()