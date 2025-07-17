# Sentiment Analysis with Bidirectional LSTM

A deep learning project for sentiment analysis using Bidirectional LSTM with GloVe embeddings, trained on the IMDb movie reviews dataset.

## Features

- **Bidirectional LSTM** architecture for better context understanding
- **Pre-trained GloVe embeddings** for improved word representations
- **Modular code structure** for easy maintenance and extension
- **Comprehensive evaluation** with metrics and visualizations
- **Interactive prediction** capabilities
- **Batch processing** support

## Project Structure

```
sentiment_analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py          # Configuration settings
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py     # Data loading utilities
│   ├── model/
│   │   ├── __init__.py
│   │   ├── model.py           # Model architecture
│   │   └── trainer.py         # Training pipeline
│   └── utils/
│       ├── __init__.py
│       └── preprocessing.py   # Text preprocessing utilities
├── scripts/
│   ├── train.py               # Training script
│   └── predict.py             # Prediction script
├── data/                      # Data directory (created automatically)
├── models/                    # Saved models directory
└── logs/                      # Training logs
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd sentiment_analysis
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

**Basic training:**
```bash
python scripts/train.py
```

**Training with custom parameters:**
```bash
python scripts/train.py --epochs 15 --batch-size 32 --save-plots
```

The training script will:
- Automatically download GloVe embeddings
- Load and preprocess the IMDb dataset
- Train the Bidirectional LSTM model
- Evaluate the model and display metrics
- Save the trained model

### Making Predictions

**Interactive mode:**
```bash
python scripts/predict.py --interactive
```

**Single text prediction:**
```bash
python scripts/predict.py --text "This movie was absolutely fantastic!"
```

**Batch prediction from file:**
```bash
python scripts/predict.py --file path/to/your/texts.txt
```

**Sample predictions:**
```bash
python scripts/predict.py
```

### Using the Model Programmatically

```python
from src.model.model import SentimentModel
from src.utils.preprocessing import TextPreprocessor, SentimentAnalyzer

# Load trained model
model = SentimentModel()
model.load_model('models/sentiment_model.h5')

# Initialize preprocessor and analyzer
preprocessor = TextPreprocessor()
analyzer = SentimentAnalyzer()

# Predict sentiment
text = "This movie was great!"
processed_input = preprocessor.preprocess_text(text)
prediction_proba = model.predict(processed_input)[0][0]
result = analyzer.interpret_prediction(prediction_proba)

print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.4f}")
```

## Model Architecture

The model uses a Bidirectional LSTM architecture:

1. **Embedding Layer**: Uses pre-trained GloVe embeddings (100-dimensional)
2. **Bidirectional LSTM**: 128 units with dropout for regularization
3. **Dense Layer**: Single neuron with sigmoid activation for binary classification

### Hyperparameters

- **Vocabulary Size**: 10,000 words
- **Sequence Length**: 256 tokens
- **Embedding Dimension**: 100 (GloVe)
- **LSTM Units**: 128
- **Dropout Rate**: 0.3
- **Batch Size**: 64
- **Epochs**: 10

## Performance

The model achieves approximately:
- **Test Accuracy**: ~87-89%
- **Training Time**: ~10-15 minutes (depending on hardware)

## Configuration

All configuration settings are centralized in `src/config/config.py`. You can modify:

- Model hyperparameters
- File paths
- Training settings
- Prediction thresholds

## Data

The project uses:
- **IMDb Dataset**: 50,000 movie reviews (25,000 for training, 25,000 for testing)
- **GloVe Embeddings**: Pre-trained word vectors (glove.6B.100d.txt)

Data is automatically downloaded when you run the training script.

## Requirements

- Python 3.7+
- TensorFlow 2.12+
- NumPy
- scikit-learn
- matplotlib
- seaborn

See `requirements.txt` for complete dependencies.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- [GloVe](https://nlp.stanford.edu/projects/glove/) for pre-trained word embeddings
- [IMDb Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) for sentiment analysis data
- TensorFlow/Keras for the deep learning framework

## Troubleshooting

**Common Issues:**

1. **Memory Error**: Reduce batch size or sequence length in config
2. **Download Issues**: Check internet connection for GloVe download
3. **Import Errors**: Ensure all dependencies are installed correctly

**Getting Help:**
- Check the logs in `sentiment_analysis.log`
- Ensure all requirements are installed
- Verify Python version compatibility