#!/usr/bin/env python3
"""
Training script for sentiment analysis model.
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model.trainer import ModelTrainer
from src.config.config import create_directories

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train sentiment analysis model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--save-plots', action='store_true', help='Save training plots')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level (0, 1, 2)')
    
    args = parser.parse_args()
    
    # Create necessary directories
    create_directories()
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Update config with command line arguments
    trainer.data_loader.config.EPOCHS = args.epochs
    trainer.data_loader.config.BATCH_SIZE = args.batch_size
    
    try:
        # Run complete training pipeline
        results = trainer.run_complete_pipeline()
        
        # Save plots if requested
        if args.save_plots:
            trainer.plot_training_history('training_history.png')
            trainer.plot_confusion_matrix(results['confusion_matrix'], 'confusion_matrix.png')
        
        print(f"\nTraining completed successfully!")
        print(f"Final test accuracy: {results['test_accuracy']:.4f}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()