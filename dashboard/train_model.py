"""Script to train the job scam detection model using collected data."""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from typing import Dict, Optional
from sklearn.model_selection import train_test_split
from pipeline.model import JobScamDetector
from pipeline.feature_engineering import JobFeatureExtractor
from data_collection import collect_training_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(
    training_data: pd.DataFrame,
    model_dir: str = "models",
    test_size: float = 0.2
) -> Dict[str, float]:
    """
    Train model using collected data and save results.
    
    Args:
        training_data: DataFrame with training data
        model_dir: Directory to save model and metrics
        test_size: Proportion of data to use for testing
        
    Returns:
        Dictionary with model metrics
    """
    try:
        # Create model directory if it doesn't exist
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        feature_extractor = JobFeatureExtractor()
        model = JobScamDetector(
            model_path=f"{model_dir}/fraud_detector.pkl",
            metrics_path=f"{model_dir}/metrics.json"
        )
        
        # Extract features
        logger.info("Extracting features from training data...")
        features = feature_extractor.engineer_features(training_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            training_data['is_scam'],
            test_size=test_size,
            stratify=training_data['is_scam'],
            random_state=42
        )
        
        # Train model
        logger.info("Training model...")
        metrics = model.train(X_train, y_train)
        
        # Evaluate on test set
        logger.info("Evaluating model...")
        test_metrics = model._evaluate_model(X_test, y_test)
        
        # Perform cross-validation
        logger.info("Performing cross-validation...")
        cv_metrics = model.cross_validate(features, training_data['is_scam'])
        
        # Combine all metrics
        all_metrics = {
            'training': metrics,
            'test': test_metrics,
            'cross_validation': cv_metrics
        }
        
        # Save metrics
        metrics_path = Path(model_dir) / "all_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        
        logger.info("Model training completed successfully!")
        return all_metrics
        
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise

def main():
    """Main function to run model training."""
    try:
        # Collect training data
        logger.info("Collecting training data...")
        db_path = "dashboard.db"  # Update with your database path
        training_data = collect_training_data(db_path)
        
        if len(training_data) < 10:
            logger.warning("Insufficient training data. Need at least 10 samples.")
            return
        
        # Train model
        metrics = train_model(training_data)
        
        # Log results
        logger.info("Training Results:")
        logger.info(f"Training metrics: {metrics['training']}")
        logger.info(f"Test metrics: {metrics['test']}")
        logger.info(f"Cross-validation metrics: {metrics['cross_validation']}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main() 