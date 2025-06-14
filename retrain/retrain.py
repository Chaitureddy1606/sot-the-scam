import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from datetime import datetime
import joblib
import json
import matplotlib.pyplot as plt

# Add parent directory to path to import from pipeline
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.preprocessing import preprocess_data, split_data
from pipeline.feature_engineering import FeatureExtractor
from pipeline.model import FraudDetector
from pipeline.explainability import ModelExplainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(data_path: str) -> pd.DataFrame:
    """
    Load and preprocess the training data.
    
    Args:
        data_path: Path to the training data file
        
    Returns:
        pd.DataFrame: Preprocessed data
    """
    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    
    logger.info("Preprocessing data")
    return preprocess_data(data)

def train_model(data: pd.DataFrame, model_dir: Path) -> None:
    """
    Train the model and save it along with performance metrics.
    
    Args:
        data: Preprocessed training data
        model_dir: Directory to save the model and metrics
    """
    # Create model directory if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Split data
    train_df, test_df = split_data(data)
    
    # Initialize feature extractor and extract features
    logger.info("Extracting features")
    feature_extractor = FeatureExtractor()
    X_train = feature_extractor.fit_transform(train_df)
    X_test = feature_extractor.transform(test_df)
    
    # Get target variables
    y_train = train_df['is_fraud'].values
    y_test = test_df['is_fraud'].values
    
    # Train model
    logger.info("Training model")
    model = FraudDetector(model_type='lightgbm')
    model.train(X_train, y_train)
    
    # Evaluate model
    logger.info("Evaluating model")
    metrics = model.evaluate(X_test, y_test)
    
    # Save model
    model_path = model_dir / "fraud_detector.pkl"
    logger.info(f"Saving model to {model_path}")
    model.save_model(model_path)
    
    # Save feature extractor
    feature_extractor_path = model_dir / "feature_extractor.pkl"
    logger.info(f"Saving feature extractor to {feature_extractor_path}")
    joblib.dump(feature_extractor, feature_extractor_path)
    
    # Save metrics
    metrics_path = model_dir / "metrics.json"
    logger.info(f"Saving metrics to {metrics_path}")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Generate and save feature importance plot
    logger.info("Generating feature importance plot")
    explainer = ModelExplainer(model, feature_extractor.get_feature_names())
    explainer.fit(X_train)
    explainer.plot_feature_importance(X_test)
    plt.savefig(model_dir / "feature_importance.png")
    plt.close()

def main():
    """Main function to retrain the model."""
    try:
        # Set up paths
        project_root = Path(__file__).parent.parent
        data_path = project_root / "data" / "jobs_sample.csv"
        model_dir = project_root / "models"
        
        # Create timestamp for versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_dir = model_dir / f"version_{timestamp}"
        
        # Load and preprocess data
        data = load_and_preprocess_data(str(data_path))
        
        # Train and save model
        train_model(data, version_dir)
        
        # Update symlink to latest version
        latest_link = model_dir / "latest"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(version_dir)
        
        logger.info("Model retraining completed successfully")
        
    except Exception as e:
        logger.error(f"Error during model retraining: {e}")
        raise

if __name__ == "__main__":
    main() 