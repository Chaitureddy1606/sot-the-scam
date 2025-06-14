"""Script to train and evaluate the job scam detection model."""

import logging
import pandas as pd
from pipeline.data_loader import EMSCADLoader
from pipeline.model import JobScamDetector
from pipeline.feature_engineering import JobFeatureExtractor
from pipeline.explainability import JobScamExplainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main training function."""
    # Initialize components
    data_loader = EMSCADLoader()
    feature_extractor = JobFeatureExtractor()
    model = JobScamDetector()
    explainer = JobScamExplainer()
    
    # Load and preprocess data
    logger.info("Loading dataset...")
    df = data_loader.load_data('data/fake_job_postings.csv')
    
    # Get dataset statistics
    stats = data_loader.get_dataset_stats(df)
    logger.info(f"Dataset statistics:\n{stats}")
    
    # Preprocess data
    logger.info("Preprocessing data...")
    df_processed = data_loader.preprocess_data(df)
    
    # Split data
    train_df, test_df = data_loader.split_data(df_processed)
    
    # Extract features
    logger.info("Extracting features...")
    X_train = feature_extractor.engineer_features(train_df, is_training=True)
    X_test = feature_extractor.engineer_features(test_df, is_training=False)
    
    y_train = train_df['fraudulent']
    y_test = test_df['fraudulent']
    
    # Train model
    logger.info("Training model...")
    metrics = model.train(X_train, y_train)
    logger.info(f"Training metrics:\n{metrics}")
    
    # Evaluate on test set
    logger.info("Evaluating model...")
    test_metrics = model._evaluate_model(X_test, y_test)
    logger.info(f"Test metrics:\n{test_metrics}")
    
    # Cross-validation
    logger.info("Performing cross-validation...")
    cv_metrics = model.cross_validate(X_train, y_train)
    logger.info(f"Cross-validation metrics:\n{cv_metrics}")
    
    # Fit explainer
    logger.info("Fitting SHAP explainer...")
    explainer.fit(model.model, X_train)
    
    # Save model
    logger.info("Saving model...")
    model.save_model()
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main() 