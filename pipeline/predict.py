import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import joblib
import logging
from preprocessing import clean_text
from feature_engineering import JobFeatureExtractor
from model import JobScamDetector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_predict(
    input_file: str,
    model_path: str = "models/fraud_detector.pkl"
) -> pd.DataFrame:
    """
    Load job listings and make predictions using the trained model.
    
    Args:
        input_file: Path to CSV file containing job listings
        model_path: Path to trained model file
        
    Returns:
        DataFrame with original data and predictions
    """
    # Load data
    logger.info(f"Loading data from {input_file}")
    data = pd.read_csv(input_file)
    
    # Clean text fields
    logger.info("Cleaning text data...")
    data['description'] = data['description'].fillna("").apply(clean_text)
    if 'company_profile' in data.columns:
        data['company_profile'] = data['company_profile'].fillna("").apply(clean_text)
    
    # Extract features
    logger.info("Extracting features...")
    feature_extractor = JobFeatureExtractor()
    features = feature_extractor.engineer_features(data)
    
    # Load model and make predictions
    logger.info("Loading model and making predictions...")
    detector = JobScamDetector(model_path=model_path)
    detector.load_model()
    
    predictions, probabilities = detector.predict(features)
    
    # Add predictions to data
    data['predicted_fraud'] = predictions
    data['fraud_probability'] = probabilities
    
    # Sort by fraud probability for easier review
    data = data.sort_values('fraud_probability', ascending=False)
    
    return data

def format_predictions(predictions: pd.DataFrame) -> str:
    """Format predictions for display."""
    output = []
    for _, row in predictions.iterrows():
        fraud_prob = row['fraud_probability'] * 100
        status = "🚨 LIKELY FRAUD" if row['predicted_fraud'] else "✅ LEGITIMATE"
        
        output.append(f"\n{'='*80}")
        output.append(f"Title: {row['title']}")
        output.append(f"Location: {row['location']}")
        output.append(f"Company: {row['company_profile']}")
        output.append(f"\nPrediction: {status} (Confidence: {fraud_prob:.1f}%)")
        output.append(f"\nDescription: {row['description'][:200]}...")
        
    return "\n".join(output)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict fraud probability for job listings')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output', type=str, help='Path to save detailed predictions CSV')
    args = parser.parse_args()
    
    # Make predictions
    predictions = load_and_predict(args.input)
    
    # Display formatted results
    print("\nPrediction Results:")
    print(format_predictions(predictions))
    
    # Save detailed results if requested
    if args.output:
        predictions.to_csv(args.output, index=False)
        logger.info(f"Detailed predictions saved to {args.output}") 