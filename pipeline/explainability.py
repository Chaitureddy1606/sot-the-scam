import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
import logging
import os
from typing import Dict, List, Tuple, Optional
from pipeline.model import JobScamDetector
from pipeline.feature_engineering import JobFeatureExtractor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JobScamExplainer:
    """Generate explanations for job scam predictions using SHAP."""
    
    def __init__(self):
        """Initialize the SHAP explainer."""
        self.explainer = None
        self.feature_names = None
        
    def fit(self, model, X_train: pd.DataFrame):
        """
        Fit the SHAP explainer.
        
        Args:
            model: Trained model
            X_train: Training data
        """
        # Get the actual model from the pipeline
        if hasattr(model, 'named_steps'):
            clf = model.named_steps['classifier']
        else:
            clf = model
            
        # Initialize TreeExplainer for tree-based models
        self.explainer = shap.TreeExplainer(clf)
        self.feature_names = X_train.columns.tolist()
        
    def explain_prediction(self, instance: pd.Series) -> Dict[str, float]:
        """
        Explain a single prediction.
        
        Args:
            instance: Single instance to explain
            
        Returns:
            Dictionary mapping features to their importance values
        """
        if self.explainer is None or self.feature_names is None:
            logger.warning("Explainer not fitted. Returning empty explanation.")
            return {}
            
        try:
            # Calculate SHAP values for the instance
            shap_values = self.explainer.shap_values(instance)
            
            # For binary classification, we might get a list of arrays
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Take the positive class
                
            # Create a dictionary of feature importances
            feature_importance = dict(zip(self.feature_names, shap_values))
            
            # Sort by absolute importance and take top features
            sorted_importance = {k: v for k, v in sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )}
            
            return sorted_importance
            
        except Exception as e:
            logger.error(f"Error explaining prediction: {str(e)}")
            return {}
    
    def explain_model(self, X: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Generate global SHAP explanations for the model.
        
        Args:
            X: Dataset to explain
            
        Returns:
            Dictionary containing global feature importance values
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
            
        # Calculate SHAP values for all instances
        shap_values = self.explainer.shap_values(X)
        
        # For binary classification, shap_values might be a list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Get values for positive class
            
        # Calculate mean absolute SHAP values for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create dictionary of feature importances
        global_importance = {}
        for feature_name, importance in zip(self.feature_names, mean_abs_shap):
            global_importance[feature_name] = float(importance)
            
        return global_importance

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate SHAP explanations for job scam predictions')
    parser.add_argument('--data', type=str, required=True, help='Path to job listings CSV')
    parser.add_argument('--output', type=str, default='models/plots', help='Directory to save plots')
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    data = pd.read_csv(args.data)
    
    # Initialize explainer
    explainer = JobScamExplainer()
    
    # Generate global explanations
    logger.info("Generating global SHAP explanations...")
    global_importance = explainer.explain_model(data)
    
    # Generate individual explanations for each listing
    logger.info("Generating individual explanations...")
    for idx, row in data.iterrows():
        output_path = f"{args.output}/shap_explanation_{idx}.png"
        contributions = explainer.explain_prediction(row)
        
        # Log top contributing features
        logger.info(f"\nTop features for: {row['title']}")
        for feature, value in list(contributions.items())[:5]:
            logger.info(f"{feature}: {value:.3f}")

    # Plot global importance
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap.TreeExplainer(data.iloc[:, :-1].values).shap_values(data.iloc[:, :-1].values),
        data.iloc[:, :-1].values,
        max_display=20,
        show=False
    )
    plt.tight_layout()
    plt.savefig(f"{args.output}/shap_global_importance.png")
    plt.close()

    # Log global importance
    logger.info("\nGlobal SHAP Importance:")
    for feature, importance in global_importance.items():
        logger.info(f"{feature}: {importance:.3f}") 