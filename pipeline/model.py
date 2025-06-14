import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_curve, roc_curve, auc,
    f1_score, precision_score, recall_score
)
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import json
import logging
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JobScamDetector:
    """Job scam detection model with training, evaluation, and prediction capabilities."""
    
    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        model_path: str = "models/fraud_detector.pkl",
        metrics_path: str = "models/metrics.json"
    ):
        """
        Initialize the model with default or custom parameters.
        
        Args:
            model_params: Custom model parameters (optional)
            model_path: Path to save/load the model
            metrics_path: Path to save model metrics
        """
        self.model_path = model_path
        self.metrics_path = metrics_path
        
        # Default XGBoost parameters optimized for fraud detection
        self.default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 100,
            'use_label_encoder': False,
            'random_state': 42
        }
        
        self.model_params = model_params if model_params else self.default_params
        self.model = None
        self.feature_importance = None
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42)
        
    def prepare_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training with proper train-test split.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of dataset to include in the test split
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Preparing data for training...")
        
        # Calculate class weights for imbalanced dataset
        self.scale_pos_weight = sum(y == 0) / sum(y == 1)
        self.model_params['scale_pos_weight'] = self.scale_pos_weight
        
        # Adjust test_size for small datasets
        n_samples = len(y)
        min_test_samples = 2  # Minimum samples needed per class in test set
        n_classes = len(np.unique(y))
        min_test_size = (min_test_samples * n_classes) / n_samples
        
        if test_size < min_test_size:
            logger.warning(f"Adjusting test_size from {test_size} to {min_test_size} due to small dataset size")
            test_size = min_test_size
        
        # Split data
        return train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=self.model_params['random_state']
        )
    
    def tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X_train: Training feature matrix
            y_train: Training target vector
            
        Returns:
            Best parameters found
        """
        logger.info("Tuning hyperparameters...")
        
        param_grid = {
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [100, 200],
            'min_child_weight': [1, 3],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
        
        grid_search = GridSearchCV(
            xgb.XGBClassifier(**self.model_params),
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        logger.info(f"Best parameters found: {grid_search.best_params_}")
        return grid_search.best_params_
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        tune_params: bool = True
    ) -> Dict[str, float]:
        """
        Train the model with optimization for imbalanced data.
        
        Args:
            X: Feature matrix
            y: Target labels
            tune_params: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Starting model training...")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(X, y)
        
        # Create pipeline with SMOTE for handling imbalance
        self.model = Pipeline([
            ('scaler', self.scaler),
            ('smote', self.smote),
            ('classifier', GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            ))
        ])
        
        # Train model
        logger.info("Training model...")
        self.model.fit(X_train, y_train)
        
        # Get feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.named_steps['classifier'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Evaluate model
        metrics = self._evaluate_model(X_test, y_test)
        
        # Save model and metrics
        self.save_model()
        self.save_metrics(metrics)
        
        return metrics
    
    def _evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance with focus on F1-Score.
        
        Args:
            X_test: Test feature matrix
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'f1_score': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        }
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Add additional metrics
        metrics.update({
            'specificity': tn / (tn + fp),
            'false_positive_rate': fp / (fp + tn),
            'false_negative_rate': fn / (fn + tp)
        })
        
        # Calculate ROC-AUC
        fpr, tpr, _ = roc_curve(y_test, self.model.predict_proba(X_test)[:, 1])
        metrics['roc_auc'] = auc(fpr, tpr)
        
        # Generate and save plots
        self._save_evaluation_plots(X_test, y_test, y_pred)
        
        logger.info(f"Model metrics: {metrics}")
        return metrics
    
    def _save_evaluation_plots(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_pred: np.ndarray
    ) -> None:
        """Save evaluation plots."""
        # Set style
        plt.style.use('default')
        
        # Create plots directory if it doesn't exist
        os.makedirs('models/plots', exist_ok=True)
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, self.model.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig('models/plots/roc_curve.png')
        plt.close()
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, self.model.predict_proba(X_test)[:, 1])
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower right")
        plt.savefig('models/plots/pr_curve.png')
        plt.close()
    
    def predict(
        self,
        X: pd.DataFrame,
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            threshold: Classification threshold
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            self.load_model()
        
        probabilities = self.model.predict_proba(X)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        return predictions, probabilities
    
    def save_model(self) -> None:
        """Save the trained model to disk."""
        if self.model is not None:
            joblib.dump(self.model, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self) -> None:
        """Load a trained model from disk."""
        self.model = joblib.load(self.model_path)
        logger.info(f"Model loaded from {self.model_path}")
    
    def save_metrics(self, metrics: Dict[str, float]) -> None:
        """Save evaluation metrics to disk."""
        with open(self.metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to {self.metrics_path}")

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation with focus on F1-Score.
        
        Args:
            X: Feature matrix
            y: Target labels
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary of cross-validation metrics
        """
        # Create pipeline for cross-validation
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('classifier', GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            ))
        ])
        
        # Calculate cross-validation scores
        cv_scores = {
            'f1': cross_val_score(pipeline, X, y, cv=cv, scoring='f1'),
            'precision': cross_val_score(pipeline, X, y, cv=cv, scoring='precision'),
            'recall': cross_val_score(pipeline, X, y, cv=cv, scoring='recall')
        }
        
        # Calculate mean and std for each metric
        cv_metrics = {}
        for metric, scores in cv_scores.items():
            cv_metrics[f'{metric}_mean'] = scores.mean()
            cv_metrics[f'{metric}_std'] = scores.std()
        
        return cv_metrics

def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    tune_params: bool = True
) -> JobScamDetector:
    """
    Convenience function to train and evaluate a model.
    
    Args:
        X: Feature matrix
        y: Target vector
        tune_params: Whether to perform hyperparameter tuning
        
    Returns:
        Trained JobScamDetector instance
    """
    detector = JobScamDetector()
    detector.train(X, y, tune_params=tune_params)
    return detector

if __name__ == "__main__":
    import argparse
    from pipeline.preprocessing import clean_text
    from pipeline.feature_engineering import JobFeatureExtractor

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train the job scam detection model')
    parser.add_argument('--data', type=str, required=True, help='Path to training data CSV file')
    parser.add_argument('--output', type=str, default='models/', help='Directory to save model files')
    args = parser.parse_args()

    # Load and preprocess data
    logger.info(f"Loading data from {args.data}")
    data = pd.read_csv(args.data)
    
    # Clean text fields
    logger.info("Cleaning text data...")
    data['description'] = data['description'].fillna("").apply(clean_text)
    if 'company_profile' in data.columns:
        data['company_profile'] = data['company_profile'].fillna("").apply(clean_text)
    
    # Extract features
    logger.info("Extracting features...")
    feature_extractor = JobFeatureExtractor()
    features = feature_extractor.engineer_features(data, is_training=True)
    
    # Add fraud labels (for demo data, mark jobs with suspicious keywords as fraud)
    if 'fraud' not in data.columns:
        logger.info("Creating fraud labels for demo data...")
        suspicious_patterns = [
            'urgent', 'guarantee', 'no experience', 'work from home',
            'western union', 'bitcoin', 'registration fee', 'processing fee'
        ]
        data['fraud'] = data['description'].str.lower().apply(
            lambda x: any(pattern in x for pattern in suspicious_patterns)
        ).astype(int)
    
    # Train model
    logger.info("Training model...")
    detector = train_and_evaluate(features, data['fraud'], tune_params=True)
    
    logger.info("Training completed successfully!") 