"""Data loader for the Employment Scam Aegean Dataset (EMSCAD)."""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class EMSCADLoader:
    """Loader for the Employment Scam Aegean Dataset."""
    
    def __init__(self):
        """Initialize the data loader."""
        self.categorical_features = [
            'employment_type',
            'required_experience',
            'required_education',
            'industry',
            'function',
            'department'
        ]
        
        self.binary_features = [
            'telecommuting',
            'has_company_logo',
            'has_questions'
        ]
        
        self.text_features = [
            'title',
            'company_profile',
            'description',
            'requirements',
            'benefits'
        ]
        
        self.location_feature = 'location'
        self.salary_feature = 'salary_range'
        self.target = 'fraudulent'
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load the EMSCAD dataset.
        
        Args:
            file_path: Path to the dataset CSV file
            
        Returns:
            Loaded DataFrame
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded dataset with {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
            
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the dataset.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()
        
        # Handle missing values
        for feature in self.text_features:
            df[feature] = df[feature].fillna('')
            
        for feature in self.categorical_features:
            df[feature] = df[feature].fillna('Unknown')
            
        # Convert binary features to int
        for feature in self.binary_features:
            df[feature] = df[feature].fillna(0).astype(int)
            
        # Process salary range
        df['has_salary'] = df[self.salary_feature].notna().astype(int)
        
        # Process location
        df['is_remote'] = df[self.location_feature].str.contains(
            'remote|Remote|REMOTE',
            na=False
        ).astype(int)
        
        # Create additional features
        df['text_length'] = df['description'].str.len()
        df['has_benefits'] = df['benefits'].notna().astype(int)
        df['requirements_length'] = df['requirements'].str.len()
        
        logger.info("Data preprocessing completed")
        return df
        
    def split_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets.
        
        Args:
            df: Preprocessed DataFrame
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            Train and test DataFrames
        """
        # Stratify by target to maintain class distribution
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[self.target]
        )
        
        logger.info(f"Split data into {len(train_df)} train and {len(test_df)} test samples")
        
        # Log class distribution
        train_dist = train_df[self.target].value_counts(normalize=True)
        test_dist = test_df[self.target].value_counts(normalize=True)
        logger.info(f"Train set class distribution:\n{train_dist}")
        logger.info(f"Test set class distribution:\n{test_dist}")
        
        return train_df, test_df
        
    def get_feature_groups(self) -> Dict[str, list]:
        """
        Get features grouped by type.
        
        Returns:
            Dictionary of feature groups
        """
        return {
            'categorical': self.categorical_features,
            'binary': self.binary_features,
            'text': self.text_features,
            'location': [self.location_feature],
            'salary': [self.salary_feature],
            'target': [self.target]
        }
        
    def get_dataset_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            'total_samples': len(df),
            'fraudulent_samples': int(df[self.target].sum()),
            'genuine_samples': int(len(df) - df[self.target].sum()),
            'fraud_ratio': float(df[self.target].mean()),
            'missing_values': df.isnull().sum().to_dict(),
            'categorical_counts': {
                feature: df[feature].nunique()
                for feature in self.categorical_features
            }
        }
        
        logger.info(f"Dataset statistics:\n{stats}")
        return stats 