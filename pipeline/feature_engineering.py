"""Feature engineering module for job scam detection."""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from typing import Tuple, Dict, Any, List, Union
import re
import joblib
import os
import logging
from .preprocessing import clean_text, preprocess_job_listing

logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self):
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.fitted = False
    
    def extract_text_features(self, texts: pd.Series) -> np.ndarray:
        """
        Extract TF-IDF features from text data.
        
        Args:
            texts: Series of text data
            
        Returns:
            np.ndarray: TF-IDF features
        """
        if not self.fitted:
            return self.tfidf.fit_transform(texts).toarray()
        return self.tfidf.transform(texts).toarray()
    
    def extract_embeddings(self, texts: pd.Series) -> np.ndarray:
        """
        Extract sentence embeddings using transformer model.
        
        Args:
            texts: Series of text data
            
        Returns:
            np.ndarray: Sentence embeddings
        """
        return self.sentence_transformer.encode(
            texts.tolist(),
            show_progress_bar=True,
            batch_size=32
        )
    
    def extract_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract numerical features from the dataset.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with numerical features
        """
        features = pd.DataFrame()
        
        # Text length features
        text_columns = ['title', 'description', 'requirements', 'company_profile']
        for col in text_columns:
            if col in df.columns:
                features[f'{col}_length'] = df[col].str.len()
                features[f'{col}_word_count'] = df[col].str.split().str.len()
        
        # Salary features (if available)
        if 'salary_range' in df.columns:
            features['has_salary'] = df['salary_range'].notna().astype(int)
        
        return features
    
    def combine_features(self, tfidf_features: np.ndarray, 
                        embeddings: np.ndarray, 
                        numerical_features: pd.DataFrame) -> np.ndarray:
        """
        Combine all features into a single array.
        
        Args:
            tfidf_features: TF-IDF features array
            embeddings: Sentence embeddings array
            numerical_features: Numerical features dataframe
            
        Returns:
            np.ndarray: Combined features
        """
        return np.hstack([
            tfidf_features,
            embeddings,
            numerical_features.values
        ])
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform the feature extractor on training data.
        
        Args:
            df: Input dataframe
            
        Returns:
            np.ndarray: Combined features
        """
        # Extract features
        text_data = df['description'].fillna('')  # Adjust column name if needed
        tfidf_features = self.extract_text_features(text_data)
        embeddings = self.extract_embeddings(text_data)
        numerical_features = self.extract_numerical_features(df)
        
        self.fitted = True
        return self.combine_features(tfidf_features, embeddings, numerical_features)
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted feature extractor.
        
        Args:
            df: Input dataframe
            
        Returns:
            np.ndarray: Combined features
        """
        if not self.fitted:
            raise ValueError("FeatureExtractor must be fitted before transform")
        
        # Extract features
        text_data = df['description'].fillna('')  # Adjust column name if needed
        tfidf_features = self.extract_text_features(text_data)
        embeddings = self.extract_embeddings(text_data)
        numerical_features = self.extract_numerical_features(df)
        
        return self.combine_features(tfidf_features, embeddings, numerical_features)

class JobFeatureExtractor:
    def __init__(self):
        """Initialize the feature extractor."""
        self.tfidf = TfidfVectorizer(
            max_features=300,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.fitted = False
        
    def extract_text_features(self, text: Union[str, pd.Series]) -> Dict[str, float]:
        """
        Extract basic text features.
        
        Args:
            text: Input text or pandas Series
            
        Returns:
            Dictionary of text features
        """
        # If input is a pandas Series, process each text individually
        if isinstance(text, pd.Series):
            return text.apply(self._extract_text_features_single)
        else:
            return self._extract_text_features_single(text)
    
    def _extract_text_features_single(self, text: str) -> Dict[str, float]:
        """Extract features from a single text string."""
        if not text or pd.isna(text):
            return {
                'text_length': 0,
                'word_count': 0,
                'avg_word_length': 0,
                'special_char_count': 0
            }
            
        # Clean text
        cleaned_text = clean_text(str(text))
        words = cleaned_text.split()
        
        # Calculate features
        features = {
            'text_length': len(cleaned_text),
            'word_count': len(words),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'special_char_count': len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', str(text)))
        }
        
        return features
    
    def extract_red_flags(self, text: Union[str, pd.Series]) -> Dict[str, int]:
        """
        Extract features related to common scam indicators.
        
        Args:
            text: Input text or pandas Series
            
        Returns:
            Dictionary of red flag features
        """
        # If input is a pandas Series, process each text individually
        if isinstance(text, pd.Series):
            return text.apply(self._extract_red_flags_single)
        else:
            return self._extract_red_flags_single(text)
    
    def _extract_red_flags_single(self, text: str) -> Dict[str, int]:
        """Extract red flags from a single text string."""
        if not text or pd.isna(text):
            return {
                'urgency_count': 0,
                'payment_count': 0,
                'suspicious_count': 0,
                'personal_info_count': 0,
                'pressure_count': 0,
                'total_red_flags': 0
            }
            
        text = str(text).lower()
        
        red_flags = {
            'urgency': ['urgent', 'immediate', 'asap', 'quick', 'fast'],
            'payment': ['payment', 'salary', 'money', 'cash', 'paid', 'fee'],
            'suspicious': ['guarantee', 'work from home', 'no experience', 'easy'],
            'personal_info': ['bank', 'account', 'ssn', 'social security', 'passport'],
            'pressure': ['limited time', 'act now', 'don\'t wait', 'hurry']
        }
        
        features = {}
        for category, terms in red_flags.items():
            count = sum(1 for term in terms if term in text)
            features[f'{category}_count'] = count
            
        features['total_red_flags'] = sum(features.values())
        return features
    
    def engineer_features(
        self,
        df: pd.DataFrame,
        is_training: bool = True
    ) -> pd.DataFrame:
        """
        Engineer all features for job listings.
        
        Args:
            df: DataFrame containing job listings
            is_training: Whether this is training data
            
        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame()
        
        # Process text columns
        text_columns = ['title', 'description', 'company_profile']
        for col in text_columns:
            if col in df.columns:
                # Extract basic text features
                col_features = df[col].apply(self.extract_text_features)
                col_features = pd.DataFrame(col_features.tolist())
                col_features = col_features.add_prefix(f'{col}_')
                features = pd.concat([features, col_features], axis=1)
                
                # Extract red flags
                red_flags = df[col].apply(self.extract_red_flags)
                red_flags = pd.DataFrame(red_flags.tolist())
                red_flags = red_flags.add_prefix(f'{col}_')
                features = pd.concat([features, red_flags], axis=1)
                
                # TF-IDF features
                if is_training:
                    tfidf_features = self.tfidf.fit_transform(df[col].fillna(''))
                    self.fitted = True
                else:
                    if not self.fitted:
                        raise ValueError("TfidfVectorizer not fitted. Call with is_training=True first.")
                    tfidf_features = self.tfidf.transform(df[col].fillna(''))
                
                tfidf_df = pd.DataFrame(
                    tfidf_features.toarray(),
                    columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
                )
                features = pd.concat([features, tfidf_df], axis=1)
        
        return features 