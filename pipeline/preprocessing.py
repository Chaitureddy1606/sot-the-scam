"""Text preprocessing module for job scam detection."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, List
import re
import string
import nltk
import spacy
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from typing import Union, Dict
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize spaCy
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the job posting dataset.
    
    Args:
        file_path: Path to the jobs_sample.csv file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    return pd.read_csv(file_path)

def clean_text(text: str) -> str:
    """
    Clean and normalize text data.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove phone numbers
    text = re.sub(r'\+?[\d\-\(\) ]{10,}', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def extract_entities(text: str) -> List[dict]:
    """
    Extract named entities from text using spaCy.
    
    Args:
        text: Input text
        
    Returns:
        List of dictionaries containing entity text and label
    """
    doc = nlp(text)
    entities = []
    
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'label': ent.label_,
            'start': ent.start_char,
            'end': ent.end_char
        })
    
    return entities

def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into words.
    
    Args:
        text: Input text
        
    Returns:
        List of tokens
    """
    return nltk.word_tokenize(text)

def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Remove stopwords from list of tokens.
    
    Args:
        tokens: List of word tokens
        
    Returns:
        List of tokens with stopwords removed
    """
    stopwords = set(nltk.corpus.stopwords.words('english'))
    return [token for token in tokens if token.lower() not in stopwords]

def extract_features(text: str) -> Dict[str, int]:
    """
    Extract additional features from text.
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Dictionary of extracted features
    """
    features = {
        'text_length': len(text),
        'word_count': len(text.split()),
        'has_email': 1 if re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text) else 0,
        'has_phone': 1 if re.search(r'\d{3}[-.]?\d{3}[-.]?\d{4}', text) else 0,
        'has_url': 1 if re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text) else 0,
        'has_money_symbol': 1 if re.search(r'[$€£¥]', text) else 0,
        'has_urgency_terms': 1 if any(word in text.lower() for word in ['urgent', 'immediate', 'asap', 'quick']) else 0
    }
    return features

def preprocess_job_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess job posting data.
    
    Args:
        df (pd.DataFrame): Input dataframe with job postings
        
    Returns:
        pd.DataFrame: Preprocessed dataframe with additional features
    """
    # Create a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # Clean text fields
    text_columns = ['title', 'description', 'company_profile']
    for col in text_columns:
        if col in df_processed.columns:
            df_processed[f'{col}_cleaned'] = df_processed[col].apply(clean_text)
    
    # Extract features from description
    if 'description' in df_processed.columns:
        features = df_processed['description'].apply(extract_features)
        feature_df = pd.DataFrame(features.tolist())
        df_processed = pd.concat([df_processed, feature_df], axis=1)
    
    # Handle location data
    if 'location' in df_processed.columns:
        df_processed['is_remote'] = df_processed['location'].str.lower().str.contains('remote|work from home|wfh').astype(int)
        df_processed['location_cleaned'] = df_processed['location'].fillna('Unknown').str.lower()
    
    return df_processed

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the job posting dataset.
    
    Args:
        df: Raw dataframe
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    df = df.copy()
    
    # Clean text columns (adjust column names based on your data)
    text_columns = ['title', 'description', 'requirements', 'company_profile']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)
    
    # Handle missing values
    df = df.fillna("")
    
    return df

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets.
    
    Args:
        df: Input dataframe
        test_size: Proportion of dataset to include in the test split
        random_state: Random state for reproducibility
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and testing dataframes
    """
    from sklearn.model_selection import train_test_split
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['is_fraud'] if 'is_fraud' in df.columns else None
    )
    
    return train_df, test_df

def preprocess_job_listing(
    title: Optional[str] = None,
    description: Optional[str] = None,
    company: Optional[str] = None
) -> dict:
    """
    Preprocess all text fields of a job listing.
    
    Args:
        title: Job title
        description: Job description
        company: Company description
        
    Returns:
        Dictionary containing processed text fields
    """
    processed = {}
    
    if title:
        processed['title'] = clean_text(title)
        processed['title_tokens'] = remove_stopwords(tokenize_text(processed['title']))
        processed['title_entities'] = extract_entities(title)
    
    if description:
        processed['description'] = clean_text(description)
        processed['description_tokens'] = remove_stopwords(tokenize_text(processed['description']))
        processed['description_entities'] = extract_entities(description)
    
    if company:
        processed['company'] = clean_text(company)
        processed['company_tokens'] = remove_stopwords(tokenize_text(processed['company']))
        processed['company_entities'] = extract_entities(company)
    
    return processed 