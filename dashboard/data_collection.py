"""Data collection module for model training."""

import pandas as pd
import sqlite3
from typing import Dict, List, Tuple
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def collect_training_data(db_path: str) -> pd.DataFrame:
    """
    Collect training data from user feedback and analysis history.
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        DataFrame with training data
    """
    conn = None
    try:
        # Validate database path
        if not Path(db_path).exists():
            raise FileNotFoundError(f"Database file not found at {db_path}")
            
        conn = sqlite3.connect(db_path)
        
        # Get analysis history with user feedback
        query = """
        SELECT 
            h.title,
            h.description,
            h.location,
            h.company_profile,
            h.prediction,
            h.confidence,
            f.feedback_type,
            f.rating,
            f.comments,
            h.timestamp
        FROM analysis_history h
        LEFT JOIN user_feedback f ON h.id = f.analysis_id
        WHERE f.feedback_type IS NOT NULL
        """
        
        df = pd.read_sql_query(query, conn)
        
        # Check if we have any data
        if len(df) == 0:
            logger.warning("No training data found with user feedback")
            return pd.DataFrame()
            
        # Validate required columns
        required_columns = ['title', 'description', 'prediction', 'feedback_type']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Handle missing values
        df['description'] = df['description'].fillna("")
        df['location'] = df['location'].fillna("Unknown")
        df['company_profile'] = df['company_profile'].fillna("")
        
        # Convert feedback to binary labels
        df['is_scam'] = df['prediction'].apply(lambda x: 1 if x == 'Scam' else 0)
        df['feedback_correct'] = df['feedback_type'].apply(lambda x: 1 if x == '✅ Correct' else 0)
        
        # Create final label based on feedback
        df['is_scam'] = df.apply(
            lambda row: row['is_scam'] if row['feedback_correct'] else 1 - row['is_scam'],
            axis=1
        )
        
        # Log class distribution
        scam_count = df['is_scam'].sum()
        total_count = len(df)
        logger.info(f"Collected {total_count} samples: {scam_count} scam, {total_count - scam_count} legitimate")
        
        # Select relevant columns for training
        training_data = df[[
            'title',
            'description',
            'location',
            'company_profile',
            'is_scam'
        ]].copy()
        
        return training_data
        
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error collecting training data: {e}")
        raise
    finally:
        if conn:
            conn.close()

def save_training_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save collected training data to CSV file.
    
    Args:
        df: DataFrame with training data
        output_path: Path to save CSV file
    """
    try:
        # Validate input DataFrame
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
            
        if len(df) == 0:
            logger.warning("Empty DataFrame provided, no data to save")
            return
            
        # Create directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Training data saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving training data: {e}")
        raise 