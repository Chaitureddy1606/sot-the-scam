import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_explore_data():
    """Load and perform initial exploration of the EMSCAD dataset."""
    try:
        # Load the dataset
        logger.info("Loading the EMSCAD dataset...")
        df = pd.read_csv("data/fake_job_postings.csv")
        
        # Basic dataset information
        logger.info("\nDataset Overview:")
        logger.info(f"Total number of job postings: {len(df)}")
        logger.info(f"Number of features: {len(df.columns)}")
        
        # Check class distribution
        fraudulent_count = df['fraudulent'].sum()
        logger.info("\nClass Distribution:")
        logger.info(f"Legitimate postings: {len(df) - fraudulent_count}")
        logger.info(f"Fraudulent postings: {fraudulent_count}")
        logger.info(f"Fraud rate: {(fraudulent_count/len(df))*100:.2f}%")
        
        # Check missing values
        missing_values = df.isnull().sum()
        logger.info("\nMissing Values:")
        for column, missing in missing_values[missing_values > 0].items():
            logger.info(f"{column}: {missing} ({(missing/len(df))*100:.2f}%)")
        
        # Display sample of features
        logger.info("\nFeature Names:")
        logger.info(", ".join(df.columns))
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

if __name__ == "__main__":
    df = load_and_explore_data() 