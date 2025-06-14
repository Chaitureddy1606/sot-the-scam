from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from typing import List, Optional

# Add parent directory to path to import from pipeline
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.preprocessing import preprocess_data
from pipeline.feature_engineering import FeatureExtractor
from pipeline.model import FraudDetector
from pipeline.explainability import ModelExplainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Job Scam Detector API",
    description="API for detecting fraudulent job postings",
    version="1.0.0"
)

# Input models
class JobPosting(BaseModel):
    title: str
    company_profile: Optional[str] = ""
    description: str
    requirements: Optional[str] = ""
    salary_range: Optional[str] = ""

class BatchJobPostings(BaseModel):
    jobs: List[JobPosting]

# Output models
class PredictionResult(BaseModel):
    fraud_probability: float
    risk_level: str
    explanation: str

class BatchPredictionResult(BaseModel):
    results: List[PredictionResult]
    summary: dict

# Load model and feature extractor
try:
    model_path = Path(__file__).parent.parent / "models" / "fraud_detector.pkl"
    model = FraudDetector.load_model(model_path)
    feature_extractor = FeatureExtractor()
    logger.info("Model and feature extractor loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise RuntimeError("Failed to load model")

def get_risk_level(probability: float) -> str:
    """Determine risk level based on fraud probability."""
    if probability > 0.7:
        return "High"
    elif probability > 0.3:
        return "Medium"
    return "Low"

@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {
        "name": "Job Scam Detector API",
        "version": "1.0.0",
        "status": "active"
    }

@app.post("/predict", response_model=PredictionResult)
async def predict(job: JobPosting):
    """
    Predict fraud probability for a single job posting.
    """
    try:
        # Convert to DataFrame
        data = pd.DataFrame([job.dict()])
        
        # Preprocess and extract features
        processed_data = preprocess_data(data)
        features = feature_extractor.transform(processed_data)
        
        # Get prediction
        fraud_prob = model.predict_proba(features)[0][1]
        risk_level = get_risk_level(fraud_prob)
        
        # Generate explanation
        explainer = ModelExplainer(model, feature_extractor.get_feature_names())
        explanation = explainer.generate_explanation_text(features)
        
        return PredictionResult(
            fraud_probability=float(fraud_prob),
            risk_level=risk_level,
            explanation=explanation
        )
    
    except Exception as e:
        logger.error(f"Error processing prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResult)
async def predict_batch(batch: BatchJobPostings):
    """
    Predict fraud probability for multiple job postings.
    """
    try:
        # Convert to DataFrame
        data = pd.DataFrame([job.dict() for job in batch.jobs])
        
        # Preprocess and extract features
        processed_data = preprocess_data(data)
        features = feature_extractor.transform(processed_data)
        
        # Get predictions
        fraud_probs = model.predict_proba(features)[:, 1]
        
        # Generate results
        results = []
        for i, prob in enumerate(fraud_probs):
            risk_level = get_risk_level(prob)
            
            # Generate explanation for each prediction
            explainer = ModelExplainer(model, feature_extractor.get_feature_names())
            explanation = explainer.generate_explanation_text(features[i:i+1])
            
            results.append(PredictionResult(
                fraud_probability=float(prob),
                risk_level=risk_level,
                explanation=explanation
            ))
        
        # Generate summary
        risk_levels = [r.risk_level for r in results]
        summary = {
            "total_analyzed": len(results),
            "high_risk": sum(1 for r in risk_levels if r == "High"),
            "medium_risk": sum(1 for r in risk_levels if r == "Medium"),
            "low_risk": sum(1 for r in risk_levels if r == "Low"),
        }
        
        return BatchPredictionResult(results=results, summary=summary)
    
    except Exception as e:
        logger.error(f"Error processing batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"} 