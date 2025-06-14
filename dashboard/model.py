import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import re
import os

class JobScamDetector:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the trained model and vectorizer."""
        model_path = os.path.join(os.path.dirname(__file__), 'models')
        try:
            self.vectorizer = joblib.load(os.path.join(model_path, 'vectorizer.joblib'))
            self.model = joblib.load(os.path.join(model_path, 'model.joblib'))
        except FileNotFoundError:
            # If model doesn't exist, create a basic one
            self._create_basic_model()

    def _create_basic_model(self):
        """Create a basic model with common scam indicators."""
        # Initialize vectorizer and model
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        # Create basic training data
        scam_examples = [
            "Work from home opportunity! Earn $10,000/week guaranteed!",
            "No experience needed! Immediate start with high salary!",
            "Send personal bank details for direct deposit setup",
            "Investment required for training materials",
            "Urgent position! Send SSN and ID for immediate start"
        ]
        legitimate_examples = [
            "Software Engineer position at established company",
            "Marketing Manager role with competitive salary",
            "Administrative Assistant needed for local office",
            "Sales Representative position with base salary plus commission",
            "Project Manager role with 5 years experience required"
        ]

        # Combine examples and create labels
        X = scam_examples + legitimate_examples
        y = [1] * len(scam_examples) + [0] * len(legitimate_examples)

        # Fit vectorizer and model
        X_vectorized = self.vectorizer.fit_transform(X)
        self.model.fit(X_vectorized, y)

    def _preprocess_text(self, text):
        """Preprocess job posting text."""
        if not isinstance(text, str):
            return ""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        return text

    def _extract_risk_factors(self, text, probability):
        """Extract potential risk factors from the job posting."""
        risk_factors = []
        text_lower = text.lower()

        # Common scam indicators
        if probability > 0.5:
            indicators = {
                'urgent_hiring': (
                    any(word in text_lower for word in ['urgent', 'immediate start', 'apply now']),
                    "Urgent or immediate hiring mentioned"
                ),
                'high_salary': (
                    any(word in text_lower for word in ['$$$', 'guaranteed salary', 'huge salary']),
                    "Unrealistic salary promises"
                ),
                'no_experience': (
                    'no experience' in text_lower or 'no skills needed' in text_lower,
                    "No experience or skills required"
                ),
                'personal_info': (
                    any(word in text_lower for word in ['ssn', 'bank details', 'bank account']),
                    "Requests for personal/financial information"
                ),
                'investment': (
                    any(word in text_lower for word in ['investment', 'payment required', 'fees']),
                    "Requires upfront payment or investment"
                ),
                'poor_grammar': (
                    len(re.findall(r'[!]{2,}', text)) > 0 or len(re.findall(r'[?]{2,}', text)) > 0,
                    "Poor grammar or excessive punctuation"
                ),
                'vague_description': (
                    len(text.split()) < 50,
                    "Vague or very short job description"
                )
            }

            for check, (condition, message) in indicators.items():
                if condition:
                    risk_factors.append(message)

        return risk_factors

    def _calculate_verification_score(self, text, risk_factors):
        """Calculate a verification score based on various factors."""
        base_score = 1.0 - (len(risk_factors) * 0.1)  # Deduct 0.1 for each risk factor
        
        # Check for professional elements
        professional_indicators = [
            'experience required',
            'qualifications',
            'responsibilities',
            'benefits',
            'company culture',
            'job requirements'
        ]
        
        for indicator in professional_indicators:
            if indicator in text.lower():
                base_score += 0.05  # Add 0.05 for each professional indicator
        
        # Normalize score between 0 and 1
        return max(0.0, min(1.0, base_score))

    def analyze_posting(self, title, description, location=None, company_profile=None):
        """Analyze a single job posting."""
        # Combine all text fields
        combined_text = f"{title} {description}"
        if location:
            combined_text += f" {location}"
        if company_profile:
            combined_text += f" {company_profile}"

        # Preprocess text
        processed_text = self._preprocess_text(combined_text)
        
        # Vectorize
        X = self.vectorizer.transform([processed_text])
        
        # Get prediction and probability
        probability = self.model.predict_proba(X)[0][1]
        
        # Extract risk factors
        risk_factors = self._extract_risk_factors(combined_text, probability)
        
        # Calculate verification score
        verification_score = self._calculate_verification_score(combined_text, risk_factors)
        
        return {
            'probability': probability,
            'risk_factors': risk_factors,
            'verification_score': verification_score
        }

    def analyze_batch(self, df):
        """Analyze multiple job postings."""
        results = []
        for _, row in df.iterrows():
            analysis = self.analyze_posting(
                title=row['job_title'],
                description=row['job_description'],
                location=row.get('location'),
                company_profile=row.get('company_profile')
            )
            results.append(analysis)
        return results 