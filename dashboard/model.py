import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy

class JobScamDetector:
    """Simple job scam detector using rule-based and NLP approaches."""
    
    def __init__(self):
        """Initialize the detector with necessary resources."""
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        # Load spaCy model
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            spacy.cli.download('en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
        
        # Initialize risk patterns
        self.risk_patterns = {
            'urgency': [
                r'urgent',
                r'immediate',
                r'start asap',
                r'quick money',
                r'fast cash'
            ],
            'payment': [
                r'\$\d+[kK]?\s*(?:per|/|\-)\s*(?:hour|hr|day|week|month)',
                r'registration fee',
                r'training fee',
                r'certification fee',
                r'investment required'
            ],
            'contact': [
                r'whatsapp',
                r'telegram',
                r'\+\d{10,}',
                r'contact.*urgently',
                r'dm for details'
            ],
            'suspicious_terms': [
                r'work from home',
                r'no experience',
                r'earn \$\d+k?\+?',
                r'weekly pay',
                r'daily pay'
            ]
        }
    
    def analyze_posting(self, title="", description="", location="", company_profile=""):
        """Analyze a job posting for potential scam indicators."""
        try:
            # Combine all text fields
            text = f"{title} {description} {location} {company_profile}".lower()
            
            # Initialize result
            result = {
                'probability': 0.0,
                'risk_factors': [],
                'explanation': []
            }
            
            # Check for risk patterns
            risk_score = 0
            total_patterns = 0
            
            for category, patterns in self.risk_patterns.items():
                for pattern in patterns:
                    total_patterns += 1
                    if re.search(pattern, text, re.IGNORECASE):
                        risk_score += 1
                        result['risk_factors'].append(f"Found suspicious {category} pattern: {pattern}")
            
            # Add NLP-based analysis
            doc = self.nlp(text[:1000000])  # Limit text length to avoid memory issues
            
            # Check for unusual email domains
            emails = re.findall(r'[\w\.-]+@[\w\.-]+', text)
            suspicious_domains = ['gmail', 'yahoo', 'hotmail', 'outlook']
            for email in emails:
                domain = email.split('@')[1].lower()
                if any(d in domain for d in suspicious_domains):
                    result['risk_factors'].append("Using personal email domain for business communication")
                    risk_score += 0.5
            
            # Check for urgency in language
            urgency_words = ['urgent', 'immediate', 'asap', 'quick', 'fast', 'hurry']
            if any(word.lower_ in urgency_words for word in doc):
                result['risk_factors'].append("Uses urgent language")
                risk_score += 0.5
            
            # Check for vague job descriptions
            tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
            if len(tokens) < 50:
                result['risk_factors'].append("Very short or vague job description")
                risk_score += 0.5
            
            # Calculate final probability
            result['probability'] = min(1.0, risk_score / (total_patterns + 3))
            
            # Add explanations
            if result['probability'] > 0.5:
                result['explanation'] = [
                    "High risk indicators detected",
                    f"Found {len(result['risk_factors'])} suspicious patterns",
                    "Consider verifying the company and job details"
                ]
            else:
                result['explanation'] = [
                    "Low risk indicators",
                    "Job posting appears legitimate",
                    "Always verify company information"
                ]
            
            return result
            
        except Exception as e:
            print(f"Error in analyze_posting: {str(e)}")
            return {
                'probability': 0.5,
                'risk_factors': ["Error analyzing job posting"],
                'explanation': ["Could not complete analysis", str(e)]
            }

# Initialize global model instance
model = JobScamDetector() 