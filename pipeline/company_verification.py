"""Company verification module for enhanced scam detection."""

import re
import requests
from typing import Dict, Optional, Tuple
import whois
import dns.resolver
import cv2
import numpy as np
from PIL import Image
import io
import logging
from bs4 import BeautifulSoup
import validators
from urllib.parse import urlparse
import pandas as pd

logger = logging.getLogger(__name__)

class CompanyVerifier:
    """Advanced company verification system."""
    
    def __init__(self):
        """Initialize the company verifier."""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
    def verify_domain(self, email: str) -> Dict[str, any]:
        """
        Verify email domain existence and registration details.
        
        Args:
            email: Email address to verify
            
        Returns:
            Dictionary containing domain verification results
        """
        try:
            # Extract domain from email
            domain = email.split('@')[-1] if '@' in email else email
            
            # Check if domain is valid
            if not validators.domain(domain):
                return {
                    'is_valid': False,
                    'reason': 'Invalid domain format',
                    'domain_age': None,
                    'has_mx_record': False
                }
            
            # Get domain registration info
            domain_info = whois.whois(domain)
            
            # Check MX records
            try:
                mx_records = dns.resolver.resolve(domain, 'MX')
                has_mx = len(mx_records) > 0
            except:
                has_mx = False
            
            # Calculate domain age
            creation_date = domain_info.creation_date
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
            
            domain_age = None
            if creation_date:
                from datetime import datetime
                domain_age = (datetime.now() - creation_date).days
            
            return {
                'is_valid': True,
                'domain_age': domain_age,
                'has_mx_record': has_mx,
                'registrar': domain_info.registrar,
                'creation_date': creation_date,
                'expiration_date': domain_info.expiration_date
            }
            
        except Exception as e:
            logger.error(f"Error verifying domain: {str(e)}")
            return {
                'is_valid': False,
                'reason': f"Verification failed: {str(e)}",
                'domain_age': None,
                'has_mx_record': False
            }
    
    def analyze_logo(self, logo_url: str) -> Dict[str, any]:
        """
        Analyze company logo for potential red flags.
        
        Args:
            logo_url: URL of the company logo
            
        Returns:
            Dictionary containing logo analysis results
        """
        try:
            # Download image
            response = requests.get(logo_url)
            img = Image.open(io.BytesIO(response.content))
            
            # Convert to OpenCV format
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Basic image analysis
            height, width = img_cv.shape[:2]
            aspect_ratio = width / height
            
            # Check image quality
            blur_score = cv2.Laplacian(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            
            # Check if image is too small or too large
            size_suspicious = (width < 50 or height < 50) or (width > 2000 or height > 2000)
            
            # Check if aspect ratio is unusual
            ratio_suspicious = aspect_ratio < 0.5 or aspect_ratio > 2.0
            
            # Check if image is too blurry
            quality_suspicious = blur_score < 100
            
            return {
                'width': width,
                'height': height,
                'aspect_ratio': aspect_ratio,
                'blur_score': blur_score,
                'size_suspicious': size_suspicious,
                'ratio_suspicious': ratio_suspicious,
                'quality_suspicious': quality_suspicious,
                'overall_suspicious': size_suspicious or ratio_suspicious or quality_suspicious
            }
            
        except Exception as e:
            logger.error(f"Error analyzing logo: {str(e)}")
            return {
                'error': str(e),
                'overall_suspicious': True
            }
    
    def verify_company_registration(self, company_name: str, country: str = 'US') -> Dict[str, any]:
        """
        Check company registration status using available databases.
        
        Args:
            company_name: Name of the company
            country: Country code for registration check
            
        Returns:
            Dictionary containing registration verification results
        """
        try:
            # This is a placeholder for actual company registration verification
            # In a production system, you would integrate with:
            # - US: SEC EDGAR database
            # - UK: Companies House
            # - Other country-specific company registries
            
            # For now, we'll do a basic web search
            search_url = f"https://www.google.com/search?q={company_name}+company+registration"
            response = requests.get(search_url, headers=self.headers)
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Count relevant results
            results_count = len(soup.find_all('div', class_='g'))
            
            return {
                'found_online': results_count > 0,
                'results_count': results_count,
                'verification_source': 'web_search',
                'verification_level': 'basic'
            }
            
        except Exception as e:
            logger.error(f"Error verifying company registration: {str(e)}")
            return {
                'error': str(e),
                'found_online': False,
                'verification_level': 'failed'
            }
    
    def scrape_company_info(self, company_website: str) -> Dict[str, any]:
        """
        Scrape and analyze company website for verification.
        
        Args:
            company_website: URL of the company website
            
        Returns:
            Dictionary containing scraped information and analysis
        """
        try:
            # Validate URL
            if not validators.url(company_website):
                return {
                    'is_valid': False,
                    'reason': 'Invalid URL format'
                }
            
            # Get website content
            response = requests.get(company_website, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract key elements
            contact_info = bool(re.search(r'contact|email|phone|address', response.text, re.I))
            about_page = bool(re.search(r'about|company|team', response.text, re.I))
            social_links = bool(soup.find_all('a', href=re.compile(r'linkedin|facebook|twitter')))
            privacy_policy = bool(re.search(r'privacy|policy|terms', response.text, re.I))
            
            # Check SSL certificate
            parsed_url = urlparse(company_website)
            has_ssl = parsed_url.scheme == 'https'
            
            # Analyze content
            text_content = soup.get_text()
            word_count = len(text_content.split())
            
            return {
                'is_valid': True,
                'has_contact_info': contact_info,
                'has_about_page': about_page,
                'has_social_links': social_links,
                'has_privacy_policy': privacy_policy,
                'has_ssl': has_ssl,
                'content_length': word_count,
                'suspicious_score': sum([
                    not contact_info,
                    not about_page,
                    not social_links,
                    not privacy_policy,
                    not has_ssl,
                    word_count < 100
                ]) / 6.0
            }
            
        except Exception as e:
            logger.error(f"Error scraping company website: {str(e)}")
            return {
                'is_valid': False,
                'reason': f"Scraping failed: {str(e)}",
                'suspicious_score': 1.0
            }
    
    def verify_company(self, 
                      company_name: str,
                      email: Optional[str] = None,
                      website: Optional[str] = None,
                      logo_url: Optional[str] = None,
                      country: str = 'US') -> Dict[str, any]:
        """
        Perform comprehensive company verification.
        
        Args:
            company_name: Name of the company
            email: Company email address
            website: Company website URL
            logo_url: URL of company logo
            country: Country code for registration check
            
        Returns:
            Dictionary containing all verification results
        """
        results = {
            'company_name': company_name,
            'verification_timestamp': pd.Timestamp.now()
        }
        
        # Verify domain if email provided
        if email:
            results['domain_verification'] = self.verify_domain(email)
        
        # Analyze logo if URL provided
        if logo_url:
            results['logo_analysis'] = self.analyze_logo(logo_url)
        
        # Check company registration
        results['registration'] = self.verify_company_registration(company_name, country)
        
        # Scrape website if provided
        if website:
            results['website_analysis'] = self.scrape_company_info(website)
        
        # Calculate overall suspicion score
        suspicion_factors = []
        
        if email and results.get('domain_verification'):
            domain_age = results['domain_verification'].get('domain_age', 0) or 0
            suspicion_factors.append(1.0 if domain_age < 90 else 0.0)  # Domain less than 90 days old
            
        if logo_url and results.get('logo_analysis'):
            suspicion_factors.append(
                1.0 if results['logo_analysis'].get('overall_suspicious') else 0.0
            )
            
        if results.get('registration'):
            suspicion_factors.append(
                0.0 if results['registration'].get('found_online') else 1.0
            )
            
        if website and results.get('website_analysis'):
            suspicion_factors.append(
                results['website_analysis'].get('suspicious_score', 1.0)
            )
        
        results['overall_suspicion_score'] = (
            sum(suspicion_factors) / len(suspicion_factors)
            if suspicion_factors
            else None
        )
        
        return results 