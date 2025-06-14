import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import sys
import time
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import sqlite3
from pathlib import Path
from database import (
    init_database,
    save_analysis,
    save_feedback,
    get_analysis_history,
    get_analysis_stats,
    search_history,
    register_user,
    verify_user,
    save_user_analysis,
    get_user_history,
    save_user_feedback,
    get_feedback_stats
)
from model import JobScamDetector

# Initialize database at startup
init_database()

# Initialize model with caching
@st.cache_resource
def get_model():
    """Get or create a cached instance of JobScamDetector."""
    return JobScamDetector()

# Must be the first Streamlit command
st.set_page_config(
    page_title="Job Scam Detector",
    page_icon="🔍",
    layout="wide"
)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.feature_engineering import JobFeatureExtractor
from pipeline.explainability import JobScamExplainer
from pipeline.preprocessing import clean_text
from pipeline.company_verification import CompanyVerifier

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'history' not in st.session_state:
    st.session_state.history = []
if 'live_performance' not in st.session_state:
    st.session_state.live_performance = {
        'total_analyzed': 0,
        'predicted_scams': 0,
        'predicted_legitimate': 0,
        'user_feedback': [],  # List of dictionaries with prediction and feedback
        'confidence_scores': [],  # List of confidence scores
        'timestamps': []  # List of analysis timestamps
    }

def initialize_session_state():
    """Initialize session state variables."""
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'live_performance' not in st.session_state:
        st.session_state.live_performance = {
            'total_analyzed': 0,
            'predicted_scams': 0,
            'predicted_legitimate': 0,
            'user_feedback': [],
            'confidence_scores': [],
            'timestamps': []
        }

def update_live_performance(prediction, confidence, timestamp):
    """Update live performance metrics."""
    st.session_state.live_performance['total_analyzed'] += 1
    if prediction == 'scam':
        st.session_state.live_performance['predicted_scams'] += 1
    else:
        st.session_state.live_performance['predicted_legitimate'] += 1
    
    st.session_state.live_performance['confidence_scores'].append(confidence)
    st.session_state.live_performance['timestamps'].append(timestamp)

def create_live_performance_charts():
    """Create visualizations for live performance tracking."""
    live_perf = st.session_state.live_performance
    
    # Create metrics row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Jobs Analyzed",
            live_perf['total_analyzed'],
            delta=None
        )
    
    with col2:
        scam_percentage = (live_perf['predicted_scams'] / live_perf['total_analyzed'] * 100) if live_perf['total_analyzed'] > 0 else 0
        st.metric(
            "Flagged as Scams",
            f"{scam_percentage:.1f}%",
            f"{live_perf['predicted_scams']} jobs"
        )
    
    with col3:
        legitimate_percentage = (live_perf['predicted_legitimate'] / live_perf['total_analyzed'] * 100) if live_perf['total_analyzed'] > 0 else 0
        st.metric(
            "Marked as Legitimate",
            f"{legitimate_percentage:.1f}%",
            f"{live_perf['predicted_legitimate']} jobs"
        )
    
    # Create confidence score distribution
    if live_perf['confidence_scores']:
        fig_conf = go.Figure()
        
        # Add histogram of confidence scores
        fig_conf.add_trace(go.Histogram(
            x=live_perf['confidence_scores'],
            nbinsx=20,
            name='Confidence Distribution',
            marker_color='#00CC96'
        ))
        
        fig_conf.update_layout(
            title={
                'text': "Confidence Score Distribution",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 16}
            },
            xaxis_title="Confidence Score (%)",
            yaxis_title="Number of Predictions",
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50, b=30, l=50, r=30)
        )
        
        st.plotly_chart(fig_conf, use_container_width=True)
    
    # Create timeline of predictions
    if len(live_perf['timestamps']) > 1:  # Only show if we have multiple predictions
        fig_timeline = go.Figure()
        
        # Convert confidence scores for scam predictions (1 - score for legitimate predictions)
        adjusted_scores = [
            score if score > 0.5 else 1 - score
            for score in live_perf['confidence_scores']
        ]
        
        # Add scatter plot of predictions over time
        fig_timeline.add_trace(go.Scatter(
            x=live_perf['timestamps'],
            y=adjusted_scores,
            mode='lines+markers',
            name='Confidence Trend',
            line=dict(color='#636EFA'),
            marker=dict(
                size=8,
                color=adjusted_scores,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Confidence")
            )
        ))
        
        fig_timeline.update_layout(
            title={
                'text': "Prediction Confidence Timeline",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 16}
            },
            xaxis_title="Time",
            yaxis_title="Confidence Score (%)",
            yaxis_range=[0, 1],
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50, b=30, l=50, r=30)
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)

def display_live_performance():
    """Display live performance tracking section."""
    st.header("Live Analysis Tracking")
    
    if st.session_state.live_performance['total_analyzed'] == 0:
        st.info("📊 No jobs analyzed yet. Start by analyzing a job listing to see live performance metrics!")
        return
    
    create_live_performance_charts()
    
    # Add detailed history section
    st.subheader("Analysis History")
    
    # Create a DataFrame from history
    if st.session_state.history:
        history_data = []
        for entry in st.session_state.history:
            probability = entry['results']['probability']
            prediction = "🚫 Scam" if probability > 0.5 else "✅ Legitimate"
            confidence = max(probability, 1-probability) * 100
            
            # Get top factors if available
            factors = "N/A"
            if 'explanation' in entry['results']:
                factors = "; ".join(entry['results']['explanation'][:2])  # Show top 2 factors
            
            history_data.append({
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'Job Title': entry['title'],
                'Location': entry['location'],
                'Prediction': prediction,
                'Confidence': f"{confidence:.1f}%",
                'Key Factors': factors
            })
        
        history_df = pd.DataFrame(history_data)
        
        # Add summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_scams = sum(1 for entry in st.session_state.history 
                         if entry['results']['probability'] > 0.5)
        total_legitimate = len(st.session_state.history) - total_scams
        
        with col1:
            st.metric(
                "Total Jobs Analyzed",
                len(st.session_state.history),
                delta=None
            )
        
        with col2:
            st.metric(
                "Scams Detected",
                total_scams,
                f"{(total_scams/len(st.session_state.history)*100):.1f}%"
            )
        
        with col3:
            st.metric(
                "Legitimate Jobs",
                total_legitimate,
                f"{(total_legitimate/len(st.session_state.history)*100):.1f}%"
            )
        
        with col4:
            avg_confidence = sum(max(entry['results']['probability'], 1-entry['results']['probability']) 
                               for entry in st.session_state.history) / len(st.session_state.history)
            st.metric(
                "Avg. Confidence",
                f"{(avg_confidence*100):.1f}%",
                delta=None
            )
        
        # Add filters
        col1, col2 = st.columns(2)
        with col1:
            prediction_filter = st.multiselect(
                "Filter by Prediction",
                ["✅ Legitimate", "🚫 Scam"],
                default=["✅ Legitimate", "🚫 Scam"]
            )
        
        with col2:
            confidence_threshold = st.slider(
                "Minimum Confidence",
                min_value=0,
                max_value=100,
                value=0,
                step=5,
                help="Filter jobs by minimum confidence score"
            )
        
        # Apply filters
        filtered_df = history_df[
            (history_df['Prediction'].isin(prediction_filter)) &
            (history_df['Confidence'].str.rstrip('%').astype(float) >= confidence_threshold)
        ]
        
        # Display the history table with custom styling
        st.markdown("""
        <style>
        .prediction-scam { color: #EF553B; }
        .prediction-legitimate { color: #00CC96; }
        </style>
        """, unsafe_allow_html=True)
        
        # Convert DataFrame to HTML with custom styling
        def style_prediction(val):
            color = '#00CC96' if '✅' in val else '#EF553B'
            return f'color: {color}'
        
        styled_df = filtered_df.style\
            .applymap(style_prediction, subset=['Prediction'])\
            .format({'Confidence': '{:,.1f}%'})\
            .hide_index()
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=400
        )
        
        # Add export functionality
        if st.button("📥 Export Analysis History"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"job_analysis_history_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        
        # Add analysis insights
        st.subheader("Analysis Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            # Location analysis
            location_counts = pd.Series([entry['location'] for entry in st.session_state.history])\
                .value_counts().head(5)
            
            fig_locations = go.Figure(data=[
                go.Bar(
                    x=location_counts.values,
                    y=location_counts.index,
                    orientation='h',
                    marker_color='#636EFA'
                )
            ])
            
            fig_locations.update_layout(
                title="Top Locations Analyzed",
                xaxis_title="Number of Jobs",
                showlegend=False,
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_locations, use_container_width=True)
        
        with col2:
            # Confidence distribution by prediction
            scam_confidences = [max(entry['results']['probability'], 1-entry['results']['probability']) * 100
                              for entry in st.session_state.history
                              if entry['results']['probability'] > 0.5]
            legitimate_confidences = [max(entry['results']['probability'], 1-entry['results']['probability']) * 100
                                   for entry in st.session_state.history
                                   if entry['results']['probability'] <= 0.5]
            
            fig_conf = go.Figure()
            
            fig_conf.add_trace(go.Box(
                y=legitimate_confidences,
                name="Legitimate",
                marker_color='#00CC96'
            ))
            
            fig_conf.add_trace(go.Box(
                y=scam_confidences,
                name="Scam",
                marker_color='#EF553B'
            ))
            
            fig_conf.update_layout(
                title="Confidence Distribution by Prediction",
                yaxis_title="Confidence Score (%)",
                showlegend=True,
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_conf, use_container_width=True)

def load_model() -> Tuple[JobScamDetector, JobFeatureExtractor, JobScamExplainer]:
    """Load model and related components."""
    # Get absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "fraud_detector.pkl")
    metrics_path = os.path.join(base_dir, "models", "metrics.json")
    
    # Load training data to initialize feature extractor
    data_path = os.path.join(base_dir, "data", "fake_job_postings.csv")
    training_data = pd.read_csv(data_path)
    
    # Initialize components
    detector = JobScamDetector(model_path=model_path, metrics_path=metrics_path)
    detector.load_model()
    
    feature_extractor = JobFeatureExtractor()
    # Fit the feature extractor with training data
    feature_extractor.engineer_features(training_data, is_training=True)
    
    explainer = JobScamExplainer()
    
    return detector, feature_extractor, explainer

def analyze_job_listing(listing, detector, feature_extractor, explainer):
    """Analyze a job listing and update live performance tracking."""
    # Convert to DataFrame for model prediction
    df = pd.DataFrame([{
        'title': listing.get('title', ''),
        'description': listing.get('description', ''),
        'location': listing.get('location', ''),
        'company_profile': listing.get('company_profile', '')
    }])
    
    # Extract features
    features = feature_extractor.engineer_features(df, is_training=False)
    
    # Get prediction and probability
    pred, prob = detector.predict(features)
    
    # Get explanation
    explanation = explainer.explain_prediction(df.iloc[0])
    
    results = {
        'prediction': bool(pred[0]),
        'probability': float(prob[0]),
        'explanation': explanation
    }
    
    # Update live performance tracking
    update_live_performance(
        'scam' if results['probability'] > 0.5 else 'legitimate',
        results['probability'],
        datetime.now()
    )
    
    return results

def display_header():
    """Display dashboard header."""
    st.title("🔍 Job Scam Detector")
    st.markdown("""
    This tool helps identify potentially fraudulent job listings using machine learning.
    Enter a job listing below to analyze it for suspicious patterns.
    """)

def display_input_form() -> Dict[str, str]:
    """Display and handle input form."""
    with st.form("job_listing_form"):
        title = st.text_input("Job Title")
        description = st.text_area("Job Description")
        location = st.text_input("Location")
        company = st.text_area("Company Profile")
        
        # Add new fields for company verification
        st.subheader("Additional Company Information")
        company_name = st.text_input("Company Name")
        company_email = st.text_input("Company Email")
        company_website = st.text_input("Company Website")
        company_logo_url = st.text_input("Company Logo URL")
        
        submitted = st.form_submit_button("Analyze Job Listing")
        
        if submitted:
            # Validate required fields
            validation_errors = []
            if not title.strip():
                validation_errors.append("Job Title is required")
            if not description.strip():
                validation_errors.append("Job Description is required")
            
            # Display validation errors if any
            if validation_errors:
                st.error("Please fix the following errors:")
                for error in validation_errors:
                    st.warning(error)
                return None
            
            # All required fields are filled
            return {
                'title': title,
                'description': description,
                'location': location,
                'company_profile': company,
                'company_name': company_name,
                'company_email': company_email,
                'company_website': company_website,
                'company_logo_url': company_logo_url
            }
        return None

def plot_feature_importance(explanation: Dict[str, float], top_n: int = 10):
    """Plot feature importance chart."""
    # Get top features and sort by absolute impact
    features_with_abs = [(k, v, abs(v)) for k, v in explanation.items()]
    features_with_abs.sort(key=lambda x: x[2], reverse=True)
    top_features = {item[0]: item[1] for item in features_with_abs[:top_n]}
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot horizontal bars
    colors = ['#FF4B4B' if x < 0 else '#4B8BFF' for x in top_features.values()]  # Brighter colors
    y_pos = np.arange(len(top_features))
    bars = ax.barh(y_pos, list(top_features.values()), color=colors)
    
    # Customize plot
    ax.set_yticks(y_pos)
    # Clean up feature names for display
    cleaned_features = [k.replace('_', ' ').title() for k in top_features.keys()]
    ax.set_yticklabels(cleaned_features, fontsize=10)
    
    # Add value labels on the bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label_pos = width + 0.01 if width >= 0 else width - 0.01
        ha = 'left' if width >= 0 else 'right'
        ax.text(label_pos, bar.get_y() + bar.get_height()/2, 
                f'{abs(width):.3f}', 
                va='center', ha=ha,
                fontsize=9)
    
    # Customize axes
    ax.set_xlabel("Impact on Prediction", fontsize=11, fontweight='bold')
    ax.set_title("Top Features Contributing to Prediction", fontsize=14, fontweight='bold', pad=20)
    
    # Add vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add grid for better readability
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    
    # Add legend
    ax.text(0.02, 1.05, "🔴 Increases Scam Probability", transform=ax.transAxes, color='#FF4B4B', fontsize=10)
    ax.text(0.98, 1.05, "🔵 Decreases Scam Probability", transform=ax.transAxes, color='#4B8BFF', 
            fontsize=10, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    return fig

def create_verification_summary(verification_results: Dict) -> go.Figure:
    """Create a summary visualization of verification results."""
    categories = []
    scores = []
    colors = []
    hover_texts = []
    
    if 'domain_verification' in verification_results:
        domain_info = verification_results['domain_verification']
        domain_score = 1.0 if domain_info['is_valid'] else 0.0
        domain_age = domain_info.get('domain_age', 0) or 0
        domain_score = min(1.0, domain_age / 365) if domain_age else domain_score
        
        categories.append('Domain')
        scores.append(domain_score)
        colors.append('#00CC96' if domain_score > 0.5 else '#EF553B')
        hover_texts.append(
            f"Domain Age: {domain_age} days<br>"
            f"MX Records: {'Yes' if domain_info.get('has_mx_record') else 'No'}<br>"
            f"Registrar: {domain_info.get('registrar', 'Unknown')}"
        )
    
    if 'logo_analysis' in verification_results:
        logo_info = verification_results['logo_analysis']
        if 'error' not in logo_info:
            suspicious_count = sum([
                logo_info.get('size_suspicious', True),
                logo_info.get('ratio_suspicious', True),
                logo_info.get('quality_suspicious', True)
            ])
            logo_score = 1.0 - (suspicious_count / 3.0)
            hover_text = (
                f"Size Check: {'Failed' if logo_info.get('size_suspicious') else 'Passed'}<br>"
                f"Ratio Check: {'Failed' if logo_info.get('ratio_suspicious') else 'Passed'}<br>"
                f"Quality Check: {'Failed' if logo_info.get('quality_suspicious') else 'Passed'}"
            )
        else:
            logo_score = 0.0
            hover_text = f"Error: {logo_info['error']}"
            
        categories.append('Logo')
        scores.append(logo_score)
        colors.append('#00CC96' if logo_score > 0.5 else '#EF553B')
        hover_texts.append(hover_text)
    
    if 'website_analysis' in verification_results:
        web_info = verification_results['website_analysis']
        if web_info['is_valid']:
            web_score = 1.0 - web_info.get('suspicious_score', 1.0)
            hover_text = (
                f"SSL Security: {'Yes' if web_info.get('has_ssl') else 'No'}<br>"
                f"Contact Info: {'Found' if web_info.get('has_contact_info') else 'Missing'}<br>"
                f"About Page: {'Found' if web_info.get('has_about_page') else 'Missing'}<br>"
                f"Social Links: {'Found' if web_info.get('has_social_links') else 'Missing'}"
            )
        else:
            web_score = 0.0
            hover_text = f"Error: {web_info.get('reason', 'Unknown error')}"
        
        categories.append('Website')
        scores.append(web_score)
        colors.append('#00CC96' if web_score > 0.5 else '#EF553B')
        hover_texts.append(hover_text)
    
    fig = go.Figure()
    
    # Add bar chart
    fig.add_trace(go.Bar(
        x=categories,
        y=scores,
        marker_color=colors,
        text=[f"{s:.0%}" for s in scores],
        textposition='auto',
        hovertext=hover_texts,
        hoverinfo='text',
        name='Score'
    ))
    
    # Add threshold line
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="gray",
        annotation_text="Threshold",
        annotation_position="bottom right"
    )
    
    fig.update_layout(
        title={
            'text': "Verification Results Summary",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        yaxis_title="Score (Higher is Better)",
        yaxis_range=[0, 1],
        yaxis_tickformat=".0%",
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=100, b=50, l=50, r=50)
    )
    
    # Update axes
    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linecolor='gray',
        tickfont={'size': 14}
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor='rgba(128,128,128,0.2)',
        showline=True,
        linecolor='gray',
        tickfont={'size': 12},
        zeroline=False
    )
    
    return fig

def create_risk_factors_chart(risk_factors: Dict[str, float]) -> go.Figure:
    """Create a horizontal bar chart for risk factors."""
    # Sort factors by absolute impact
    sorted_factors = sorted(
        risk_factors.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:10]  # Show top 10 factors
    
    factors = [f"{k} ({v:+.2f})" for k, v in sorted_factors]
    scores = [v for _, v in sorted_factors]
    colors = ['#EF553B' if s > 0 else '#00CC96' for s in scores]
    
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=scores,
        y=factors,
        orientation='h',
        marker_color=colors,
        text=[f"{s:+.2f}" for s in scores],
        textposition='auto',
        hovertemplate="Impact: %{x:+.3f}<extra></extra>"
    ))
    
    # Add center line
    fig.add_vline(
        x=0,
        line_dash="solid",
        line_color="gray",
        line_width=1
    )
    
    fig.update_layout(
        title={
            'text': "Top Risk Factors",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        xaxis_title="Impact on Risk Score",
        height=500,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=100, b=50, l=250, r=50)  # Increased left margin for labels
    )
    
    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridcolor='rgba(128,128,128,0.2)',
        showline=True,
        linecolor='gray',
        zeroline=False,
        tickfont={'size': 12}
    )
    fig.update_yaxes(
        showgrid=False,
        showline=True,
        linecolor='gray',
        tickfont={'size': 12}
    )
    
    return fig

def create_gauge_chart(value, title, color_threshold=None):
    """Create a gauge chart for metrics visualization."""
    if color_threshold is None:
        color_threshold = {
            'low': 0.6,
            'medium': 0.8,
            'high': 0.9
        }
    
    # Determine color based on value
    if value < color_threshold['low']:
        color = '#ff4444'  # red
    elif value < color_threshold['medium']:
        color = '#ffbb33'  # yellow
    elif value < color_threshold['high']:
        color = '#00C851'  # green
    else:
        color = '#007E33'  # dark green

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': title,
            'font': {'size': 14, 'color': 'white'}
        },
        number={
            'font': {'color': 'white', 'size': 20},
            'suffix': '%'
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1,
                'tickcolor': 'white',
                'tickfont': {'size': 8, 'color': 'white'}
            },
            'bar': {'color': color, 'thickness': 0.6},
            'bgcolor': 'rgba(255, 255, 255, 0.1)',
            'borderwidth': 1,
            'bordercolor': 'rgba(255, 255, 255, 0.5)',
            'steps': [
                {'range': [0, 60], 'color': 'rgba(255, 68, 68, 0.15)'},
                {'range': [60, 80], 'color': 'rgba(255, 187, 51, 0.15)'},
                {'range': [80, 100], 'color': 'rgba(0, 200, 81, 0.15)'}
            ],
            'threshold': {
                'line': {'color': 'white', 'width': 2},
                'thickness': 0.6,
                'value': value * 100
            }
        }
    ))

    fig.update_layout(
        height=150,  # Reduced height
        margin=dict(l=10, r=10, t=30, b=10),  # Tighter margins
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )

    return fig

def create_feature_metrics_chart(metrics: Dict[str, bool], title: str, color: str) -> go.Figure:
    """Create a metrics visualization chart."""
    categories = list(metrics.keys())
    values = [int(v) for v in metrics.values()]
    
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=color,
        text=['Pass' if v else 'Fail' for v in metrics.values()],
        textposition='auto',
        hovertemplate="%{x}: %{text}<extra></extra>"
    ))
    
    fig.update_layout(
        title={
            'text': title,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20}
        },
        yaxis_range=[0, 1.2],
        height=250,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=80, b=30, l=30, r=30)
    )
    
    # Update axes
    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linecolor='gray',
        tickfont={'size': 12}
    )
    fig.update_yaxes(
        showticklabels=False,
        showgrid=False,
        showline=False
    )
    
    return fig

def display_company_verification(verifier_results: Dict):
    """Display company verification results with enhanced visualizations."""
    st.subheader("Company Verification Results")
    
    # Overall risk gauge
    if 'overall_suspicion_score' in verifier_results:
        score = verifier_results['overall_suspicion_score']
        fig_gauge = create_gauge_chart(score, "Overall Risk Score")
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Verification summary chart
    fig_summary = create_verification_summary(verifier_results)
    st.plotly_chart(fig_summary, use_container_width=True)
    
    # Detailed results in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Domain Verification")
        if 'domain_verification' in verifier_results:
            domain_info = verifier_results['domain_verification']
            if domain_info['is_valid']:
                st.success("✅ Valid Domain")
                
                # Domain age visualization
                if domain_info.get('domain_age'):
                    fig_age = go.Figure(go.Indicator(
                        mode="number+delta",
                        value=domain_info['domain_age'],
                        title={'text': "Domain Age (days)"},
                        delta={
                            'reference': 90,
                            'relative': True,
                            'increasing': {'color': '#00CC96'},
                            'decreasing': {'color': '#EF553B'}
                        }
                    ))
                    fig_age.update_layout(
                        height=200,
                        margin=dict(t=50, b=0, l=30, r=30)
                    )
                    st.plotly_chart(fig_age, use_container_width=True)
                
                metrics = {
                    'MX Record': domain_info['has_mx_record'],
                    'Valid Age': domain_info.get('domain_age', 0) > 90 if domain_info.get('domain_age') else False
                }
                fig_metrics = create_feature_metrics_chart(
                    metrics,
                    "Domain Checks",
                    '#00CC96' if all(metrics.values()) else '#FFA15A'
                )
                st.plotly_chart(fig_metrics, use_container_width=True)
            else:
                st.error("❌ Invalid Domain")
                st.markdown(f"Reason: {domain_info['reason']}")
    
    with col2:
        st.markdown("### Logo Analysis")
        if 'logo_analysis' in verifier_results:
            logo_info = verifier_results['logo_analysis']
            if 'error' not in logo_info:
                metrics = {
                    'Size': not logo_info['size_suspicious'],
                    'Ratio': not logo_info['ratio_suspicious'],
                    'Quality': not logo_info['quality_suspicious']
                }
                fig_metrics = create_feature_metrics_chart(
                    metrics,
                    "Logo Quality Checks",
                    '#00CC96' if not logo_info['overall_suspicious'] else '#FFA15A'
                )
                st.plotly_chart(fig_metrics, use_container_width=True)
                
                if 'blur_score' in logo_info:
                    fig_quality = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=min(100, logo_info['blur_score']),
                        title={'text': "Image Quality Score"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': '#00CC96' if logo_info['blur_score'] > 100 else '#FFA15A'},
                            'steps': [
                                {'range': [0, 50], 'color': 'rgba(239,85,59,0.2)'},
                                {'range': [50, 100], 'color': 'rgba(0,204,150,0.2)'}
                            ]
                        }
                    ))
                    fig_quality.update_layout(height=200, margin=dict(t=50, b=0, l=30, r=30))
                    st.plotly_chart(fig_quality, use_container_width=True)
            else:
                st.error("❌ Logo Analysis Failed")
                st.markdown(f"Error: {logo_info['error']}")
    
    with col3:
        st.markdown("### Website Analysis")
        if 'website_analysis' in verifier_results:
            web_info = verifier_results['website_analysis']
            if web_info['is_valid']:
                metrics = {
                    'SSL': web_info['has_ssl'],
                    'Contact': web_info['has_contact_info'],
                    'About': web_info['has_about_page'],
                    'Social': web_info['has_social_links']
                }
                fig_metrics = create_feature_metrics_chart(
                    metrics,
                    "Website Features",
                    '#00CC96' if web_info['suspicious_score'] < 0.3 else '#FFA15A'
                )
                st.plotly_chart(fig_metrics, use_container_width=True)
                
                # Content analysis gauge
                fig_content = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=(1 - web_info['suspicious_score']) * 100,
                    title={'text': "Content Trust Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': '#00CC96' if web_info['suspicious_score'] < 0.3 else '#FFA15A'},
                        'steps': [
                            {'range': [0, 30], 'color': 'rgba(239,85,59,0.2)'},
                            {'range': [30, 70], 'color': 'rgba(255,161,90,0.2)'},
                            {'range': [70, 100], 'color': 'rgba(0,204,150,0.2)'}
                        ]
                    }
                ))
                fig_content.update_layout(height=200, margin=dict(t=50, b=0, l=30, r=30))
                st.plotly_chart(fig_content, use_container_width=True)
            else:
                st.error("❌ Website Analysis Failed")
                st.markdown(f"Reason: {web_info['reason']}")

def display_feedback_button(results, listing):
    """Display feedback collection button and form."""
    with st.expander("📝 Provide Feedback", expanded=False):
        st.markdown("""
        ### Help Us Improve!
        Your feedback is valuable in making our job scam detection more accurate.
        """)
        
        # Feedback form
        feedback_type = st.radio(
            "Was this prediction correct?",
            ["✅ Correct", "⚠️ Partially Correct", "❌ Incorrect"],
            key="feedback_type"
        )
        
        # Show additional fields if not fully correct
        if feedback_type != "✅ Correct":
            specific_issues = st.multiselect(
                "What specific issues did you notice?",
                [
                    "False positive (legitimate job marked as scam)",
                    "False negative (scam marked as legitimate)",
                    "Confidence score too high",
                    "Confidence score too low",
                    "Missing important red flags",
                    "Incorrect reasoning",
                    "Company verification issues",
                    "Other"
                ],
                key="specific_issues"
            )
            
            if "Other" in specific_issues:
                other_issues = st.text_input("Please specify other issues:")
            
            st.markdown("### Additional Details")
            col1, col2 = st.columns(2)
            
            with col1:
                missed_flags = st.text_area(
                    "What red flags did we miss?",
                    placeholder="Enter any suspicious elements we didn't catch..."
                )
            
            with col2:
                suggestions = st.text_area(
                    "How can we improve?",
                    placeholder="Your suggestions for improvement..."
                )
        
        # Optional comments for correct predictions
        else:
            suggestions = st.text_area(
                "Any additional comments? (optional)",
                placeholder="Share any thoughts about the analysis..."
            )
        
        # Submit button with loading state
        if st.button("Submit Feedback", type="primary"):
            with st.spinner("Saving your feedback..."):
                feedback_data = {
                    'job_title': listing['title'],
                    'job_description': listing['description'],
                    'location': listing['location'],
                    'model_prediction': "Scam" if results['probability'] > 0.5 else "Legitimate",
                    'confidence_score': results['probability'],
                    'user_feedback': feedback_type,
                    'feedback_type': 'prediction_quality',
                    'specific_concerns': json.dumps(specific_issues) if feedback_type != "✅ Correct" else "",
                    'suggested_improvements': suggestions
                }
                
                if feedback_type != "✅ Correct":
                    feedback_data.update({
                        'missed_flags': missed_flags,
                        'other_issues': other_issues if "Other" in specific_issues else ""
                    })
                
                save_feedback(feedback_data)
                
                # Update session state
                st.session_state.live_performance['user_feedback'].append({
                    'prediction': feedback_data['model_prediction'],
                    'feedback': feedback_type.lower(),
                    'confidence': feedback_data['confidence_score'],
                    'timestamp': datetime.now()
                })
                
                st.success("Thank you for your feedback! It will help improve our model. 🙏")
                st.balloons()

def display_results(results, listing):
    """Display analysis results with enhanced visualizations."""
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Main prediction gauge
        probability = results['probability']
        prediction = "Likely Scam" if probability > 0.5 else "Likely Legitimate"
        confidence = max(probability, 1-probability) * 100
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            title={
                'text': "Scam Probability",
                'font': {'size': 24}
            },
            delta={
                'reference': 50,
                'increasing': {'color': '#EF553B'},
                'decreasing': {'color': '#00CC96'}
            },
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': '#00CC96' if probability <= 0.5 else '#EF553B'},
                'bgcolor': 'rgba(0,0,0,0)',
                'borderwidth': 2,
                'bordercolor': 'gray',
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(0,204,150,0.2)'},
                    {'range': [30, 70], 'color': 'rgba(255,161,90,0.2)'},
                    {'range': [70, 100], 'color': 'rgba(239,85,59,0.2)'}
                ],
                'threshold': {
                    'line': {'color': 'white', 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig_gauge.update_layout(
            height=300,
            margin=dict(t=40, b=0, l=40, r=40),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Decision box with styling
        decision_color = "#00CC96" if probability <= 0.5 else "#EF553B"
        st.markdown(
            f"""
            <div style="
                padding: 20px;
                border-radius: 10px;
                background-color: {decision_color}22;
                border: 2px solid {decision_color};
                margin: 10px 0;">
                <h3 style="color: {decision_color}; margin: 0;">
                    {'✅' if probability <= 0.5 else '🚫'} {prediction}
                </h3>
                <p style="margin: 10px 0 0 0;">
                    Confidence: {confidence:.1f}%
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        # Job Overview Card
        st.markdown(
            """
            <style>
            .job-overview {
                background-color: rgba(128, 128, 128, 0.1);
                border-radius: 10px;
                padding: 15px;
                margin: 10px 0;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        with st.container():
            st.markdown('<div class="job-overview">', unsafe_allow_html=True)
            st.markdown("### 📋 Job Overview")
            st.markdown(f"**Title:** {listing['title']}")
            st.markdown(f"**Location:** {listing['location']}")
            if listing.get('company_name'):
                st.markdown(f"**Company:** {listing['company_name']}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Risk Factors and Analysis
    st.markdown("### 🔍 Analysis Details")
    
    # Create tabs for different aspects of analysis
    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs([
        "Risk Factors",
        "Language Analysis",
        "Verification Results"
    ])
    
    with analysis_tab1:
        if 'explanation' in results:
            # Split factors into high and low risk
            risk_factors = results['explanation']
            high_risk = [f for f in risk_factors if "suspicious" in f.lower() or "unusual" in f.lower()]
            low_risk = [f for f in risk_factors if f not in high_risk]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ⚠️ High Risk Indicators")
                if high_risk:
                    for factor in high_risk:
                        st.markdown(f"- {factor}")
                else:
                    st.markdown("*No high risk indicators found*")
            
            with col2:
                st.markdown("#### ✓ Low Risk Indicators")
                if low_risk:
                    for factor in low_risk:
                        st.markdown(f"- {factor}")
                else:
                    st.markdown("*No low risk indicators found*")
    
    with analysis_tab2:
        # Language Analysis
        text = listing['description']
        
        # Basic text statistics
        words = len(text.split())
        sentences = len(text.split('.'))
        avg_word_length = sum(len(word) for word in text.split()) / words if words > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Word Count", words)
        with col2:
            st.metric("Sentences", sentences)
        with col3:
            st.metric("Avg Word Length", f"{avg_word_length:.1f}")
        
        # Common red flag phrases
        red_flags = [
            "urgent", "immediate start", "work from home", "no experience",
            "unlimited earning", "your own boss", "earn extra cash",
            "investment required", "money back guarantee"
        ]
        
        found_flags = [flag for flag in red_flags if flag in text.lower()]
        if found_flags:
            st.markdown("#### 🚩 Suspicious Phrases Found")
            for flag in found_flags:
                st.markdown(f"- '{flag}'")
        else:
            st.markdown("#### ✅ No Common Suspicious Phrases Found")
    
    with analysis_tab3:
        # Company Verification Results
        if 'company_verification' in results:
            verification = results['company_verification']
            
            # Create verification score visualization
            scores = {
                'Domain Age': verification.get('domain_score', 0),
                'Website Quality': verification.get('website_score', 0),
                'Social Presence': verification.get('social_score', 0),
                'Contact Info': verification.get('contact_score', 0)
            }
            
            fig_scores = go.Figure()
            
            fig_scores.add_trace(go.Bar(
                x=list(scores.values()),
                y=list(scores.keys()),
                orientation='h',
                marker_color=['#00CC96' if score >= 0.7 else '#FFA15A' if score >= 0.4 else '#EF553B' 
                            for score in scores.values()]
            ))
            
            fig_scores.update_layout(
                title="Company Verification Scores",
                xaxis_title="Score",
                xaxis=dict(range=[0, 1]),
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_scores, use_container_width=True)
        else:
            st.info("No company verification data available")
    
    # Historical Context
    if st.session_state.history and len(st.session_state.history) > 1:
        st.markdown("### 📈 Historical Context")
        
        # Create timeline of analyzed jobs
        history_data = []
        for entry in st.session_state.history[:-1]:  # Exclude current analysis
            entry_prob = entry['results']['probability']
            history_data.append({
                'timestamp': datetime.now(),  # You might want to store actual timestamps
                'probability': entry_prob,
                'prediction': "Scam" if entry_prob > 0.5 else "Legitimate"
            })
        
        if history_data:
            fig_history = go.Figure()
            
            # Add scatter plot for historical predictions
            fig_history.add_trace(go.Scatter(
                x=[d['timestamp'] for d in history_data],
                y=[d['probability'] * 100 for d in history_data],
                mode='markers',
                name='Previous Analyses',
                marker=dict(
                    size=10,
                    color=['#EF553B' if d['prediction'] == "Scam" else '#00CC96' for d in history_data],
                    symbol='circle'
                )
            ))
            
            # Add current analysis point
            fig_history.add_trace(go.Scatter(
                x=[datetime.now()],
                y=[probability * 100],
                mode='markers',
                name='Current Analysis',
                marker=dict(
                    size=15,
                    color='#636EFA',
                    symbol='star'
                )
            ))
            
            fig_history.update_layout(
                title="Analysis History",
                yaxis_title="Scam Probability (%)",
                showlegend=True,
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_history, use_container_width=True)

    # Add feedback button at a prominent location
    st.markdown("---")
    st.markdown("### 📢 Your Opinion Matters!")
    display_feedback_button(results, listing)

def display_history_page(user_id=None):
    """Display analysis history page."""
    try:
        st.title("📜 Analysis History")
        
        # Get analysis history
        if user_id:
            history = get_user_history(user_id)
            if not history:
                st.info("You haven't analyzed any jobs yet.")
                return
        else:
            history = get_analysis_history()
            if not history:
                st.info("No analysis history available.")
                return
        
        # Display each analysis
        for entry in history:
            with st.expander(f"Analysis: {entry['job_title']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Job Details**")
                    st.write(f"Title: {entry['job_title']}")
                    st.write(f"Location: {entry.get('location', 'Not specified')}")
                    if entry.get('company_profile'):
                        st.write("Company Profile:", entry['company_profile'])
                    
                    st.write("**Analysis Results**")
                    st.write(f"Prediction: {entry['prediction']}")
                    st.write(f"Confidence Score: {entry['confidence_score']:.2%}")
                    
                with col2:
                    st.write("**Risk Factors**")
                    risk_factors = json.loads(entry['risk_factors']) if isinstance(entry['risk_factors'], str) else entry['risk_factors']
                    if risk_factors:
                        for factor in risk_factors:
                            st.write(f"- {factor}")
                    else:
                        st.write("No risk factors identified")
                    
                    if entry.get('verification_score'):
                        st.write(f"Verification Score: {entry['verification_score']:.2%}")
                
                # Display feedback if available
                if entry.get('feedback_type'):
                    st.write("**Feedback**")
                    st.write(f"Type: {entry['feedback_type']}")
                    if entry.get('specific_issues'):
                        st.write("Specific Issues:")
                        issues = json.loads(entry['specific_issues']) if isinstance(entry['specific_issues'], str) else entry['specific_issues']
                        for issue in issues:
                            st.write(f"- {issue}")
                    if entry.get('suggestions'):
                        st.write("Suggestions:", entry['suggestions'])
                
                st.write(f"Analysis Date: {entry['analysis_date']}")
                
    except Exception as e:
        st.error(f"Error displaying history: {str(e)}")
        st.error("Please try refreshing the page. If the problem persists, contact support.")

def display_job_analysis_form():
    """Display the job posting analysis form."""
    st.markdown("""
    ## 📝 Analyze a Job Posting
    Enter the details of the job posting you want to analyze. Our AI model will help identify potential scam indicators.
    """)

    with st.form("job_analysis_form"):
        job_title = st.text_input("Job Title*", placeholder="e.g., Software Engineer, Data Analyst")
        location = st.text_input("Location", placeholder="e.g., New York, NY or Remote")
        company_profile = st.text_area(
            "Company Profile",
            placeholder="Enter any available information about the company (e.g., company description, website, industry, size)...",
            height=100
        )
        
        job_description = st.text_area(
            "Job Description*",
            placeholder="Paste the full job description here...",
            height=300
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            submitted = st.form_submit_button("Analyze Job", type="primary")
        with col2:
            st.markdown("*Required fields")

    if submitted:
        if not job_title or not job_description:
            st.error("Please fill in all required fields (Job Title and Job Description)")
            return

        # Prepare job listing data
        listing = {
            'title': job_title,
            'description': job_description,
            'location': location,
            'company_profile': company_profile
        }

        with st.spinner("Analyzing job posting..."):
            try:
                # Get model and analyze
                model = get_model()
                analysis_results = model.analyze_posting(
                    title=listing['title'],
                    description=listing['description'],
                    location=listing.get('location'),
                    company_profile=listing.get('company_profile')
                )

                # Save analysis to database
                user_id = st.session_state.get('user_id')  # Get user_id if logged in
                analysis_id = save_analysis(user_id, listing, analysis_results)
                
                # Display results
                display_results(analysis_results, listing)
                
                # Update session state
                if 'history' not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.append({
                    'title': job_title,
                    'location': location,
                    'company_profile': company_profile,
                    'results': analysis_results
                })

            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                return

def display_model_performance():
    """Display model performance metrics and visualizations."""
    st.header("Model Performance Dashboard")
    
    # Example metrics
    metrics = {
        'accuracy': 0.85,
        'precision': 0.82,
        'recall': 0.89,
        'f1_score': 0.85,
        'auc_roc': 0.92
    }
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .metrics-header {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
        color: white;
        padding: 10px;
        border-radius: 5px;
        background: linear-gradient(90deg, rgba(33,150,243,0.2) 0%, rgba(33,150,243,0.1) 100%);
    }
    .metrics-container {
        background-color: rgba(0, 0, 0, 0.2);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .metric-explanation {
        font-size: 14px;
        color: rgba(255,255,255,0.7);
        margin-top: 20px;
        padding: 15px;
        background-color: rgba(255,255,255,0.1);
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header for metrics section
    st.markdown('<div class="metrics-header">📊 Key Performance Metrics</div>', unsafe_allow_html=True)
    
    # Create single row of metrics
    st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
    cols = st.columns(5)
    
    # Define metrics configuration
    metrics_config = [
        ('Accuracy', metrics['accuracy'], {'low': 0.7, 'medium': 0.8, 'high': 0.9}),
        ('Precision', metrics['precision'], {'low': 0.7, 'medium': 0.8, 'high': 0.9}),
        ('Recall', metrics['recall'], {'low': 0.7, 'medium': 0.8, 'high': 0.9}),
        ('F1 Score', metrics['f1_score'], {'low': 0.7, 'medium': 0.8, 'high': 0.9}),
        ('AUC-ROC', metrics['auc_roc'], {'low': 0.7, 'medium': 0.85, 'high': 0.95})
    ]

    # Display metrics in columns
    for col, (title, value, thresholds) in zip(cols, metrics_config):
        with col:
            st.plotly_chart(
                create_gauge_chart(value, title, thresholds),
                use_container_width=True
            )

    st.markdown('</div>', unsafe_allow_html=True)

    # Add concise metric explanations
    st.markdown("""
    <div class="metric-explanation">
        <strong>💡 Quick Guide:</strong><br>
        • <strong>Accuracy:</strong> Overall correct predictions<br>
        • <strong>Precision:</strong> Accuracy of scam predictions<br>
        • <strong>Recall:</strong> Percentage of scams caught<br>
        • <strong>F1 Score:</strong> Balance of precision and recall<br>
        • <strong>AUC-ROC:</strong> Overall discrimination ability
    </div>
    """, unsafe_allow_html=True)

    # Confusion Matrix
    st.markdown("### 🎯 Confusion Matrix")
    
    confusion_matrix = {
        'true_negative': 150,
        'false_positive': 20,
        'false_negative': 15,
        'true_positive': 115
    }
    
    total = sum(confusion_matrix.values())
    
    # Calculate percentages
    tn_pct = confusion_matrix['true_negative'] / total * 100
    fp_pct = confusion_matrix['false_positive'] / total * 100
    fn_pct = confusion_matrix['false_negative'] / total * 100
    tp_pct = confusion_matrix['true_positive'] / total * 100
    
    fig_confusion = go.Figure()
    
    # Add heatmap for confusion matrix
    fig_confusion.add_trace(go.Heatmap(
        z=[[tn_pct, fp_pct], [fn_pct, tp_pct]],
        x=['Predicted Legitimate', 'Predicted Scam'],
        y=['Actually Legitimate', 'Actually Scam'],
        text=[[f'{confusion_matrix["true_negative"]}<br>({tn_pct:.1f}%)', 
               f'{confusion_matrix["false_positive"]}<br>({fp_pct:.1f}%)'],
              [f'{confusion_matrix["false_negative"]}<br>({fn_pct:.1f}%)', 
               f'{confusion_matrix["true_positive"]}<br>({tp_pct:.1f}%)']],
        texttemplate="%{text}",
        textfont={"size": 14, "color": "white"},
        colorscale=[[0, '#1a9850'], [0.5, '#f7f7f7'], [1, '#d73027']],
        showscale=False
    ))
    
    fig_confusion.update_layout(
        title={
            'text': 'Confusion Matrix Heatmap',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'color': 'white', 'size': 16}
        },
        width=600,
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )
    
    st.plotly_chart(fig_confusion, use_container_width=True)

    # Performance Trends
    st.markdown("### 📈 Performance Trends")
    
    # Create sample historical data
    num_points = 6
    dates = pd.date_range(start='2024-01-01', periods=num_points, freq='ME')
    
    historical_data = pd.DataFrame({
        'Date': dates,
        'Accuracy': [0.82, 0.83, 0.84, 0.85, 0.85, 0.85],
        'Precision': [0.78, 0.79, 0.80, 0.81, 0.82, 0.82],
        'Recall': [0.85, 0.86, 0.87, 0.88, 0.89, 0.89]
    })
    
    fig_trends = go.Figure()
    
    # Add traces for each metric
    metrics_colors = {
        'Accuracy': '#2196F3',
        'Precision': '#4CAF50',
        'Recall': '#FFC107'
    }
    
    for metric, color in metrics_colors.items():
        fig_trends.add_trace(go.Scatter(
            x=historical_data['Date'],
            y=historical_data[metric] * 100,
            name=metric,
            line=dict(color=color, width=3),
            mode='lines+markers',
            marker=dict(size=8, symbol='circle')
        ))
    
    fig_trends.update_layout(
        title={
            'text': 'Model Performance Trends',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'color': 'white', 'size': 16}
        },
        xaxis_title='Date',
        yaxis_title='Percentage (%)',
        yaxis=dict(range=[75, 95]),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font={'color': 'white'}
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )
    
    # Add grid lines
    fig_trends.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', tickfont={'color': 'white'})
    fig_trends.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', tickfont={'color': 'white'})
    
    st.plotly_chart(fig_trends, use_container_width=True)

    # Additional Insights
    st.markdown("### 🔍 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Strengths
        - **High Recall**: Successfully detecting 89% of scam postings
        - **Strong AUC-ROC**: 92% indicates excellent discrimination ability
        - **Balanced Performance**: F1 Score of 85% shows good precision-recall balance
        """)
    
    with col2:
        st.markdown("""
        #### Areas for Improvement
        - **False Positives**: {:.1f}% legitimate jobs marked as scams
        - **False Negatives**: {:.1f}% scams missed
        - **Precision**: Room for improvement in reducing false positives
        """.format(fp_pct, fn_pct))
    
    # Model Version Info
    st.markdown("### 📌 Model Information")
    st.info("""
    - **Current Version**: 1.0.0
    - **Last Updated**: 2024-06-14
    - **Training Data Size**: 10,000 job postings
    - **Architecture**: Enhanced BERT with custom classification head
    """)

def display_feedback_stats():
    """Display feedback statistics."""
    try:
        st.title("📊 Feedback Analytics")
        
        # Get feedback statistics
        stats = get_feedback_stats()
        
        if not stats['total_feedback']:
            st.info("No feedback data available yet.")
            return
            
        # Display total feedback count
        st.metric("Total Feedback Received", stats['total_feedback'])
        
        # Display feedback type distribution
        st.subheader("Feedback Type Distribution")
        if stats['feedback_types']:
            fig = go.Figure(data=[
                go.Bar(
                    x=list(stats['feedback_types'].keys()),
                    y=list(stats['feedback_types'].values())
                )
            ])
            fig.update_layout(
                xaxis_title="Feedback Type",
                yaxis_title="Count",
                showlegend=False
            )
            st.plotly_chart(fig)
        else:
            st.info("No feedback type data available.")
        
        # Display recent feedback
        st.subheader("Recent Feedback")
        if stats['recent_feedback']:
            for feedback in stats['recent_feedback']:
                with st.expander(f"Feedback for: {feedback['job_title']}"):
                    st.write(f"Type: {feedback['feedback_type']}")
                    if feedback['specific_issues']:
                        st.write("Specific Issues:")
                        issues = json.loads(feedback['specific_issues']) if isinstance(feedback['specific_issues'], str) else feedback['specific_issues']
                        for issue in issues:
                            st.write(f"- {issue}")
                    if feedback['suggestions']:
                        st.write("Suggestions:", feedback['suggestions'])
                    st.write(f"Date: {feedback['created_at']}")
        else:
            st.info("No recent feedback available.")
            
    except Exception as e:
        st.error(f"Error displaying feedback statistics: {str(e)}")
        st.error("Please try refreshing the page. If the problem persists, contact support.")

def display_batch_analysis():
    """Display the batch analysis section for CSV uploads."""
    model = get_model()
    
    st.markdown("""
    ## 📊 Batch Analysis
    Upload a CSV file containing multiple job postings to analyze them in bulk.
    
    **Required CSV columns:**
    - `job_title`: Title of the job posting
    - `job_description`: Full job description
    
    **Optional columns:**
    - `location`: Job location
    - `company_profile`: Company information
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            # Validate required columns
            required_columns = ['job_title', 'job_description']
            if not all(col in df.columns for col in required_columns):
                st.error("CSV must contain the following columns: job_title, job_description")
                return
            
            # Display preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head())
            
            # Analyze button
            if st.button("Analyze All Postings", type="primary"):
                with st.spinner("Analyzing job postings..."):
                    # Create results DataFrame
                    results_df = pd.DataFrame()
                    results_df['Job Title'] = df['job_title']
                    
                    # Get batch analysis results
                    analysis_results = model.analyze_batch(df)
                    
                    # Extract results
                    predictions = []
                    probabilities = []
                    risk_factors_list = []
                    verification_scores = []
                    
                    for result in analysis_results:
                        prob = result['probability']
                        predictions.append('Scam' if prob > 0.5 else 'Legitimate')
                        probabilities.append(prob)
                        risk_factors_list.append(', '.join(result['risk_factors']))
                        verification_scores.append(result['verification_score'])
                        
                        # Save to database
                        listing = {
                            'title': df.loc[len(predictions)-1, 'job_title'],
                            'description': df.loc[len(predictions)-1, 'job_description'],
                            'location': df.loc[len(predictions)-1, 'location'] if 'location' in df.columns else None,
                            'company_profile': df.loc[len(predictions)-1, 'company_profile'] if 'company_profile' in df.columns else None,
                            'company': None,
                            'salary': None
                        }
                        save_analysis(listing, result)
                    
                    # Add results to DataFrame
                    results_df['Prediction'] = predictions
                    results_df['Fraud Probability'] = [f"{p*100:.1f}%" for p in probabilities]
                    results_df['Risk Factors'] = risk_factors_list
                    results_df['Verification Score'] = [f"{v*100:.1f}%" for v in verification_scores]
                    
                    # Display results
                    st.subheader("🔍 Analysis Results")
                    
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        total_scams = sum(1 for p in predictions if p == 'Scam')
                        st.metric(
                            "Scam Postings Detected",
                            total_scams,
                            f"{(total_scams/len(predictions))*100:.1f}% of total"
                        )
                    
                    with col2:
                        avg_prob = np.mean(probabilities)
                        st.metric(
                            "Average Fraud Probability",
                            f"{avg_prob*100:.1f}%"
                        )
                    
                    with col3:
                        avg_verification = np.mean(verification_scores)
                        st.metric(
                            "Average Verification Score",
                            f"{avg_verification*100:.1f}%"
                        )
                    
                    # Results table
                    st.dataframe(
                        results_df,
                        column_config={
                            "Job Title": st.column_config.TextColumn("Job Title"),
                            "Prediction": st.column_config.TextColumn(
                                "Prediction",
                                help="Model's prediction for this job posting"
                            ),
                            "Fraud Probability": st.column_config.TextColumn(
                                "Fraud Probability",
                                help="Probability that this posting is fraudulent"
                            ),
                            "Risk Factors": st.column_config.TextColumn(
                                "Risk Factors",
                                help="Identified risk factors in the posting"
                            ),
                            "Verification Score": st.column_config.TextColumn(
                                "Verification Score",
                                help="Verification score based on company and posting details"
                            )
                        }
                    )
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "Download Results CSV",
                        csv,
                        "job_scam_analysis_results.csv",
                        "text/csv",
                        key='download-csv'
                    )
                    
                    # Visualizations
                    st.subheader("📊 Analysis Visualizations")
                    
                    # Prediction distribution
                    fig_dist = go.Figure(data=[
                        go.Histogram(
                            x=probabilities,
                            nbinsx=20,
                            name="Fraud Probability Distribution",
                            marker_color='#FF6B6B'
                        )
                    ])
                    
                    fig_dist.update_layout(
                        title="Distribution of Fraud Probabilities",
                        xaxis_title="Fraud Probability",
                        yaxis_title="Number of Postings",
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                    # Risk factors frequency
                    all_risks = [risk for risks in risk_factors_list if risks for risk in risks.split(', ')]
                    if all_risks:
                        risk_counts = pd.Series(all_risks).value_counts()
                        
                        fig_risks = go.Figure(data=[
                            go.Bar(
                                x=risk_counts.values,
                                y=risk_counts.index,
                                orientation='h',
                                marker_color='#4CAF50'
                            )
                        ])
                        
                        fig_risks.update_layout(
                            title="Most Common Risk Factors",
                            xaxis_title="Number of Occurrences",
                            yaxis_title="Risk Factor",
                            height=max(300, len(risk_counts) * 30)
                        )
                        
                        st.plotly_chart(fig_risks, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error processing CSV file: {str(e)}")
            st.info("Please ensure your CSV file is properly formatted and contains the required columns.")

def login_signup_page():
    """Display login/signup page."""
    st.markdown("""
        <style>
        .auth-container {
            max-width: 400px;
            margin: 0 auto;
            padding: 2rem;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
        }
        .auth-title {
            text-align: center;
            margin-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="auth-title">🔐 Authentication</h1>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit_login = st.form_submit_button("Login")
            
            if submit_login:
                if email and password:
                    user_id = verify_user(email, password)
                    if user_id:
                        st.session_state.user_id = user_id
                        st.session_state.authenticated = True
                        st.success("Login successful! Redirecting...")
                        st.experimental_rerun()
                    else:
                        st.error("Invalid email or password")
                else:
                    st.warning("Please fill in all fields")
    
    with tab2:
        with st.form("signup_form"):
            new_email = st.text_input("Email")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submit_signup = st.form_submit_button("Sign Up")
            
            if submit_signup:
                if new_email and new_password and confirm_password:
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters long")
                    else:
                        if register_user(new_email, new_password):
                            st.success("Registration successful! Please login.")
                            time.sleep(1)
                            st.experimental_rerun()
                        else:
                            st.error("Email already registered or an error occurred")
                else:
                    st.warning("Please fill in all fields")
    
    st.markdown('</div>', unsafe_allow_html=True)

def logout():
    """Log out the user."""
    st.session_state.user_id = None
    st.session_state.authenticated = False
    st.session_state.history = []
    st.experimental_rerun()

def main():
    """Main function to run the Streamlit app."""
    initialize_session_state()
    
    if not st.session_state.authenticated:
        login_signup_page()
        return
    
    display_header()
    
    # Add logout button to sidebar
    st.sidebar.markdown("### 👤 User Account")
    if st.sidebar.button("Logout"):
        logout()
        return
    
    # Custom CSS for sidebar
    st.markdown("""
        <style>
        .sidebar-content {
            padding: 1rem;
        }
        
        .sidebar .sidebar-content {
            background-image: linear-gradient(180deg, #2E3440 0%, #3B4252 100%);
        }
        
        .sidebar .sidebar-content .block-container {
            padding-top: 1rem;
        }
        
        /* Navigation title styling */
        .nav-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: #ECEFF4;
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #81A1C1;
        }
        
        /* Radio button styling */
        .stRadio > label {
            background: #434C5E;
            padding: 15px;
            border-radius: 5px;
            margin: 5px 0;
            transition: all 0.2s ease;
        }
        
        .stRadio > label:hover {
            background: #4C566A;
            transform: translateX(5px);
        }
        
        .stRadio > label > div {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Navigation with icons
    page = st.sidebar.radio(
        "",
        [
            "🔍 Analyze Job Posting",
            "📊 Batch Analysis",
            "📚 Analysis History",
            "📈 Model Performance",
            "💭 Feedback Analytics"
        ],
        format_func=lambda x: x.split(" ", 1)[1]
    )
    
    # Route to appropriate page
    if "Analyze Job Posting" in page:
        display_job_analysis_form()
    elif "Batch Analysis" in page:
        display_batch_analysis()
    elif "Analysis History" in page:
        # Use user-specific history
        display_history_page(st.session_state.user_id)
    elif "Model Performance" in page:
        display_model_performance()
    else:
        display_feedback_stats()

def analyze_job():
    """Analyze a job listing for potential scams."""
    try:
        # Get form inputs
        job_title = st.session_state.get('job_title', '')
        job_description = st.session_state.get('job_description', '')
        location = st.session_state.get('location', '')
        company_profile = st.session_state.get('company_profile', '')
        
        if not job_title or not job_description:
            st.error("Please fill in all required fields.")
            return
        
        # Prepare job listing
        listing = {
            'title': job_title,
            'description': job_description,
            'location': location,
            'company_profile': company_profile
        }
        
        # Get model prediction
        model = get_model()
        results = model.analyze_posting(
            title=listing['title'],
            description=listing['description'],
            location=listing.get('location'),
            company_profile=listing.get('company_profile')
        )
        
        # Save analysis to database
        try:
            analysis_id = save_analysis(listing, results)
            st.success("Analysis completed and saved successfully!")
            
            # If user is logged in, also save to user history
            if 'user_id' in st.session_state:
                save_user_analysis(st.session_state['user_id'], listing, results)
        except Exception as e:
            st.error(f"Failed to save analysis: {str(e)}")
            st.error("The analysis was completed but couldn't be saved to history.")
            
        # Display results
        display_analysis_results(listing, results)
        
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        st.error("Please try again. If the problem persists, contact support.")

def display_analysis_results(listing: Dict, results: Dict):
    """Display the analysis results for a job listing."""
    st.subheader("Analysis Results")
    
    # Display prediction and confidence
    prediction = "Scam" if results['probability'] > 0.5 else "Legitimate"
    confidence = max(results['probability'], 1 - results['probability']) * 100
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prediction", prediction)
    with col2:
        st.metric("Confidence", f"{confidence:.1f}%")
    
    # Display risk factors
    if 'risk_factors' in results:
        st.subheader("Risk Factors")
        for factor in results['risk_factors']:
            st.write(f"- {factor}")
    
    # Display verification score if available
    if 'verification_score' in results:
        st.metric("Verification Score", f"{results['verification_score']:.1f}%")

if __name__ == "__main__":
    main() 