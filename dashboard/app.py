import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import sys
import time
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import sqlite3
from pathlib import Path
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
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
    get_feedback_stats,
    get_model_performance_metrics
)
from model import JobScamDetector

# Initialize database at startup
init_database()

# Initialize model with caching
@st.cache_resource(show_spinner=False)
def get_model():
    """Get or create a cached instance of JobScamDetector."""
    try:
        model = JobScamDetector()
        # Test the model with a simple input to ensure it's working
        test_result = model.analyze_posting(
            title="Software Engineer",
            description="Test job posting",
            location="Remote",
            company_profile="Test company"
        )
        if test_result is None:
            raise Exception("Model initialization test failed")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please ensure all required files are present and dependencies are installed.")
        return None

# Must be the first Streamlit command
st.set_page_config(
    page_title="Job Scam Detector",
    page_icon="🔍",
    layout="wide"
)

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

def load_model() -> JobScamDetector:
    """Load model and related components."""
    try:
        detector = JobScamDetector()
        return detector
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def analyze_job_listing(listing, detector=None):
    """Analyze a job listing and update live performance tracking."""
    try:
        if detector is None:
            detector = get_model()
            if detector is None:
                raise Exception("Could not initialize model")
        
        # Get prediction from model
        results = detector.analyze_posting(
            title=listing.get('title', ''),
            description=listing.get('description', ''),
            location=listing.get('location', ''),
            company_profile=listing.get('company_profile', '')
        )
        
        if results is None:
            raise Exception("Model returned no results")
        
        # Update live performance tracking
        update_live_performance(
            'scam' if results.get('probability', 0) > 0.5 else 'legitimate',
            results.get('probability', 0),
            datetime.now()
        )
        
        return results
        
    except Exception as e:
        st.error(f"Error analyzing job listing: {str(e)}")
        return None

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

def display_feedback_button(results, listing, analysis_id):
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
            key=f"feedback_type_{analysis_id}"
        )
        
        # Show additional fields if not fully correct
        specific_issues = []
        other_issues = ""
        suggestions = ""
        
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
                key=f"specific_issues_{analysis_id}"
            )
            
            if "Other" in specific_issues:
                other_issues = st.text_input(
                    "Please specify other issues:",
                    key=f"other_issues_{analysis_id}"
                )
        
        suggestions = st.text_area(
            "Additional Comments" if feedback_type == "✅ Correct" else "How can we improve?",
            placeholder="Share your thoughts to help us improve...",
            key=f"suggestions_{analysis_id}"
        )
        
        # Submit button with loading state
        if st.button("Submit Feedback", type="primary", key=f"submit_{analysis_id}"):
            with st.spinner("Saving your feedback..."):
                try:
                    if not hasattr(st.session_state, 'user_id'):
                        st.error("Please log in to submit feedback.")
                        return
                    
                    # Prepare feedback data
                    all_specific_issues = specific_issues.copy()
                    if other_issues:
                        all_specific_issues.append(other_issues)
                    
                    feedback_data = {
                        'feedback_type': feedback_type,
                        'specific_issues': all_specific_issues if all_specific_issues else None,
                        'suggestions': suggestions if suggestions else None
                    }
                    
                    # Save feedback
                    if save_user_feedback(
                        st.session_state.user_id,
                        analysis_id,
                        feedback_data
                    ):
                        st.success("Thank you for your feedback! It will help improve our model. 🙏")
                        st.balloons()
                        
                        # Update session state
                        if 'live_performance' not in st.session_state:
                            st.session_state.live_performance = {
                                'user_feedback': []
                            }
                        
                        st.session_state.live_performance['user_feedback'].append({
                            'prediction': results.get('prediction', 'Unknown'),
                            'feedback': feedback_type,
                            'confidence': results.get('probability', 0.0),
                            'timestamp': datetime.now()
                        })
                        
                        # Clear the form
                        for key in st.session_state.keys():
                            if key.endswith(str(analysis_id)):
                                del st.session_state[key]
                        
                        # Rerun to update the UI
                        st.rerun()
                    else:
                        st.error("Failed to save feedback. Please try again.")
                except Exception as e:
                    st.error(f"Error saving feedback: {str(e)}")
                    logger.error(f"Error in feedback submission: {e}")

def display_results(results, listing, analysis_id):
    """Display analysis results with enhanced visualizations."""
    if not results:
        st.error("No analysis results available.")
        return

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
            st.markdown(f"**Location:** {listing.get('location', 'Not specified')}")
            if listing.get('company_profile'):
                st.markdown(f"**Company Profile:** {listing['company_profile']}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Risk Factors and Analysis
    st.markdown("### 🔍 Analysis Details")
    
    # Create tabs for different aspects of analysis
    analysis_tab1, analysis_tab2 = st.tabs([
        "Risk Factors",
        "Verification Results"
    ])
    
    with analysis_tab1:
        if 'risk_factors' in results:
            risk_factors = results['risk_factors']
            if risk_factors:
                st.markdown("#### ⚠️ Identified Risk Factors")
                for factor in risk_factors:
                    st.markdown(f"- {factor}")
            else:
                st.markdown("*No significant risk factors identified*")
    
    with analysis_tab2:
        if 'verification_score' in results:
            score = results['verification_score']
            st.metric("Verification Score", f"{score*100:.1f}%")
            
            # Score interpretation
            if score >= 0.8:
                st.success("✅ High credibility - Most verification checks passed")
            elif score >= 0.5:
                st.warning("⚠️ Medium credibility - Some verification checks failed")
            else:
                st.error("❌ Low credibility - Multiple verification checks failed")
    
    # Add feedback button
    st.markdown("---")
    st.markdown("### 📢 Your Opinion Matters!")
    display_feedback_button(results, listing, analysis_id)

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
                if model is None:
                    return
                    
                analysis_results = model.analyze_posting(
                    title=listing['title'],
                    description=listing['description'],
                    location=listing.get('location'),
                    company_profile=listing.get('company_profile')
                )

                # Save analysis to database
                try:
                    user_id = st.session_state.get('user_id')  # Get user_id if logged in
                    analysis_id = save_analysis(user_id, listing, analysis_results)
                except Exception as e:
                    st.warning("Analysis completed but couldn't be saved to history.")
                    st.error(f"Database error: {str(e)}")
                
                # Display results
                display_results(analysis_results, listing, analysis_id)
                
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
                st.error("Please try again. If the problem persists, contact support.")
                return

def create_metric_gauge(value, title, color):
    """Create a gauge chart for metrics."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={'text': title, 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "rgba(255, 255, 255, 0.1)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 60], 'color': 'rgba(255, 0, 0, 0.1)'},
                {'range': [60, 80], 'color': 'rgba(255, 165, 0, 0.1)'},
                {'range': [80, 100], 'color': 'rgba(0, 255, 0, 0.1)'}
            ],
        }
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )
    return fig

def display_model_performance():
    """Display model performance metrics and visualizations."""
    try:
        st.title("📈 Model Performance")
        
        # Add model training section
        st.sidebar.markdown("### 🔄 Model Training")
        if st.sidebar.button("Train Model"):
            with st.spinner("Training model with collected data..."):
                try:
                    from train_model import train_model
                    from data_collection import collect_training_data
                    
                    # Collect training data
                    training_data = collect_training_data("dashboard.db")
                    
                    if len(training_data) < 10:
                        st.warning("⚠️ Insufficient training data. Need at least 10 samples with feedback.")
                        st.info("Continue analyzing jobs and providing feedback to build the training dataset.")
                    else:
                        # Train model
                        metrics = train_model(training_data)
                        st.success(f"✅ Model trained successfully with {len(training_data)} samples!")
                        
                        # Show training results
                        st.write("### Training Results")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Training F1-Score",
                                f"{metrics['training']['f1_score']:.3f}",
                                help="F1-Score on training data"
                            )
                        
                        with col2:
                            st.metric(
                                "Test F1-Score",
                                f"{metrics['test']['f1_score']:.3f}",
                                help="F1-Score on test data"
                            )
                        
                        with col3:
                            st.metric(
                                "CV F1-Score",
                                f"{metrics['cross_validation']['f1_mean']:.3f}",
                                help="Mean F1-Score from cross-validation"
                            )
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")
        
        # Load metrics from trained model
        try:
            with open("models/metrics.json", "r") as f:
                metrics = json.load(f)
            
            # Calculate accuracy if not present
            if 'accuracy' not in metrics:
                total = metrics.get('true_positives', 0) + metrics.get('true_negatives', 0) + \
                       metrics.get('false_positives', 0) + metrics.get('false_negatives', 0)
                if total > 0:
                    metrics['accuracy'] = (metrics.get('true_positives', 0) + metrics.get('true_negatives', 0)) / total
                else:
                    metrics['accuracy'] = metrics.get('roc_auc', 0.9)
            
            using_real_metrics = True
            
        except FileNotFoundError:
            st.error("⚠️ No trained model found. Please use the 'Train Model' button to train with fake_job_postings dataset.")
            using_real_metrics = False
            metrics = {
                'F1-Score': {'value': 0.81, 'color': '#2ecc71'},
                'Precision': {'value': 0.76, 'color': '#3498db'},
                'Recall': {'value': 0.88, 'color': '#e74c3c'},
                'ROC-AUC': {'value': 0.99, 'color': '#f1c40f'}
            }
        
        # Create metric gauges in a grid
        st.write("### Key Performance Metrics")
        cols = st.columns(4)
        
        if using_real_metrics:
            metric_configs = {
                'F1-Score': {'value': metrics['f1_score'], 'color': '#2ecc71'},
                'Precision': {'value': metrics['precision'], 'color': '#3498db'},
                'Recall': {'value': metrics['recall'], 'color': '#e74c3c'},
                'ROC-AUC': {'value': metrics['roc_auc'], 'color': '#f1c40f'}
            }
            
            # Show training dataset info
            st.info("ℹ️ Model trained on fake_job_postings dataset with real performance metrics")
        else:
            metric_configs = metrics
            # Show error message
            st.error("⚠️ No trained model found. Please use the 'Train Model' button to train with fake_job_postings dataset.")
        
        for i, (metric, data) in enumerate(metric_configs.items()):
            with cols[i]:
                fig = create_metric_gauge(data['value'], metric, data['color'])
                st.plotly_chart(fig, use_container_width=True)
        
        # Create tabs for detailed visualizations
        tab1, tab2 = st.tabs(["📊 Performance Analysis", "📈 Detailed Metrics"])
        
        with tab1:
            # Create two columns for visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("F1-Score Components")
                f1_data = {
                    'Class': ['Scam', 'Legitimate'],
                    'Precision': [0.92, 0.88],
                    'Recall': [0.87, 0.93],
                    'F1-Score': [0.89, 0.90]
                }
                
                fig_f1 = go.Figure()
                bar_colors = ['#2ecc71', '#3498db', '#e74c3c']
                metrics_to_plot = ['Precision', 'Recall', 'F1-Score']
                
                for i, metric in enumerate(metrics_to_plot):
                    fig_f1.add_trace(go.Bar(
                        name=metric,
                        x=f1_data['Class'],
                        y=[f1_data[metric][j] for j in range(len(f1_data['Class']))],
                        marker_color=bar_colors[i]
                    ))
                
                fig_f1.update_layout(
                    barmode='group',
                    xaxis_title='Class',
                    yaxis_title='Score',
                    yaxis=dict(range=[0, 1]),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=400,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    font={'color': 'white'}
                )
                
                st.plotly_chart(fig_f1, use_container_width=True)
            
            with col2:
                st.subheader("Confusion Matrix")
                conf_matrix = np.array([
                    [150, 20],  # True Neg, False Pos
                    [10, 120]   # False Neg, True Pos
                ])
                
                # Calculate percentages for annotations
                total = conf_matrix.sum()
                percentages = conf_matrix / total * 100
                
                annotations = []
                for i in range(2):
                    for j in range(2):
                        annotations.append(
                            dict(
                                text=f"{conf_matrix[i, j]}<br>({percentages[i, j]:.1f}%)",
                                x=['Predicted Legitimate', 'Predicted Scam'][j],
                                y=['Actual Legitimate', 'Actual Scam'][i],
                                showarrow=False,
                                font=dict(color='white', size=14)
                            )
                        )
                
                fig_matrix = go.Figure(data=go.Heatmap(
                    z=conf_matrix,
                    x=['Predicted Legitimate', 'Predicted Scam'],
                    y=['Actual Legitimate', 'Actual Scam'],
                    colorscale=[[0, '#2ecc71'], [1, '#e74c3c']],
                    showscale=False
                ))
                
                fig_matrix.update_layout(
                    title='Confusion Matrix',
                    height=400,
                    annotations=annotations,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': 'white'}
                )
                
                st.plotly_chart(fig_matrix, use_container_width=True)
        
        with tab2:
            st.subheader("Performance Trends")
            
            # Generate sample time series data
            dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='W')
            base = np.linspace(0.80, 0.95, len(dates))
            noise = np.random.normal(0, 0.02, len(dates))
            
            performance_data = {
                'F1-Score': np.clip(base + noise, 0, 1),
                'Precision': np.clip(base + np.random.normal(0, 0.02, len(dates)), 0, 1),
                'Recall': np.clip(base + np.random.normal(0, 0.02, len(dates)), 0, 1)
            }
            
            fig_trends = go.Figure()
            colors = {'F1-Score': '#2ecc71', 'Precision': '#3498db', 'Recall': '#e74c3c'}
            
            for metric, values in performance_data.items():
                fig_trends.add_trace(go.Scatter(
                    x=dates,
                    y=values,
                    name=metric,
                    mode='lines+markers',
                    line=dict(color=colors[metric], width=3),
                    marker=dict(size=8)
                ))
            
            fig_trends.update_layout(
                title='Metric Trends Over Time',
                xaxis_title='Date',
                yaxis_title='Score',
                yaxis=dict(range=[0.75, 1.0]),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=400,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                hovermode='x unified',
                font={'color': 'white'}
            )
            
            st.plotly_chart(fig_trends, use_container_width=True)
            
            # Add metric insights
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### Performance Insights")
                st.markdown("""
                - **Strong Overall Performance**: All metrics consistently above 85%
                - **High Precision**: 92% accuracy in identifying scams
                - **Good Recall**: 87% of actual scams detected
                - **Balanced F1-Score**: Strong balance between precision and recall
                """)
            
            with col2:
                st.write("### Areas for Improvement")
                st.markdown("""
                - **False Positives**: Work on reducing legitimate jobs marked as scams
                - **Recall Enhancement**: Focus on catching more subtle scam patterns
                - **Class Balance**: Monitor and adjust for data imbalance
                - **Model Updates**: Regular retraining with new feedback
                """)
        
        # Add note about demo data
        st.info("ℹ️ Currently showing demo metrics. Start analyzing job listings to see real performance data!")
        
    except Exception as e:
        st.error(f"Error displaying model performance: {str(e)}")
        st.error("Please try refreshing the page. If the problem persists, contact support.")

def display_model_metrics_tab(model_metrics):
    """Display model performance metrics optimized for imbalanced datasets."""
    st.subheader("Model Performance Metrics")
    
    # If no real data, use demo data
    if model_metrics['total_predictions'] == 0:
        model_metrics = {
            'total_predictions': 100,
            'true_positives': 35,
            'false_positives': 10,
            'false_negatives': 5,
            'true_negatives': 50,
            'precision': 0.778,
            'recall': 0.875,
            'f1_score': 0.824,
            'accuracy': 0.85,
            'weighted_f1_score': 0.836,
            'balanced_accuracy': 0.862,
            'specificity': 0.833,
            'negative_predictive_value': 0.909,
            'false_positive_rate': 0.167,
            'false_negative_rate': 0.125,
            'class_distribution': {
                'scam': 40,
                'legitimate': 60
            }
        }
        st.info("ℹ️ Showing demo metrics. Start analyzing job listings to see real performance data!")
    
    # Display class distribution
    st.write("### Class Distribution")
    dist_col1, dist_col2 = st.columns(2)
    with dist_col1:
        total = sum(model_metrics['class_distribution'].values())
        if total > 0:
            scam_percent = (model_metrics['class_distribution']['scam'] / total) * 100
            legitimate_percent = (model_metrics['class_distribution']['legitimate'] / total) * 100
            
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Bar(
                x=['Scam', 'Legitimate'],
                y=[scam_percent, legitimate_percent],
                text=[f'{scam_percent:.1f}%', f'{legitimate_percent:.1f}%'],
                textposition='auto',
                marker_color=['#EF553B', '#00CC96']
            ))
            fig_dist.update_layout(
                title="Dataset Distribution",
                yaxis_title="Percentage",
                showlegend=False,
                height=300,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_dist, use_container_width=True)
    
    with dist_col2:
        st.write("#### Imbalance Metrics")
        imbalance_ratio = max(model_metrics['class_distribution'].values()) / (min(model_metrics['class_distribution'].values()) or 1)
        st.metric("Imbalance Ratio", f"{imbalance_ratio:.2f}:1")
        st.write("*A ratio > 1.5:1 indicates class imbalance*")
    
    # Display confusion matrix
    st.write("### Confusion Matrix")
    confusion_matrix = [
        [model_metrics['true_negatives'], model_metrics['false_positives']],
        [model_metrics['false_negatives'], model_metrics['true_positives']]
    ]
    
    fig_matrix = go.Figure(data=go.Heatmap(
        z=confusion_matrix,
        x=['Predicted Legitimate', 'Predicted Scam'],
        y=['Actual Legitimate', 'Actual Scam'],
        text=[[str(val) for val in row] for row in confusion_matrix],
        texttemplate="%{text}",
        textfont={"size": 16},
        colorscale=[[0, '#00CC96'], [1, '#EF553B']],
        showscale=False
    ))
    
    fig_matrix.update_layout(
        title="Confusion Matrix",
        height=400,
        margin=dict(t=30, b=0, l=0, r=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig_matrix, use_container_width=True)
    
    # Display key metrics
    st.write("### Key Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Weighted F1 Score",
            f"{model_metrics['weighted_f1_score']:.3f}",
            help="F1 score that accounts for class imbalance"
        )
    
    with col2:
        st.metric(
            "Balanced Accuracy",
            f"{model_metrics['balanced_accuracy']:.3f}",
            help="Average of recall and specificity"
        )
    
    with col3:
        st.metric(
            "Recall (Sensitivity)",
            f"{model_metrics['recall']:.3f}",
            help="True Positive Rate"
        )
    
    with col4:
        st.metric(
            "Specificity",
            f"{model_metrics['specificity']:.3f}",
            help="True Negative Rate"
        )
    
    # Display detailed metrics comparison
    st.write("### Detailed Metrics Comparison")
    
    metrics_data = {
        'Standard F1': model_metrics['f1_score'],
        'Weighted F1': model_metrics['weighted_f1_score'],
        'Precision': model_metrics['precision'],
        'Recall': model_metrics['recall'],
        'Specificity': model_metrics['specificity'],
        'NPV': model_metrics['negative_predictive_value'],
        'Balanced Acc.': model_metrics['balanced_accuracy']
    }
    
    fig_metrics = go.Figure()
    fig_metrics.add_trace(go.Bar(
        x=list(metrics_data.keys()),
        y=list(metrics_data.values()),
        marker_color='#00CC96'
    ))
    
    fig_metrics.update_layout(
        title="Performance Metrics Comparison",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        showlegend=False,
        height=300,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig_metrics, use_container_width=True)
    
    # Display error analysis
    st.write("### Error Analysis")
    error_col1, error_col2 = st.columns(2)
    
    with error_col1:
        error_rates = {
            'False Positive Rate': model_metrics['false_positive_rate'],
            'False Negative Rate': model_metrics['false_negative_rate']
        }
        
        fig_errors = go.Figure()
        fig_errors.add_trace(go.Bar(
            x=list(error_rates.keys()),
            y=list(error_rates.values()),
            marker_color=['#FFA15A', '#EF553B'],
            text=[f'{rate:.1%}' for rate in error_rates.values()],
            textposition='auto'
        ))
        
        fig_errors.update_layout(
            title="Error Rates",
            yaxis_title="Rate",
            yaxis=dict(range=[0, 1]),
            showlegend=False,
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_errors, use_container_width=True)
    
    with error_col2:
        st.write("#### Error Rate Analysis")
        st.write("""
        - **False Positive Rate**: Legitimate jobs incorrectly marked as scams
        - **False Negative Rate**: Scams incorrectly marked as legitimate
        - **Impact**: In scam detection, false negatives (missing scams) are typically more costly than false positives
        """)
    
    # Add explanation of metrics
    with st.expander("ℹ️ Understanding the Metrics"):
        st.markdown("""
        ### Model Performance Metrics Explained
        
        #### Primary Metrics for Imbalanced Data
        - **Weighted F1 Score**: F1 score that accounts for class imbalance by weighting each class based on its frequency
        - **Balanced Accuracy**: Average of recall and specificity, better than standard accuracy for imbalanced datasets
        
        #### Standard Metrics
        - **Precision**: Ratio of true positive predictions to total positive predictions (TP / (TP + FP))
        - **Recall (Sensitivity)**: Ratio of true positive predictions to total actual positives (TP / (TP + FN))
        - **Specificity**: Ratio of true negative predictions to total actual negatives (TN / (TN + FP))
        - **NPV (Negative Predictive Value)**: Ratio of true negative predictions to total negative predictions
        
        #### Error Metrics
        - **False Positive Rate**: Rate of legitimate jobs incorrectly marked as scams
        - **False Negative Rate**: Rate of scams incorrectly marked as legitimate
        
        #### Class Imbalance
        - **Imbalance Ratio**: Ratio between the majority and minority class
        - A ratio > 1.5:1 indicates significant class imbalance
        
        Where:
        - TP = True Positives (correctly identified scams)
        - TN = True Negatives (correctly identified legitimate jobs)
        - FP = False Positives (legitimate jobs incorrectly marked as scams)
        - FN = False Negatives (scams incorrectly marked as legitimate)
        """)

def display_feedback_stats():
    """Display feedback analytics."""
    st.title("💭 Feedback Analytics")
    
    try:
        # Get feedback data and model performance metrics
        feedback_data = get_feedback_stats()
        model_metrics = get_model_performance_metrics()
        
        if feedback_data['total_feedback'] == 0:
            st.info("📊 No feedback received yet. Start by analyzing some job listings and providing feedback!")
            return
        
        # Create tabs for different analytics views
        tab1, tab2 = st.tabs(["📊 Feedback Overview", "🎯 Model Performance"])
        
        with tab1:
            # Create metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Feedback",
                    feedback_data['total_feedback'],
                    help="Total number of feedback entries received"
                )
            
            with col2:
                correct_rate = feedback_data['feedback_types'].get('✅ Correct', 0) / feedback_data['total_feedback'] * 100
                st.metric(
                    "Correct Predictions",
                    f"{correct_rate:.1f}%",
                    help="Percentage of predictions marked as correct"
                )
            
            with col3:
                partial_rate = feedback_data['feedback_types'].get('⚠️ Partially Correct', 0) / feedback_data['total_feedback'] * 100
                st.metric(
                    "Partially Correct",
                    f"{partial_rate:.1f}%",
                    help="Percentage of predictions marked as partially correct"
                )
            
            with col4:
                incorrect_rate = feedback_data['feedback_types'].get('❌ Incorrect', 0) / feedback_data['total_feedback'] * 100
                st.metric(
                    "Incorrect Predictions",
                    f"{incorrect_rate:.1f}%",
                    help="Percentage of predictions marked as incorrect"
                )
            
            # Create feedback type distribution chart
            st.subheader("Feedback Distribution")
            
            feedback_types = feedback_data['feedback_types']
            if feedback_types:
                fig_dist = go.Figure()
                
                labels = list(feedback_types.keys())
                values = list(feedback_types.values())
                colors = ['#00CC96' if '✅' in label else '#FFA15A' if '⚠️' in label else '#EF553B' for label in labels]
                
                fig_dist.add_trace(go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.4,
                    marker=dict(colors=colors)
                ))
                
                fig_dist.update_layout(
                    showlegend=True,
                    height=400,
                    margin=dict(t=30, b=0, l=0, r=0)
                )
                
                st.plotly_chart(fig_dist, use_container_width=True)
            
            # Display error analysis
            st.subheader("Error Analysis")
            if feedback_data.get('false_positives', 0) > 0 or feedback_data.get('false_negatives', 0) > 0:
                error_col1, error_col2 = st.columns(2)
                
                with error_col1:
                    st.metric(
                        "False Positives",
                        f"{feedback_data['false_positives']:.1f}%",
                        help="Legitimate jobs incorrectly marked as scams"
                    )
                
                with error_col2:
                    st.metric(
                        "False Negatives",
                        f"{feedback_data['false_negatives']:.1f}%",
                        help="Scam jobs incorrectly marked as legitimate"
                    )
            
            # Display recent feedback
            st.subheader("Recent Feedback")
            if feedback_data['recent_feedback']:
                for feedback in feedback_data['recent_feedback']:
                    with st.expander(f"{feedback['feedback_type']} - {feedback['job_title']}", expanded=False):
                        st.write(f"**Feedback Type:** {feedback['feedback_type']}")
                        if feedback['specific_issues']:
                            st.write("**Specific Issues:**")
                            issues = json.loads(feedback['specific_issues']) if isinstance(feedback['specific_issues'], str) else feedback['specific_issues']
                            for issue in issues:
                                st.write(f"- {issue}")
                        if feedback['suggestions']:
                            st.write("**Suggestions:**", feedback['suggestions'])
                        st.write(f"**Date:** {feedback['created_at']}")
            else:
                st.info("No recent feedback available.")
        
        with tab2:
            display_model_metrics_tab(model_metrics)
            
    except Exception as e:
        st.error(f"Error displaying feedback analytics: {str(e)}")
        logger.error(f"Error in display_feedback_stats: {e}")
        st.error("Please try refreshing the page. If the problem persists, contact support.")

def display_batch_analysis():
    """Display batch analysis page."""
    st.title("📊 Batch Analysis")
    
    st.write("Upload multiple job postings for analysis")
    
    # Add CSV requirements section
    with st.expander("ℹ️ CSV File Requirements", expanded=True):
        st.markdown("""
        ### Required CSV Format
        Your CSV file must include the following columns:
        - `job_description` (required): The full job posting text
        - `title` (optional): Job title
        - `location` (optional): Job location
        - `company_profile` (optional): Company information
        
        ### Example CSV Format:
        ```csv
        title,job_description,location,company_profile
        Software Engineer,We are looking for a talented developer...,New York,Tech company founded in...
        Data Analyst,Seeking an experienced data analyst...,Remote,Leading data analytics firm...
        ```
        
        ### File Requirements:
        - File format: CSV (Comma Separated Values)
        - Maximum file size: 200MB
        - Encoding: UTF-8
        - First row must be header row
        - At least one job posting required
        
        ### Tips:
        - Export your spreadsheet as CSV before uploading
        - Make sure text fields are properly quoted if they contain commas
        - Remove any special formatting or merged cells before exporting
        """)
    
    # File uploader with better instructions
    st.write("### Upload CSV file")
    uploaded_file = st.file_uploader(
        "Drag and drop file here",
        type="csv",
        help="Upload a CSV file containing job postings. Click 'CSV File Requirements' above for format details."
    )
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Check for required column
            if 'job_description' not in df.columns:
                st.error("❌ CSV must contain a 'job_description' column")
                st.stop()
            
            # Show preview of uploaded data
            st.write("### Preview of uploaded data")
            st.write(f"Total job postings: {len(df)}")
            
            # Show sample of the data
            with st.expander("📋 View Data Preview"):
                st.dataframe(
                    df.head(5),
                    use_container_width=True
                )
            
            # Add analysis button
            if st.button("🔍 Analyze Job Postings"):
                with st.spinner("Analyzing job postings..."):
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, row in df.iterrows():
                        # Prepare job listing data
                        listing = {
                            'title': row.get('title', ''),
                            'description': row['job_description'],
                            'location': row.get('location', ''),
                            'company_profile': row.get('company_profile', '')
                        }
                        
                        # Analyze the listing
                        result = analyze_job_listing(listing)
                        if result:
                            results.append({
                                'Title': listing['title'],
                                'Location': listing['location'],
                                'Prediction': 'Scam' if result['probability'] > 0.5 else 'Legitimate',
                                'Confidence': f"{result['probability']*100:.1f}%",
                                'Risk Factors': len(result.get('risk_factors', []))
                            })
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(df))
                    
                    # Show results
                    if results:
                        results_df = pd.DataFrame(results)
                        
                        # Summary metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            total_scams = sum(1 for r in results if r['Prediction'] == 'Scam')
                            st.metric(
                                "Scams Detected",
                                f"{total_scams}",
                                f"{total_scams/len(results)*100:.1f}%"
                            )
                        
                        with col2:
                            avg_confidence = sum(float(r['Confidence'].rstrip('%')) for r in results) / len(results)
                            st.metric(
                                "Average Confidence",
                                f"{avg_confidence:.1f}%"
                            )
                        
                        with col3:
                            high_risk = sum(1 for r in results if float(r['Confidence'].rstrip('%')) > 90)
                            st.metric(
                                "High Risk Postings",
                                f"{high_risk}",
                                f"{high_risk/len(results)*100:.1f}%"
                            )
                        
                        # Results table
                        st.write("### Analysis Results")
                        st.dataframe(
                            results_df.style\
                                .apply(lambda x: ['color: #EF553B' if v == 'Scam' else 'color: #00CC96' for v in x], subset=['Prediction'])\
                                .format({'Confidence': '{:}'}),
                            use_container_width=True
                        )
                        
                        # Export results button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name=f"job_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error("No results generated. Please check your input data and try again.")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.error("Please check that your file matches the required format and try again.")

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
        .demo-button {
            margin-top: 1rem;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="auth-title">🔐 Authentication</h1>', unsafe_allow_html=True)
    
    # Add Demo User button at the top
    if st.button("👋 Try Demo Account", type="primary"):
        # Create or get demo user
        demo_email = "demo@example.com"
        demo_password = "demo123"
        try:
            # Try to register demo user (will fail if already exists)
            register_user(demo_email, demo_password)
        except:
            pass
        # Log in as demo user
        user_id = verify_user(demo_email, demo_password)
        if user_id:
            st.session_state.user_id = user_id
            st.session_state.authenticated = True
            st.session_state.is_demo = True
            st.success("Logged in as Demo User! Redirecting...")
            st.rerun()
    
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
                        st.session_state.is_demo = False
                        st.success("Login successful! Redirecting...")
                        st.rerun()
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
                            st.rerun()
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
    st.rerun()

def main():
    """Main function to run the Streamlit app."""
    initialize_session_state()
    
    if not st.session_state.authenticated:
        login_signup_page()
        return
    
    display_header()
    
    # Add logout button and user info to sidebar
    st.sidebar.markdown("### 👤 User Account")
    if st.session_state.get('is_demo'):
        st.sidebar.info("🎯 You are using a Demo Account")
    if st.sidebar.button("Logout"):
        logout()
        return
    
    # Add a welcome message for demo users
    if st.session_state.get('is_demo'):
        st.info("""
        👋 Welcome to the Demo! Try these features:
        1. Analyze a job posting for potential scams
        2. View analysis history and statistics
        3. Provide feedback on analysis results
        4. Check model performance metrics
        """)
    
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
        
        .nav-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: #ECEFF4;
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #81A1C1;
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
        display_history_page(st.session_state.user_id)
    elif "Model Performance" in page:
        display_model_performance()
    else:
        display_feedback_stats()

if __name__ == "__main__":
    main() 