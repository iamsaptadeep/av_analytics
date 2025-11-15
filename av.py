import streamlit as st

# Set page config - this must be the first Streamlit command
st.set_page_config(
    page_title="EchoMind",
    layout="wide",
    page_icon="",
    initial_sidebar_state="expanded"
)

def set_custom_theme():
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary: #2563EB;
        --primary-dark: #1E40AF;
        --primary-light: #3B82F6;
        --background: #FFFFFF;
        --card-bg: #F8FAFC;
        --text: #1E293B;
        --text-light: #64748B;
        --border: #E2E8F0;
    }
    
    /* Main app background */
    .stApp {
        background-color: var(--background);
    }
    
    /* Premium header styling */
    .premium-header {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        padding: 4rem 2rem;
        border-radius: 0 0 20px 20px;
        margin-bottom: 3rem;
        text-align: center;
    }
    
    .premium-header h1 {
        color: white !important;
        font-size: 3rem !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
    }
    
    .premium-header p {
        color: rgba(255,255,255,0.9) !important;
        font-size: 1.2rem !important;
        max-width: 600px;
        margin: 0 auto !important;
    }
    
    /* Feature highlight cards */
    .feature-highlight {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid var(--border);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .feature-highlight:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(37, 99, 235, 0.15);
    }
    
    /* Navigation prompt */
    .nav-prompt {
        background: var(--card-bg);
        border: 2px dashed var(--primary-light);
        border-radius: 12px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 2rem 0;
    }
    
    /* Stats container */
    .stats-container {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .stat-item {
        text-align: center;
        padding: 1rem;
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary-dark);
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        color: var(--text-light);
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Apply the theme
set_custom_theme()

# Premium Header Section
st.markdown("""
<div class="premium-header">
    <h1>EchoMind</h1>
    <p>Professional media analysis tools for comprehensive audio and video insights</p>
</div>
""", unsafe_allow_html=True)

# Stats Overview
st.markdown("""
<div class="stats-container">
    <div class="stat-item">
        <div class="stat-number">4</div>
        <div class="stat-label">Analysis Tools</div>
    </div>
    <div class="stat-item">
        <div class="stat-number">10+</div>
        <div class="stat-label">Supported Formats</div>
    </div>
    <div class="stat-item">
        <div class="stat-number">Real-time</div>
        <div class="stat-label">Processing</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation Prompt
st.markdown("""
<div class="nav-prompt">
    <h3 style="color: var(--primary-dark); margin-bottom: 1rem;">Get Started</h3>
    <p style="color: var(--text-light); margin-bottom: 2rem; font-size: 1.1rem;">
        Access professional media analysis tools through the sidebar navigation
    </p>
    <div style="color: var(--primary); font-weight: 600; font-size: 1.2rem;">
        ← Select a tool from the sidebar to begin analysis
    </div>
</div>
""", unsafe_allow_html=True)

# Feature Highlights
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-highlight">
        <h4 style="color: var(--primary-dark); margin-bottom: 1rem;">Audio Analysis</h4>
        <p style="color: var(--text-light); margin: 0; line-height: 1.6;">
            Comprehensive audio file processing with transcription and sentiment analysis. 
            Extract meaningful insights from speech content with detailed metrics.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-highlight">
        <h4 style="color: var(--primary-dark); margin-bottom: 1rem;">Audio Comparison</h4>
        <p style="color: var(--text-light); margin: 0; line-height: 1.6;">
            Compare multiple audio files with advanced visualization tools. 
            Identify differences and similarities with precision analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-highlight">
        <h4 style="color: var(--primary-dark); margin-bottom: 1rem;">Video Analysis</h4>
        <p style="color: var(--text-light); margin: 0; line-height: 1.6;">
            Analyze YouTube video content with engagement metrics and comment insights. 
            Understand audience sentiment and content performance.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-highlight">
        <h4 style="color: var(--primary-dark); margin-bottom: 1rem;">Comment Analysis</h4>
        <p style="color: var(--text-light); margin: 0; line-height: 1.6;">
            Process user comments with advanced sentiment scoring and topic modeling. 
            Gain valuable insights from audience feedback and discussions.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Bottom CTA
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem;">
    <p style="color: var(--text-light); font-size: 1.1rem;">
        Ready to analyze your media? <strong>Select a tool from the sidebar</strong> to get started with professional-grade analysis.
    </p>
</div>
""", unsafe_allow_html=True)

