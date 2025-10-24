import streamlit as st
import pandas as pd
from modules.youtube_comments import run_analysis

st.set_page_config(page_title="YouTube Analysis", layout="wide")
st.title("YouTube Comment Sentiment Analysis")

st.markdown("""
Analyze sentiment of YouTube comments using the YouTube Data API.

**Requirements:**
- YouTube Data API key (set in Streamlit secrets)
- Public YouTube video (comments enabled)
""")

# API key check
try:
    api_key = st.secrets.get("YOUTUBE_API_KEY")
    if not api_key:
        st.error("""
        ❌ YouTube API key not found!
        
        Please add your YouTube Data API key to Streamlit secrets:
        
        **Local testing**: Create `.streamlit/secrets.toml` with:
        ```toml
        YOUTUBE_API_KEY = "your_api_key_here"
        ```
        
        **Streamlit Cloud**: Go to Settings → Secrets and add:
        ```toml
        YOUTUBE_API_KEY = "your_api_key_here"  
        ```
        """)
        st.stop()
except Exception:
    st.error("Secrets not accessible - check Streamlit configuration")
    st.stop()

# Video input
st.header("1. Video Selection")
video_input = st.text_input(
    "YouTube URL or Video ID",
    placeholder="https://www.youtube.com/watch?v=... or dQw4w9WgXcQ",
    help="Paste full YouTube URL or just the video ID"
)

# Analysis parameters
st.header("2. Analysis Settings")
col1, col2 = st.columns(2)
max_comments = col1.number_input(
    "Max comments to analyze",
    min_value=50,
    max_value=5000,
    value=500,
    step=50,
    help="Higher values take longer but provide better insights"
)

include_replies = col2.checkbox(
    "Include comment replies", 
    value=True,
    help="Analyze replies to top-level comments"
)

if st.button("Analyze Comments", type="primary"):
    if not video_input.strip():
        st.error("❌ Please enter a YouTube URL or Video ID")
    else:
        with st.spinner("🔄 Fetching and analyzing comments... This may take a while for videos with many comments"):
            try:
                # Run analysis
                df, kpi, pos_wc_bytes, neg_wc_bytes = run_analysis(
                    video_input, 
                    max_comments=int(max_comments)
                )
                
                # Display results
                st.success(f"✅ Analysis complete! Processed {len(df)} comments")
                
                # Key metrics
                st.header("📈 Key Metrics")
                kpi_cols = st.columns(4)
                kpi_cols[0].metric("Total Comments", kpi["n_comments"])
                kpi_cols[1].metric("Positive", f"{kpi['pct_positive']:.1f}%")
                kpi_cols[2].metric("Neutral", f"{kpi['pct_neutral']:.1f}%") 
                kpi_cols[3].metric("Negative", f"{kpi['pct_negative']:.1f}%")
                
                st.metric("Overall Sentiment Score", f"{kpi['avg_compound']:.3f}",
                         help="Compound score: -1 (most negative) to +1 (most positive)")
                
                # Sentiment distribution
                st.header("😊 Sentiment Distribution")
                sentiment_data = {
                    "Positive": kpi['pct_positive'],
                    "Neutral": kpi['pct_neutral'], 
                    "Negative": kpi['pct_negative']
                }
                st.bar_chart(sentiment_data)
                
                # Word clouds
                st.header("☁️ Word Clouds")
                if pos_wc_bytes or neg_wc_bytes:
                    wc_col1, wc_col2 = st.columns(2)
                    
                    if pos_wc_bytes:
                        wc_col1.image(pos_wc_bytes, use_column_width=True,
                                    caption="Most frequent words in Positive comments")
                    else:
                        wc_col1.info("No positive comments to generate word cloud")
                        
                    if neg_wc_bytes:
                        wc_col2.image(neg_wc_bytes, use_column_width=True,
                                    caption="Most frequent words in Negative comments")
                    else:
                        wc_col2.info("No negative comments to generate word cloud")
                else:
                    st.info("Not enough text data to generate word clouds")
                
                # Comments table
                st.header("💬 Sample Comments")
                display_df = df[[
                    'published', 'author', 'clean_comment', 
                    'compound', 'sentiment_vader'
                ]].sort_values('published', ascending=False).head(100)
                
                st.dataframe(
                    display_df,
                    use_container_width='stretch',
                    column_config={
                        "published": "Date",
                        "author": "Author", 
                        "clean_comment": "Comment",
                        "compound": "Sentiment Score",
                        "sentiment_vader": "Sentiment"
                    }
                )
                
                # Download option
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "💾 Download Full Results (CSV)",
                    data=csv_data,
                    file_name=f"youtube_analysis_{kpi['video_id']}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.info("""
                💡 Common issues:
                - Invalid YouTube URL/ID
                - Video has disabled comments
                - YouTube API quota exceeded
                - Video is private or unavailable
                """)



                