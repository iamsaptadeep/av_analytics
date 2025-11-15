# pages/Video_Analysis.py
import streamlit as st
from modules.youtube_comments import (
    fetch_youtube_comments,
    analyze_comments_sentiment,
)
import pandas as pd
from io import BytesIO

# Optional visualization packages
try:
    from wordcloud import WordCloud, STOPWORDS
    WC_AVAILABLE = True
except Exception:
    WC_AVAILABLE = False

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False


st.set_page_config(page_title="YouTube Comment Analysis", layout="wide")
st.title("YouTube Comment Sentiment Analysis")

# ---------------------------------------------------------------------
# Sidebar Settings
# ---------------------------------------------------------------------
st.sidebar.header("Settings")

max_comments = st.sidebar.slider("Max comments (incl. replies)", 100, 5000, 1000, step=100)
method = st.sidebar.selectbox("Sentiment method", ["vader", "transformer"], index=0)
transformer_model_name = st.sidebar.text_input(
    "Transformer sentiment model", "distilbert-base-uncased-finetuned-sst-2-english"
)

include_replies = st.sidebar.checkbox("Include replies", True)
use_sarcasm_model = st.sidebar.checkbox("Enable sarcasm detection (Transformer)", True)
sarcasm_model_name = st.sidebar.text_input(
    "Sarcasm model", "cardiffnlp/twitter-roberta-base-irony"
)
sarcasm_conf_threshold = st.sidebar.slider("Sarcasm confidence threshold", 0.5, 0.95, 0.6)
invert_on_sarcasm = st.sidebar.checkbox("Invert sentiment if sarcasm detected", True)

wordcloud_max_words = st.sidebar.slider("Wordcloud max words", 50, 500, 200, 50)



# ---------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------
video_url = st.text_input("YouTube video URL or ID", placeholder="https://www.youtube.com/watch?v=...")

if st.button("Analyze Comments", type="primary"):
    if not video_url.strip():
        st.error("Please provide a valid YouTube video URL or ID.")
        st.stop()

    # -------------------- Fetch Comments --------------------
    with st.spinner("Fetching comments..."):
        try:
            comments = fetch_youtube_comments(video_url, max_comments=max_comments, include_replies=include_replies)
            st.success(f"Fetched {len(comments)} comments.")
        except Exception as e:
            st.error(f"Error fetching comments: {e}")
            st.stop()

    if not comments:
        st.warning("No comments retrieved.")
        st.stop()

    # -------------------- Sentiment & Sarcasm Analysis --------------------
    with st.spinner("Analyzing sentiments and sarcasm..."):
        try:
            analyzed = analyze_comments_sentiment(
                comments,
                method=method,
                transformer_model=(transformer_model_name if method == "transformer" else None),
                use_sarcasm_model=use_sarcasm_model,
                sarcasm_model_name=sarcasm_model_name,
                sarcasm_confidence_threshold=sarcasm_conf_threshold,
                invert_on_sarcasm=invert_on_sarcasm,
            )
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

    total = len(analyzed)
    pos = sum(1 for c in analyzed if c.get("_sentiment_label") == "Positive")
    neg = sum(1 for c in analyzed if c.get("_sentiment_label") == "Negative")
    neu = sum(1 for c in analyzed if c.get("_sentiment_label") == "Neutral")
    sar_true = sum(1 for c in analyzed if c.get("_sarcasm"))

    # ---------------------------------------------------------------------
    # Sentiment Distribution Bar Chart
    # ---------------------------------------------------------------------
    st.subheader("Sentiment Distribution")

    if PLOTLY_AVAILABLE:
        df_chart = pd.DataFrame({
            "Sentiment": ["Positive", "Neutral", "Negative"],
            "Count": [pos, neu, neg]
        })
        fig = px.bar(
            df_chart,
            x="Sentiment",
            y="Count",
            color="Sentiment",
            color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"},
            text="Count",
            title="Sentiment Distribution of Comments"
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            xaxis_title="Sentiment Category",
            yaxis_title="Number of Comments",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write(f"Positive: {pos}, Neutral: {neu}, Negative: {neg}")

    col1, col2 = st.columns(2)
    col1.metric("Total Comments", total)
    col2.metric("Sarcasm Detected", sar_true)

    # ---------------------------------------------------------------------
    # Word Clouds (side-by-side, memory safe)
    # ---------------------------------------------------------------------
    if WC_AVAILABLE:
        st.subheader("☁️ Word Clouds")

        pos_text = " ".join(c.get("text", "") for c in analyzed if c.get("_sentiment_label") == "Positive")
        neg_text = " ".join(c.get("text", "") for c in analyzed if c.get("_sentiment_label") == "Negative")

        stopwords = set(STOPWORDS)
        stopwords.update(["https", "http", "amp"])

        col1, col2 = st.columns(2)

        if pos_text.strip():
            with col1:
                st.markdown("**Positive Comments Word Cloud**")
                wc_pos = WordCloud(
                    width=800, height=400, background_color="white",
                    colormap="Greens", stopwords=stopwords, max_words=wordcloud_max_words
                ).generate(pos_text)
                buf = BytesIO()
                wc_pos.to_image().save(buf, format="PNG")
                st.image(buf.getvalue(), use_container_width=True)

        if neg_text.strip():
            with col2:
                st.markdown("**Negative Comments Word Cloud**")
                wc_neg = WordCloud(
                    width=800, height=400, background_color="white",
                    colormap="Reds", stopwords=stopwords, max_words=wordcloud_max_words
                ).generate(neg_text)
                buf = BytesIO()
                wc_neg.to_image().save(buf, format="PNG")
                st.image(buf.getvalue(), use_container_width=True)
    else:
        st.info("Install 'wordcloud' to see text cloud visuals.")

    # ---------------------------------------------------------------------
    # Top Comments Tables (visible actual comments)
    # ---------------------------------------------------------------------
    st.subheader("Top 10 Positive and Negative Comments")

    df = pd.DataFrame(analyzed)

    df_display = pd.DataFrame({
        "Author": df["author"],
        "Likes": df["like_count"],
        "Sentiment": df["_sentiment_label"],
        "Sentiment Score": df["_sentiment_score"],
        "Sarcasm Score": df.get("_sarcasm_score", 0),
        "Comment": df["text"],
    })

    # Split by sentiment
    top_pos = df_display[df_display["Sentiment"] == "Positive"].sort_values("Likes", ascending=False).head(10)
    top_neg = df_display[df_display["Sentiment"] == "Negative"].sort_values("Likes", ascending=False).head(10)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 👍 Top 10 Positive Comments")
        st.data_editor(
            top_pos[["Author", "Likes", "Sentiment Score", "Sarcasm Score", "Comment"]],
            use_container_width=True,
            hide_index=True,
            disabled=True,
            column_config={
                "Comment": st.column_config.TextColumn("Comment", width="large")
            },
        )

    with col2:
        st.markdown("#### 👎 Top 10 Negative Comments")
        st.data_editor(
            top_neg[["Author", "Likes", "Sentiment Score", "Sarcasm Score", "Comment"]],
            use_container_width=True,
            hide_index=True,
            disabled=True,
            column_config={
                "Comment": st.column_config.TextColumn("Comment", width="large")
            },
        )

    st.caption("Each table shows top comments by like count, with sarcasm probabilities and full visible text.")

    st.success("✅ YouTube comment analysis complete.")
