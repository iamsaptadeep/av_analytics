# pages/Audio_Analysis.py
import streamlit as st
from modules.audio_transcription import transcribe_from_bytes
from modules.sentiment_analysis import analyze_sentiment_chunked, get_sentiment_breakdown
import hashlib
import time

# Page config
st.set_page_config(page_title="Audio Transcription & Sentiment", layout="wide")
st.title("Audio to Text Analysis")

# --- Helper: compute audio hash so we can cache transcription results ---
def _hash_audio(audio_bytes: bytes) -> str:
    return hashlib.sha256(audio_bytes).hexdigest() if audio_bytes else ""

# --- Model selection and analysis configuration ---
st.sidebar.header("Analysis Configuration")
model_name = st.sidebar.selectbox(
    "Select Whisper Model",
    ["tiny", "base", "small", "medium", "large"],
    index=1,
    help="Smaller models are faster but less accurate.",
)
chunk_size = st.sidebar.slider(
    "Chunk size for sentiment analysis",
    min_value=1,
    max_value=5,
    value=2,
    help="Number of sentences per analysis chunk.",
)
show_debug = st.sidebar.checkbox("Show Debug Info", value=False)

# --- Audio Input Section ---
st.header("Provide Audio Input")
col1, col2 = st.columns(2)

uploaded_file = col1.file_uploader(
    "Upload audio file",
    type=["wav", "mp3", "m4a", "ogg", "flac"],
    help="Supported formats: WAV, MP3, M4A, OGG, FLAC",
)

recorded_audio = None
try:
    from streamlit_audiorecorder import audio_recorder
    with col2:
        st.write("Or record using your microphone:")
        recorded_audio = audio_recorder(
            pause_threshold=2.0,
            sample_rate=44100,
            text="Click to Record",
            recording_color="#e8b62c",
            neutral_color="#6aa36f",
        )
        if recorded_audio:
            st.audio(recorded_audio, format="audio/wav")
except ImportError:
    col2.info("Microphone recording requires: pip install streamlit-audiorecorder")

# --- Determine which source to use ---
audio_bytes = None
source_label = ""
if uploaded_file:
    audio_bytes = uploaded_file.read()
    source_label = f"Uploaded file: {uploaded_file.name}"
    st.success(f"✅ {source_label} ({len(audio_bytes)} bytes)")
elif recorded_audio and len(recorded_audio) > 0:
    audio_bytes = recorded_audio
    source_label = "Microphone recording"
    st.success(f"✅ {source_label} captured")
else:
    st.info("Upload an audio file or record to begin.")

# --- Caching mechanism for transcription ---
@st.cache_data(show_spinner=False)
def _cached_transcription(audio_hash: str, audio_bytes: bytes, model_name: str):
    start = time.time()
    transcript = transcribe_from_bytes(audio_bytes, model_name=model_name)
    elapsed = time.time() - start
    return transcript, elapsed

# --- Run analysis ---
if st.button("Transcribe and Analyze", type="primary"):
    if not audio_bytes:
        st.error("Please provide audio input first.")
    else:
        audio_hash = _hash_audio(audio_bytes)
        with st.spinner("Processing audio..."):
            transcript, load_time = _cached_transcription(audio_hash, audio_bytes, model_name)

        if transcript and transcript.strip():
            st.subheader("Transcription Result")
            st.text_area("Transcript", transcript, height=180)

            if show_debug:
                st.caption(f"Model: {model_name} | Transcription time: {load_time:.2f}s")

            # --- Sentiment Analysis ---
            st.subheader("Sentiment Analysis")
            with st.spinner("Analyzing sentiment..."):
                sentiment_result = analyze_sentiment_chunked(transcript, chunk_size=chunk_size)

            if "error" in sentiment_result:
                st.error(f"Sentiment analysis failed: {sentiment_result['error']}")
            else:
                cols = st.columns(4)
                cols[0].metric("Overall Sentiment", sentiment_result["sentiment"])
                cols[1].metric("Compound Score", f"{sentiment_result['compound_score']:.3f}")
                cols[2].metric("Positive", f"{sentiment_result['positive']:.3f}")
                cols[3].metric("Negative", f"{sentiment_result['negative']:.3f}")

                dist_cols = st.columns(3)
                dist_cols[0].metric(
                    "Positive Chunks",
                    sentiment_result["sentiment_distribution"]["Positive"],
                    f"{sentiment_result['sentiment_percentages']['positive']:.1f}%",
                )
                dist_cols[1].metric(
                    "Neutral Chunks",
                    sentiment_result["sentiment_distribution"]["Neutral"],
                    f"{sentiment_result['sentiment_percentages']['neutral']:.1f}%",
                )
                dist_cols[2].metric(
                    "Negative Chunks",
                    sentiment_result["sentiment_distribution"]["Negative"],
                    f"{sentiment_result['sentiment_percentages']['negative']:.1f}%",
                )

                st.write("### Weighted Sentiment Scores")
                st.progress(sentiment_result["positive"], text=f"Positive: {sentiment_result['positive']:.3f}")
                st.progress(sentiment_result["neutral"], text=f"Neutral: {sentiment_result['neutral']:.3f}")
                st.progress(sentiment_result["negative"], text=f"Negative: {sentiment_result['negative']:.3f}")

                # --- Chunk breakdown ---
                st.write("### Detailed Chunk Analysis")
                breakdown = get_sentiment_breakdown(sentiment_result["chunk_analysis"])

                tab1, tab2, tab3 = st.tabs(["Positive", "Negative", "Neutral"])
                def render_chunks(tab_container, chunks, label):
                    if not chunks:
                        tab_container.info(f"No {label.lower()} chunks found.")
                        return
                    for c in chunks:
                        with tab_container.expander(f"Chunk {c['chunk_number']} (Score: {c['compound_score']})"):
                            st.write(f"**Text:** {c['text']}")
                            st.write(
                                f"**Scores:** Pos {c['positive']} | Neu {c['neutral']} | Neg {c['negative']}"
                            )

                with tab1:
                    render_chunks(tab1, breakdown["positive_chunks"], "Positive")
                with tab2:
                    render_chunks(tab2, breakdown["negative_chunks"], "Negative")
                with tab3:
                    render_chunks(tab3, breakdown["neutral_chunks"], "Neutral")

                # --- Key Findings ---
                st.write("### Key Findings")
                colA, colB = st.columns(2)
                if breakdown["strongest_positive"]:
                    with colA:
                        st.success("Most Positive Phrase")
                        st.write(f"*{breakdown['strongest_positive']['text']}*")
                        st.caption(f"Score: {breakdown['strongest_positive']['compound_score']}")
                if breakdown["strongest_negative"]:
                    with colB:
                        st.error("Most Negative Phrase")
                        st.write(f"*{breakdown['strongest_negative']['text']}*")
                        st.caption(f"Score: {breakdown['strongest_negative']['compound_score']}")
        else:
            st.warning("No speech detected or transcription failed.")

# --- Explanation Section ---
with st.expander("ℹ️ About Chunk-Based Sentiment Analysis"):
    st.markdown(
        """
        **Chunk-based analysis** breaks long transcripts into smaller parts for more accurate sentiment detection.  
        It helps identify emotional highs and lows across the speech, rather than averaging everything together.
        """
    )