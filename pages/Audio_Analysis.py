import streamlit as st
from modules.audio_transcription import transcribe_from_bytes
from modules.sentiment_analysis import analyze_sentiment_chunked, get_sentiment_breakdown

st.set_page_config(page_title="Audio Transcription", layout="wide")
st.title("Audio Transcription and Sentiment Analysis")

# Mic recording setup
try:
    from streamlit_audiorecorder import audio_recorder
    HAS_RECORDER = True
except ImportError:
    HAS_RECORDER = False
    st.warning("streamlit-audiorecorder not installed. Run: pip install streamlit-audiorecorder")

# Audio input methods
st.header("Provide Audio Input")
col1, col2 = st.columns(2)

uploaded_file = col1.file_uploader(
    "Upload audio file", 
    type=["wav", "mp3", "m4a", "ogg"],
    help="Supported formats: WAV, MP3, M4A, OGG"
)

recorded_audio = None
if HAS_RECORDER:
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
else:
    col2.info("Microphone recording requires streamlit-audiorecorder package")

# Determine which audio to use
audio_bytes = None
if uploaded_file:
    audio_bytes = uploaded_file.read()
    st.success(f"Uploaded: {uploaded_file.name} ({len(audio_bytes)} bytes)")
    
    # Show format info
    if audio_bytes[:4] == b'RIFF':
        st.info("WAV format detected - optimal compatibility")
    elif audio_bytes[:3] == b'ID3':
        st.info("MP3 format detected")
    elif audio_bytes[:4] == b'fLaC':
        st.info("FLAC format detected")
    else:
        st.info("Unknown format - attempting transcription")
        
elif recorded_audio is not None and len(recorded_audio) > 0:
    audio_bytes = recorded_audio
    st.success("Recording captured - ready to transcribe")

# Sentiment analysis configuration
st.header("Analysis Configuration")
chunk_size = st.slider(
    "Chunk size for sentiment analysis",
    min_value=1,
    max_value=5,
    value=2,
    help="Number of sentences per analysis chunk. Smaller chunks provide more granular analysis."
)

# Transcription and analysis
if st.button("Transcribe and Analyze", type="primary"):
    if not audio_bytes:
        st.error("Please provide audio via upload or recording")
    else:
        with st.spinner("Processing audio..."):
            try:
                transcript = transcribe_from_bytes(audio_bytes)
                
                if transcript and transcript.strip():
                    st.subheader("Transcription Result")
                    st.text_area("Transcript", transcript, height=150)
                    
                    st.subheader("Sentiment Analysis")
                    sentiment_result = analyze_sentiment_chunked(transcript, chunk_size=chunk_size)
                    
                    if "error" not in sentiment_result:
                        # Display overall sentiment metrics
                        st.write("### Overall Analysis")
                        cols = st.columns(4)
                        cols[0].metric("Overall Sentiment", sentiment_result["sentiment"])
                        cols[1].metric("Compound Score", f"{sentiment_result['compound_score']:.3f}")
                        cols[2].metric("Positive Score", f"{sentiment_result['positive']:.3f}")
                        cols[3].metric("Negative Score", f"{sentiment_result['negative']:.3f}")
                        
                        # Sentiment distribution
                        st.write("### Sentiment Distribution")
                        dist_cols = st.columns(3)
                        dist_cols[0].metric(
                            "Positive Chunks", 
                            f"{sentiment_result['sentiment_distribution']['Positive']}",
                            f"{sentiment_result['sentiment_percentages']['positive']:.1f}%"
                        )
                        dist_cols[1].metric(
                            "Neutral Chunks", 
                            f"{sentiment_result['sentiment_distribution']['Neutral']}",
                            f"{sentiment_result['sentiment_percentages']['neutral']:.1f}%"
                        )
                        dist_cols[2].metric(
                            "Negative Chunks", 
                            f"{sentiment_result['sentiment_distribution']['Negative']}",
                            f"{sentiment_result['sentiment_percentages']['negative']:.1f}%"
                        )
                        
                        # Progress bars for sentiment scores
                        st.write("### Weighted Sentiment Scores")
                        st.progress(sentiment_result["positive"], text=f"Positive: {sentiment_result['positive']:.3f}")
                        st.progress(sentiment_result["neutral"], text=f"Neutral: {sentiment_result['neutral']:.3f}") 
                        st.progress(sentiment_result["negative"], text=f"Negative: {sentiment_result['negative']:.3f}")
                        
                        # Detailed chunk analysis
                        st.write("### Detailed Chunk Analysis")
                        breakdown = get_sentiment_breakdown(sentiment_result["chunk_analysis"])
                        
                        # Display chunks in expandable sections by sentiment
                        tab1, tab2, tab3 = st.tabs(["Positive Chunks", "Negative Chunks", "Neutral Chunks"])
                        
                        with tab1:
                            if breakdown["positive_chunks"]:
                                for chunk in breakdown["positive_chunks"]:
                                    with st.expander(f"Chunk {chunk['chunk_number']} (Score: {chunk['compound_score']})"):
                                        st.write(f"**Text:** {chunk['text']}")
                                        st.write(f"**Scores:** Positive: {chunk['positive']} | Neutral: {chunk['neutral']} | Negative: {chunk['negative']}")
                            else:
                                st.info("No positive chunks found")
                        
                        with tab2:
                            if breakdown["negative_chunks"]:
                                for chunk in breakdown["negative_chunks"]:
                                    with st.expander(f"Chunk {chunk['chunk_number']} (Score: {chunk['compound_score']})"):
                                        st.write(f"**Text:** {chunk['text']}")
                                        st.write(f"**Scores:** Positive: {chunk['positive']} | Neutral: {chunk['neutral']} | Negative: {chunk['negative']}")
                            else:
                                st.info("No negative chunks found")
                        
                        with tab3:
                            if breakdown["neutral_chunks"]:
                                for chunk in breakdown["neutral_chunks"]:
                                    with st.expander(f"Chunk {chunk['chunk_number']} (Score: {chunk['compound_score']})"):
                                        st.write(f"**Text:** {chunk['text']}")
                                        st.write(f"**Scores:** Positive: {chunk['positive']} | Neutral: {chunk['neutral']} | Negative: {chunk['negative']}")
                            else:
                                st.info("No neutral chunks found")
                        
                        # Strongest sentiments
                        st.write("### Key Findings")
                        strong_col1, strong_col2 = st.columns(2)
                        
                        with strong_col1:
                            if breakdown["strongest_positive"]:
                                st.success("**Most Positive Phrase**")
                                st.write(f"*{breakdown['strongest_positive']['text']}*")
                                st.write(f"Score: {breakdown['strongest_positive']['compound_score']}")
                        
                        with strong_col2:
                            if breakdown["strongest_negative"]:
                                st.error("**Most Negative Phrase**")
                                st.write(f"*{breakdown['strongest_negative']['text']}*")
                                st.write(f"Score: {breakdown['strongest_negative']['compound_score']}")
                        
                    else:
                        st.error(f"Sentiment analysis failed: {sentiment_result['error']}")
                else:
                    st.warning("No speech detected in audio")
                    
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
                st.info("Try a different audio file or check if the audio contains clear speech")

# Information about the analysis method
with st.expander("About Chunk-Based Sentiment Analysis"):
    st.markdown("""
    **Why chunk-based analysis?**
    
    Traditional sentiment analysis often averages scores across entire text, which can dilute emotional content.
    
    **Chunk-based approach:**
    - Breaks text into smaller segments (sentences or word groups)
    - Analyzes each segment independently
    - Provides more granular sentiment detection
    - Identifies emotional highlights within the text
    - Reduces neutral bias from averaging
    
    **Benefits:**
    - Better detection of mixed emotions
    - Identifies specific positive/negative phrases
    - More accurate overall sentiment classification
    - Detailed breakdown of emotional content
    """)