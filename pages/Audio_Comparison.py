import streamlit as st
from modules.audio_comparison import compare_audio_bytes, compute_similarity
import numpy as np
from scipy.spatial.distance import cosine

st.set_page_config(page_title="Audio Comparison", layout="wide")
st.title("Audio Comparison Analysis")

try:
    from streamlit_audiorecorder import audio_recorder
    HAS_RECORDER = True
except ImportError:
    HAS_RECORDER = False
    st.warning("streamlit-audiorecorder not installed. Run: pip install streamlit-audiorecorder")

st.header("Provide Audio Files for Comparison")

# Reference Audio
st.subheader("Reference Audio")
ref_col1, ref_col2 = st.columns(2)

ref_choice = ref_col1.radio(
    "Reference source",
    ["Sample Audio", "Upload Reference"],
    key="ref_source"
)

ref_bytes = None
if ref_choice == "Sample Audio":
    try:
        with open("assets/sample_media/hello.wav", "rb") as f:
            ref_bytes = f.read()
        ref_col2.success("Sample audio loaded: reference recording")
    except FileNotFoundError:
        ref_col2.warning("Sample audio not found. Please upload a reference file.")
        
elif ref_choice == "Upload Reference":
    ref_file = ref_col2.file_uploader(
        "Upload reference audio", 
        type=["wav", "mp3", "m4a", "ogg"],
        key="ref_upload"
    )
    if ref_file:
        ref_bytes = ref_file.read()
        ref_col2.success(f"Reference uploaded: {ref_file.name}")

# Target Audio  
st.subheader("Target Audio")
tgt_col1, tgt_col2 = st.columns(2)

tgt_choice = tgt_col1.radio(
    "Target source",
    ["Upload Target", "Record Target"] if HAS_RECORDER else ["Upload Target"],
    key="tgt_source"
)

tgt_bytes = None
if tgt_choice == "Upload Target":
    tgt_file = tgt_col2.file_uploader(
        "Upload target audio",
        type=["wav", "mp3", "m4a", "ogg"], 
        key="tgt_upload"
    )
    if tgt_file:
        tgt_bytes = tgt_file.read()
        tgt_col2.success(f"Target uploaded: {tgt_file.name}")
        
elif tgt_choice == "Record Target" and HAS_RECORDER:
    st.write("Record target audio:")
    recorded = audio_recorder(
        pause_threshold=2.0,
        sample_rate=44100,
        text="Click to Record Target"
    )
    if recorded and len(recorded) > 0:
        tgt_bytes = recorded
        st.success("Target recording captured")
        st.audio(recorded, format="audio/wav")

# Enhanced comparison with cosine similarity
def compute_cosine_similarity(ref_bytes, tgt_bytes):
    """Compute cosine similarity between audio features"""
    try:
        import librosa
        from io import BytesIO
        
        # Load both audios
        y_ref, sr_ref = librosa.load(BytesIO(ref_bytes), sr=None)
        y_tgt, sr_tgt = librosa.load(BytesIO(tgt_bytes), sr=None)
        
        # Resample if needed to match sample rates
        if sr_ref != sr_tgt:
            y_tgt = librosa.resample(y_tgt, orig_sr=sr_tgt, target_sr=sr_ref)
            sr_tgt = sr_ref
        
        # Ensure both audios have the same length by padding/truncating
        max_length = max(len(y_ref), len(y_tgt))
        if len(y_ref) < max_length:
            y_ref = np.pad(y_ref, (0, max_length - len(y_ref)), mode='constant')
        if len(y_tgt) < max_length:
            y_tgt = np.pad(y_tgt, (0, max_length - len(y_tgt)), mode='constant')
        
        # Extract MFCC features with consistent parameters
        n_mfcc = 13
        mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=sr_ref, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
        mfcc_tgt = librosa.feature.mfcc(y=y_tgt, sr=sr_tgt, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
        
        # Flatten MFCC features
        mfcc_ref_flat = mfcc_ref.flatten()
        mfcc_tgt_flat = mfcc_tgt.flatten()
        
        # Ensure same length for cosine calculation
        min_len = min(len(mfcc_ref_flat), len(mfcc_tgt_flat))
        mfcc_ref_flat = mfcc_ref_flat[:min_len]
        mfcc_tgt_flat = mfcc_tgt_flat[:min_len]
        
        # Compute cosine similarity (1 - cosine distance)
        if min_len > 0:
            cosine_sim_mfcc = 1 - cosine(mfcc_ref_flat, mfcc_tgt_flat)
        else:
            cosine_sim_mfcc = 0.0
        
        # Extract spectral centroids with consistent parameters
        centroid_ref = librosa.feature.spectral_centroid(y=y_ref, sr=sr_ref, n_fft=2048, hop_length=512)
        centroid_tgt = librosa.feature.spectral_centroid(y=y_tgt, sr=sr_tgt, n_fft=2048, hop_length=512)
        
        # Flatten spectral centroids
        centroid_ref_flat = centroid_ref.flatten()
        centroid_tgt_flat = centroid_tgt.flatten()
        
        # Ensure same length
        min_len_centroid = min(len(centroid_ref_flat), len(centroid_tgt_flat))
        centroid_ref_flat = centroid_ref_flat[:min_len_centroid]
        centroid_tgt_flat = centroid_tgt_flat[:min_len_centroid]
        
        # Compute spectral similarity
        if min_len_centroid > 0:
            cosine_sim_spectral = 1 - cosine(centroid_ref_flat, centroid_tgt_flat)
        else:
            cosine_sim_spectral = 0.0
        
        # Calculate RMS energy similarity
        rms_ref = librosa.feature.rms(y=y_ref, frame_length=2048, hop_length=512)
        rms_tgt = librosa.feature.rms(y=y_tgt, frame_length=2048, hop_length=512)
        
        rms_ref_flat = rms_ref.flatten()
        rms_tgt_flat = rms_tgt.flatten()
        min_len_rms = min(len(rms_ref_flat), len(rms_tgt_flat))
        rms_ref_flat = rms_ref_flat[:min_len_rms]
        rms_tgt_flat = rms_tgt_flat[:min_len_rms]
        
        if min_len_rms > 0:
            rms_similarity = 1 - cosine(rms_ref_flat, rms_tgt_flat)
        else:
            rms_similarity = 0.0
        
        # Weighted average of all similarity measures
        similarity_score = (cosine_sim_mfcc * 0.5 + cosine_sim_spectral * 0.3 + rms_similarity * 0.2)
        
        return {
            "cosine_similarity_mfcc": max(0, min(1, cosine_sim_mfcc)),
            "cosine_similarity_spectral": max(0, min(1, cosine_sim_spectral)),
            "rms_similarity": max(0, min(1, rms_similarity)),
            "similarity_score": max(0, min(1, similarity_score))
        }
        
    except Exception as e:
        st.error(f"Cosine similarity computation failed: {e}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None

# Comparison type
st.header("Choose Analysis Type")
compare_type = st.radio(
    "Analysis depth",
    ["Quick Metrics", "Advanced Comparison", "Visual Comparison"],
    help="Quick: Basic metrics | Advanced: Cosine similarity | Visual: Full waveforms and spectrograms"
)

if st.button("Run Comparison", type="primary"):
    if not ref_bytes or not tgt_bytes:
        st.error("Please provide both reference and target audio")
    else:
        if compare_type == "Quick Metrics":
            with st.spinner("Computing similarity metrics..."):
                try:
                    result = compute_similarity(ref_bytes, tgt_bytes)
                    
                    st.subheader("Quick Comparison Metrics")
                    
                    # Duration comparison
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Reference Duration", f"{result['duration_ref']:.2f}s")
                    col2.metric("Target Duration", f"{result['duration_target']:.2f}s")
                    
                    duration_ratio = result.get('duration_ratio')
                    if duration_ratio:
                        if duration_ratio > 1.2:
                            status = "Longer"
                        elif duration_ratio < 0.8:
                            status = "Shorter" 
                        else:
                            status = "Similar"
                        col3.metric("Duration Ratio", f"{duration_ratio:.2f}", status)
                    
                    # Amplitude comparison
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Reference Peak", f"{result['peak_ref']:.3f}")
                    col2.metric("Target Peak", f"{result['peak_target']:.3f}")
                    
                    peak_ratio = result.get('peak_ratio')
                    if peak_ratio:
                        if peak_ratio > 1.2:
                            status = "Louder"
                        elif peak_ratio < 0.8:
                            status = "Quieter"
                        else:
                            status = "Similar" 
                        col3.metric("Loudness Ratio", f"{peak_ratio:.2f}", status)
                    
                    # Correlation
                    if result.get('corr_coef') is not None:
                        st.metric(
                            "Waveform Correlation", 
                            f"{result['corr_coef']:.3f}",
                            help="1.0 = identical, 0.0 = no correlation, -1.0 = inverted"
                        )
                        
                except Exception as e:
                    st.error(f"Quick comparison failed: {str(e)}")
                    
        elif compare_type == "Advanced Comparison":
            with st.spinner("Computing advanced audio features and cosine similarity..."):
                try:
                    # Get basic metrics
                    basic_result = compute_similarity(ref_bytes, tgt_bytes)
                    
                    # Compute cosine similarity
                    cosine_result = compute_cosine_similarity(ref_bytes, tgt_bytes)
                    
                    if cosine_result:
                        st.subheader("Advanced Audio Similarity Analysis")
                        
                        # Similarity scores
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric(
                            "MFCC Similarity", 
                            f"{cosine_result['cosine_similarity_mfcc']:.3f}",
                            help="Similarity based on Mel-frequency cepstral coefficients (0-1)"
                        )
                        col2.metric(
                            "Spectral Similarity", 
                            f"{cosine_result['cosine_similarity_spectral']:.3f}",
                            help="Similarity based on spectral centroids (0-1)"
                        )
                        col3.metric(
                            "Energy Similarity", 
                            f"{cosine_result['rms_similarity']:.3f}",
                            help="Similarity based on RMS energy (0-1)"
                        )
                        col4.metric(
                            "Overall Similarity", 
                            f"{cosine_result['similarity_score']:.3f}",
                            help="Weighted combination of all features (0-1)"
                        )
                        
                        # Similarity interpretation
                        overall_score = cosine_result['similarity_score']
                        if overall_score > 0.8:
                            similarity_text = "Very Similar"
                            color = "green"
                        elif overall_score > 0.6:
                            similarity_text = "Similar"
                            color = "lightgreen"
                        elif overall_score > 0.4:
                            similarity_text = "Moderately Similar"
                            color = "orange"
                        else:
                            similarity_text = "Not Similar"
                            color = "red"
                            
                        st.markdown(f"<h3 style='color: {color};'>Similarity Assessment: {similarity_text}</h3>", 
                                  unsafe_allow_html=True)
                        
                        # Feature importance explanation
                        with st.expander("Feature Weights"):
                            st.markdown("""
                            **Similarity Score Breakdown:**
                            - MFCC Similarity (50%): Spectral shape and timbre
                            - Spectral Similarity (30%): Sound brightness and frequency distribution  
                            - Energy Similarity (20%): Loudness and amplitude patterns
                            """)
                        
                        # Basic metrics for reference
                        st.subheader("Basic Audio Metrics")
                        col1, col2 = st.columns(2)
                        col1.metric("Reference Duration", f"{basic_result['duration_ref']:.2f}s")
                        col2.metric("Target Duration", f"{basic_result['duration_target']:.2f}s")
                        col1.metric("Waveform Correlation", f"{basic_result.get('corr_coef', 0):.3f}")
                        col2.metric("Duration Ratio", f"{basic_result.get('duration_ratio', 0):.2f}")
                        
                    else:
                        st.error("Failed to compute cosine similarity")
                        
                except Exception as e:
                    st.error(f"Advanced comparison failed: {str(e)}")
                    
        else:  # Visual Comparison
            with st.spinner("Generating visual comparisons..."):
                try:
                    result = compare_audio_bytes(ref_bytes, tgt_bytes)
                    
                    st.subheader("Comparison Summary")
                    summary_col1, summary_col2, summary_col3 = st.columns(3)
                    summary_col1.metric("Duration", f"{result['duration_target']:.2f}s", 
                                      f"{result['duration_flag'].capitalize()}")
                    summary_col2.metric("Loudness", f"{result['peak_target']:.3f}",
                                      f"{result['loudness_flag'].capitalize()}")
                    summary_col3.metric("Reference", f"{result['duration_ref']:.2f}s")
                    
                    # Visualizations
                    st.subheader("Waveform Comparison")
                    st.image(result['waveform_side_png'], width='stretch',
                           caption="Side-by-side waveform comparison")
                    
                    st.image(result['waveform_overlay_png'], width='stretch', 
                           caption="Overlay waveform comparison")
                    
                    st.subheader("Spectrogram Analysis")
                    st.image(result['spectrogram_png'], width='stretch',
                           caption="Reference (top) vs Target (bottom) spectrograms")
                    
                except Exception as e:
                    st.error(f"Visual comparison failed: {str(e)}")

with st.expander("About Cosine Similarity in Audio Analysis"):
    st.markdown("""
    **Cosine Similarity** measures how similar two audio files are based on their acoustic features:
    
    - **MFCC Cosine Similarity**: Compares Mel-frequency cepstral coefficients, which represent the spectral shape of audio
    - **Spectral Similarity**: Compares spectral centroids, which represent the brightness of the sound
    - **Overall Similarity Score**: Combined measure of audio similarity (0-1 scale)
    
    **Interpretation:**
    - 0.8 - 1.0: Very similar (same speaker, same content)
    - 0.6 - 0.8: Similar (same content, different speaker)
    - 0.4 - 0.6: Moderately similar (similar acoustic properties)
    - 0.0 - 0.4: Not similar (different audio characteristics)
    """)