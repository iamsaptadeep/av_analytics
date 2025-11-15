# pages/Audio_Comparison.py
import streamlit as st
from modules.audio_comparison import compare_audio_files

st.set_page_config(page_title="Audio Comparison", layout="wide")
st.title("Advanced Audio Comparison Analysis")

# -------------------------------------------------------------------------
# Sidebar options
# -------------------------------------------------------------------------
st.sidebar.header("Comparison Settings")
semantic = st.sidebar.checkbox("Enable Semantic (Whisper) Comparison", value=False)

# -------------------------------------------------------------------------
# Audio Inputs
# -------------------------------------------------------------------------
st.header("Provide Audio Inputs")

col1, col2 = st.columns(2)
ref_bytes, tgt_bytes = None, None

# Reference audio
ref_file = col1.file_uploader("Upload Reference Audio", type=["wav", "mp3", "m4a", "ogg", "flac"])
if ref_file:
    ref_bytes = ref_file.read()
    col1.audio(ref_bytes, format="audio/wav")
    col1.success(f"Loaded reference: {ref_file.name}")

# Target audio
tgt_file = col2.file_uploader("Upload Target Audio", type=["wav", "mp3", "m4a", "ogg", "flac"])
if tgt_file:
    tgt_bytes = tgt_file.read()
    col2.audio(tgt_bytes, format="audio/wav")
    col2.success(f"Loaded target: {tgt_file.name}")

# -------------------------------------------------------------------------
# Run Comparison
# -------------------------------------------------------------------------
if st.button("Run Advanced Comparison", type="primary"):
    if not ref_bytes or not tgt_bytes:
        st.error("Please upload both reference and target audio files.")
    else:
        with st.spinner("Analyzing audio similarity..."):
            result = compare_audio_files(ref_bytes, tgt_bytes, semantic=semantic)

        if "error" in result:
            st.error(result["error"])
        else:
            st.success("✅ Comparison completed successfully.")

            # -------------------------------------------------------------
            # Show Metrics
            # -------------------------------------------------------------
            st.subheader("🔢 Quantitative Metrics")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MFCC Similarity", f"{result['mfcc_similarity']*100:.1f}%")
            col2.metric("Spectral Similarity", f"{result['spectral_similarity']*100:.1f}%")
            col3.metric("Energy (RMS) Similarity", f"{result['rms_similarity']*100:.1f}%")
            col4.metric("Overall Score", f"{result['similarity_score']*100:.1f}%")

            # Interpretation
            score = result["similarity_score"]
            if score > 0.8:
                color = "green"; label = "Very Similar"
            elif score > 0.6:
                color = "lightgreen"; label = "Similar"
            elif score > 0.4:
                color = "orange"; label = "Moderately Similar"
            else:
                color = "red"; label = "Not Similar"

            st.markdown(f"### **Interpretation:** <span style='color:{color};'>{label}</span>", unsafe_allow_html=True)

            # -------------------------------------------------------------
            # Visualizations
            # -------------------------------------------------------------
            st.subheader("Visual Analysis")

            st.image(result["waveform_overlay_png"], caption="Overlay Waveform Comparison", use_container_width=True)
            st.image(result["waveform_side_png"], caption="Side-by-Side Waveform Comparison", use_container_width=True)
            st.image(result["spectrogram_png"], caption="Reference vs Target Spectrograms", use_container_width=True)

            # -------------------------------------------------------------
            # Additional Metrics
            # -------------------------------------------------------------
            with st.expander("Detailed Acoustic Metrics"):
                st.write({
                    "Duration (Reference)": f"{result['duration_ref']:.2f}s",
                    "Duration (Target)": f"{result['duration_target']:.2f}s",
                    "Duration Ratio": f"{result['duration_ratio']:.2f}",
                    "Correlation Coefficient": f"{result['corr_coef']:.3f}",
                    "Peak Ratio": f"{result['peak_ratio']:.2f}"
                })

            # -------------------------------------------------------------
            # Semantic Comparison (Whisper)
            # -------------------------------------------------------------
            if semantic:
                st.subheader("🗣️ Semantic (Whisper) Comparison")
                if result.get("semantic_similarity") is not None:
                    st.metric("Semantic Text Similarity", f"{result['semantic_similarity']*100:.1f}%")
                    with st.expander("View Transcripts"):
                        col1, col2 = st.columns(2)
                        col1.text_area("Reference Transcript", result.get("transcript_1", ""), height=200)
                        col2.text_area("Target Transcript", result.get("transcript_2", ""), height=200)
                else:
                    st.warning(result.get("note", "Semantic comparison unavailable."))

# -------------------------------------------------------------------------
# Explanation
# -------------------------------------------------------------------------
with st.expander("ℹ️ About This Analysis"):
    st.markdown("""
    **This comparison performs multi-level audio similarity analysis:**

    1. **MFCC Similarity (50%)** — Measures timbre and tonal shape.
    2. **Spectral Similarity (30%)** — Compares brightness and frequency energy distribution.
    3. **RMS Energy Similarity (20%)** — Evaluates loudness and amplitude pattern alignment.
    4. **Overall Score** — Weighted combination of all above (0–1 scale).

    **Interpretation:**
    - 0.80–1.00 → Very Similar (same voice/content)
    - 0.60–0.79 → Similar (similar timbre/content)
    - 0.40–0.59 → Moderately Similar
    - <0.40 → Not Similar

    When **Semantic Comparison** is enabled, the system transcribes both audio clips
    using Whisper and compares the transcribed text for meaning similarity.
    """)


