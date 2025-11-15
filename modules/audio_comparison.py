# modules/audio_comparison.py
"""
Advanced Audio Comparison Module
--------------------------------
Performs comprehensive comparison between two audio signals (reference and target)
using multi-level analysis:

1. Temporal metrics  - duration, amplitude, correlation
2. Spectral metrics  - spectral centroid, RMS, MFCCs
3. Cosine similarity across features (MFCCs, spectral, energy)
4. Visual analysis   - waveform overlay, side-by-side spectrograms
5. Weighted similarity score and interpretation
6. Optional Whisper semantic (text) similarity

Dependencies:
    pip install librosa numpy scipy matplotlib pydub openai-whisper
"""

from __future__ import annotations
import io
import os
import tempfile
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from typing import Dict, Optional

try:
    import streamlit as st
except Exception:
    st = None

# Optional semantic analysis via Whisper
try:
    import whisper
except ImportError:
    whisper = None

# ---------------------------------------------------------------------------
# Utility: safe logging for Streamlit or CLI
# ---------------------------------------------------------------------------
def _log_info(msg: str):
    if st:
        st.info(msg)
    else:
        print("[INFO]", msg)

def _log_warn(msg: str):
    if st:
        st.warning(msg)
    else:
        print("[WARN]", msg)

def _log_error(msg: str):
    if st:
        st.error(msg)
    else:
        print("[ERROR]", msg)


# ---------------------------------------------------------------------------
# Core Comparison Logic
# ---------------------------------------------------------------------------
def _load_audio_from_bytes(audio_bytes: bytes, sr: Optional[int] = None):
    """Load audio from bytes using librosa."""
    return librosa.load(io.BytesIO(audio_bytes), sr=sr, mono=True)


def compute_similarity(ref_bytes: bytes, tgt_bytes: bytes) -> Dict:
    """
    Compute multi-level similarity between reference and target audio.
    Returns numeric and perceptual metrics without visualization.
    """
    y_ref, sr_ref = _load_audio_from_bytes(ref_bytes)
    y_tgt, sr_tgt = _load_audio_from_bytes(tgt_bytes)

    # Resample target if needed
    if sr_ref != sr_tgt:
        y_tgt = librosa.resample(y_tgt, orig_sr=sr_tgt, target_sr=sr_ref)
        sr_tgt = sr_ref

    # Pad or trim to same length
    min_len = min(len(y_ref), len(y_tgt))
    y_ref = y_ref[:min_len]
    y_tgt = y_tgt[:min_len]

    # Basic waveform metrics
    duration_ref = len(y_ref) / sr_ref
    duration_target = len(y_tgt) / sr_tgt
    duration_ratio = duration_target / duration_ref if duration_ref else 0

    peak_ref = np.max(np.abs(y_ref))
    peak_tgt = np.max(np.abs(y_tgt))
    peak_ratio = peak_tgt / peak_ref if peak_ref else 0

    try:
        corr_coef = pearsonr(y_ref, y_tgt)[0]
    except Exception:
        corr_coef = 0.0

    # Spectral / MFCC-based similarity
    n_mfcc = 13
    mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=sr_ref, n_mfcc=n_mfcc)
    mfcc_tgt = librosa.feature.mfcc(y=y_tgt, sr=sr_tgt, n_mfcc=n_mfcc)
    mfcc_ref_flat = mfcc_ref.flatten()
    mfcc_tgt_flat = mfcc_tgt.flatten()
    min_len_mfcc = min(len(mfcc_ref_flat), len(mfcc_tgt_flat))
    mfcc_sim = 1 - cosine(mfcc_ref_flat[:min_len_mfcc], mfcc_tgt_flat[:min_len_mfcc]) if min_len_mfcc > 0 else 0.0

    centroid_ref = librosa.feature.spectral_centroid(y=y_ref, sr=sr_ref)
    centroid_tgt = librosa.feature.spectral_centroid(y=y_tgt, sr=sr_tgt)
    min_len_cent = min(centroid_ref.shape[1], centroid_tgt.shape[1])
    spec_sim = 1 - cosine(centroid_ref[0, :min_len_cent], centroid_tgt[0, :min_len_cent]) if min_len_cent > 0 else 0.0

    rms_ref = librosa.feature.rms(y=y_ref)
    rms_tgt = librosa.feature.rms(y=y_tgt)
    min_len_rms = min(rms_ref.shape[1], rms_tgt.shape[1])
    rms_sim = 1 - cosine(rms_ref[0, :min_len_rms], rms_tgt[0, :min_len_rms]) if min_len_rms > 0 else 0.0

    # Weighted final similarity
    similarity_score = (mfcc_sim * 0.5) + (spec_sim * 0.3) + (rms_sim * 0.2)

    interpretation = (
        "Very Similar" if similarity_score > 0.8
        else "Similar" if similarity_score > 0.6
        else "Moderately Similar" if similarity_score > 0.4
        else "Not Similar"
    )

    return {
        "duration_ref": duration_ref,
        "duration_target": duration_target,
        "duration_ratio": duration_ratio,
        "peak_ref": peak_ref,
        "peak_target": peak_tgt,
        "peak_ratio": peak_ratio,
        "corr_coef": corr_coef,
        "mfcc_similarity": mfcc_sim,
        "spectral_similarity": spec_sim,
        "rms_similarity": rms_sim,
        "similarity_score": similarity_score,
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# Visualization (Waveform, Spectrogram)
# ---------------------------------------------------------------------------
def _generate_waveform_images(y_ref, y_tgt, sr, tmpdir):
    """Generate waveform comparison plots."""
    overlay_path = os.path.join(tmpdir, "waveform_overlay.png")
    side_path = os.path.join(tmpdir, "waveform_side.png")

    # Overlay
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y_ref, sr=sr, alpha=0.6, label="Reference")
    librosa.display.waveshow(y_tgt, sr=sr, alpha=0.6, color='r', label="Target")
    plt.legend()
    plt.title("Overlay Waveform Comparison")
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=200)
    plt.close()

    # Side by side
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    librosa.display.waveshow(y_ref, sr=sr, ax=axs[0], color='b')
    axs[0].set_title("Reference Waveform")
    librosa.display.waveshow(y_tgt, sr=sr, ax=axs[1], color='r')
    axs[1].set_title("Target Waveform")
    plt.tight_layout()
    fig.savefig(side_path, dpi=200)
    plt.close(fig)

    return overlay_path, side_path


def _generate_spectrogram_images(y_ref, y_tgt, sr, tmpdir):
    """Generate spectrogram comparison image."""
    spec_path = os.path.join(tmpdir, "spectrograms.png")
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)
    S_ref = librosa.amplitude_to_db(np.abs(librosa.stft(y_ref)), ref=np.max)
    S_tgt = librosa.amplitude_to_db(np.abs(librosa.stft(y_tgt)), ref=np.max)
    img1 = librosa.display.specshow(S_ref, sr=sr, x_axis='time', y_axis='log', ax=axs[0])
    axs[0].set_title("Reference Spectrogram")
    img2 = librosa.display.specshow(S_tgt, sr=sr, x_axis='time', y_axis='log', ax=axs[1])
    axs[1].set_title("Target Spectrogram")
    plt.tight_layout()
    fig.savefig(spec_path, dpi=200)
    plt.close(fig)
    return spec_path


# ---------------------------------------------------------------------------
# High-level comparison API
# ---------------------------------------------------------------------------
def compare_audio_bytes(ref_bytes: bytes, tgt_bytes: bytes) -> Dict:
    """
    Full comparison with metrics + visualizations.
    Returns dict with file paths to generated images.
    """
    tmpdir = tempfile.mkdtemp(prefix="audio_compare_")

    try:
        y_ref, sr_ref = _load_audio_from_bytes(ref_bytes)
        y_tgt, sr_tgt = _load_audio_from_bytes(tgt_bytes)

        if sr_ref != sr_tgt:
            y_tgt = librosa.resample(y_tgt, orig_sr=sr_tgt, target_sr=sr_ref)
            sr_tgt = sr_ref

        min_len = min(len(y_ref), len(y_tgt))
        y_ref = y_ref[:min_len]
        y_tgt = y_tgt[:min_len]

        metrics = compute_similarity(ref_bytes, tgt_bytes)
        overlay_path, side_path = _generate_waveform_images(y_ref, y_tgt, sr_ref, tmpdir)
        spec_path = _generate_spectrogram_images(y_ref, y_tgt, sr_ref, tmpdir)

        metrics.update({
            "waveform_overlay_png": overlay_path,
            "waveform_side_png": side_path,
            "spectrogram_png": spec_path,
        })
        return metrics
    except Exception as e:
        _log_error(f"Audio comparison failed: {e}")
        return {"error": str(e)}
    finally:
        # Note: Do NOT delete tmpdir immediately — Streamlit needs images to exist.
        pass


# ---------------------------------------------------------------------------
# Optional Semantic (Whisper-based) Comparison
# ---------------------------------------------------------------------------
def compare_audio_files(ref_bytes: bytes, tgt_bytes: bytes, semantic: bool = False) -> Dict:
    """
    Combined acoustic + optional semantic (Whisper) comparison.
    """
    results = compare_audio_bytes(ref_bytes, tgt_bytes)

    if semantic and whisper is not None:
        try:
            _log_info("Running Whisper semantic comparison...")
            tmp1 = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp1.write(ref_bytes)
            tmp2.write(tgt_bytes)
            tmp1.close()
            tmp2.close()

            model = whisper.load_model("base")
            result1 = model.transcribe(tmp1.name, fp16=False)
            result2 = model.transcribe(tmp2.name, fp16=False)
            text1 = result1.get("text", "").strip()
            text2 = result2.get("text", "").strip()

            # Basic semantic similarity via character cosine
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            tfidf = TfidfVectorizer().fit_transform([text1, text2])
            sem_sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

            results["semantic_similarity"] = sem_sim
            results["transcript_1"] = text1
            results["transcript_2"] = text2
        except Exception as e:
            results["semantic_similarity"] = None
            results["note"] = f"Semantic comparison failed: {e}"
    elif semantic and whisper is None:
        results["note"] = "Whisper not installed; semantic comparison unavailable."

    return results


# ---------------------------------------------------------------------------
# CLI test usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="Advanced Audio Comparison CLI")
    parser.add_argument("ref", help="Reference audio path")
    parser.add_argument("target", help="Target audio path")
    parser.add_argument("--semantic", action="store_true", help="Enable Whisper semantic comparison")
    args = parser.parse_args()

    with open(args.ref, "rb") as f1, open(args.target, "rb") as f2:
        ref_b = f1.read()
        tgt_b = f2.read()

    result = compare_audio_files(ref_b, tgt_b, semantic=args.semantic)
    print(json.dumps(result, indent=2))



