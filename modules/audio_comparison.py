# modules/audio_comparison.py
"""
Audio comparison helpers adapted from your script.
Functions:
 - compare_audio_bytes(ref_bytes, target_bytes)
   -> returns dict with metrics and PNG images as bytes: waveform_side, waveform_overlay, spectrogram
 - optional local_record() helper for local development (uses sounddevice)
Note: librosa/soundfile required for best results. If librosa isn't available,
fallbacks to scipy wav reader for basic metrics.
"""
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt

# Attempt to import librosa for best results
try:
    import librosa
    import librosa.display
    HAS_LIBROSA = True
except Exception:
    HAS_LIBROSA = False

# For fallback wav reading
from scipy.io import wavfile
from tempfile import mkstemp
import os

def _save_fig_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()

def _load_audio_from_bytes(bytes_obj):
    """
    Return (y, sr). Prefer librosa. If librosa not present, write temp wav and use scipy.
    """
    if HAS_LIBROSA:
        # librosa can read many formats from a bytes buffer via soundfile
        import soundfile as sf
        # SoundFile can read file-like objects but librosa.load expects a path or file, so use BytesIO with sf to read
        data, sr = sf.read(BytesIO(bytes_obj), always_2d=False)
        y = data
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        y = y.astype(np.float32)
        return y, sr
    else:
        # fallback: write to temp file and use scipy.wavfile (works only for WAV)
        fd, path = mkstemp(suffix=".wav")
        os.close(fd)
        with open(path, "wb") as f:
            f.write(bytes_obj)
        sr, arr = wavfile.read(path)
        try:
            os.remove(path)
        except Exception:
            pass
        if arr.ndim > 1:
            arr = np.mean(arr, axis=1)
        arr = arr.astype(np.float32)
        # Normalize if integer type
        if arr.dtype.kind == 'i':
            maxv = np.iinfo(arr.dtype).max
            arr = arr / maxv
        return arr, sr

def _spectrogram_db(y, sr):
    if HAS_LIBROSA:
        D = np.abs(librosa.stft(y))
        return librosa.amplitude_to_db(D, ref=np.max)
    else:
        # very rough fallback: use short-time Fourier transform via numpy (coarse)
        n_fft = 2048
        hop_length = n_fft // 4
        # pad
        if len(y) < n_fft:
            y = np.pad(y, (0, n_fft - len(y)), mode='constant')
        frames = np.lib.stride_tricks.sliding_window_view(y, n_fft)[::hop_length]
        S = np.abs(np.fft.rfft(frames, axis=1))
        S = S.T
        # convert to dB-like
        S_db = 20 * np.log10(np.maximum(S, 1e-10))
        return S_db

def compare_audio_bytes(ref_bytes: bytes, target_bytes: bytes):
    """
    Compare two audio byte objects.
    Returns:
      {
        'duration_ref': float,
        'duration_target': float,
        'peak_ref': float,
        'peak_target': float,
        'duration_flag': str,
        'loudness_flag': str,
        'waveform_side_png': bytes,
        'waveform_overlay_png': bytes,
        'spectrogram_png': bytes
      }
    """
    # load
    y_ref, sr_ref = _load_audio_from_bytes(ref_bytes)
    y_tgt, sr_tgt = _load_audio_from_bytes(target_bytes)

    # resample target if sr differs and librosa available
    if sr_ref != sr_tgt and HAS_LIBROSA:
        y_tgt = librosa.resample(y_tgt, orig_sr=sr_tgt, target_sr=sr_ref)
        sr_tgt = sr_ref

    # durations
    dur_ref = len(y_ref) / float(sr_ref)
    dur_tgt = len(y_tgt) / float(sr_tgt)

    # peak amplitudes
    peak_ref = float(np.max(np.abs(y_ref)))
    peak_tgt = float(np.max(np.abs(y_tgt)))

    # duration flag
    if dur_tgt > dur_ref * 1.2:
        duration_flag = "longer"
    elif dur_tgt < dur_ref * 0.8:
        duration_flag = "shorter"
    else:
        duration_flag = "similar"

    # loudness flag
    if peak_tgt > peak_ref * 1.2:
        loudness_flag = "louder"
    elif peak_tgt < peak_ref * 0.8:
        loudness_flag = "softer"
    else:
        loudness_flag = "similar"

    # --- Waveform side-by-side ---
    fig1 = plt.figure(figsize=(12, 3.5))
    ax1 = fig1.add_subplot(1, 2, 1)
    if HAS_LIBROSA:
        librosa.display.waveshow(y_ref, sr=sr_ref, ax=ax1)
    else:
        times = np.arange(len(y_ref)) / sr_ref
        ax1.plot(times, y_ref)
    ax1.set_title("Reference")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")

    ax2 = fig1.add_subplot(1, 2, 2)
    if HAS_LIBROSA:
        librosa.display.waveshow(y_tgt, sr=sr_tgt, ax=ax2)
    else:
        times2 = np.arange(len(y_tgt)) / sr_tgt
        ax2.plot(times2, y_tgt)
    ax2.set_title("Target")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    plt.tight_layout()
    waveform_side_png = _save_fig_to_bytes(fig1)

    # --- Waveform overlay ---
    fig2 = plt.figure(figsize=(10, 3.5))
    ax = fig2.add_subplot(1, 1, 1)
    if HAS_LIBROSA:
        librosa.display.waveshow(y_ref, sr=sr_ref, alpha=0.6, label="Reference", ax=ax)
        librosa.display.waveshow(y_tgt, sr=sr_tgt, alpha=0.6, label="Target", ax=ax)
    else:
        t_ref = np.arange(len(y_ref)) / sr_ref
        t_tgt = np.arange(len(y_tgt)) / sr_tgt
        ax.plot(t_ref, y_ref, alpha=0.6, label="Reference")
        ax.plot(t_tgt, y_tgt, alpha=0.6, label="Target")
    ax.set_title("Overlay Comparison")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc='upper right')
    plt.tight_layout()
    waveform_overlay_png = _save_fig_to_bytes(fig2)

    # --- Spectrogram side-by-side ---
    S_ref = _spectrogram_db(y_ref, sr_ref)
    S_tgt = _spectrogram_db(y_tgt, sr_tgt)
    # create time/freq axes if librosa present for nicer display
    fig3 = plt.figure(figsize=(10, 6))
    ax1 = fig3.add_subplot(2, 1, 1)
    if HAS_LIBROSA:
        librosa.display.specshow(S_ref, sr=sr_ref, x_axis='time', y_axis='hz', ax=ax1)
    else:
        ax1.imshow(S_ref, aspect='auto', origin='lower', interpolation='nearest')
    ax1.set_title("Reference Spectrogram")
    ax1.set_xlabel("")
    ax1.set_ylabel("Frequency")

    ax2 = fig3.add_subplot(2, 1, 2)
    if HAS_LIBROSA:
        librosa.display.specshow(S_tgt, sr=sr_tgt, x_axis='time', y_axis='hz', ax=ax2)
    else:
        ax2.imshow(S_tgt, aspect='auto', origin='lower', interpolation='nearest')
    ax2.set_title("Target Spectrogram")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency")
    plt.tight_layout()
    spectrogram_png = _save_fig_to_bytes(fig3)

    result = {
        "duration_ref": dur_ref,
        "duration_target": dur_tgt,
        "peak_ref": peak_ref,
        "peak_target": peak_tgt,
        "duration_flag": duration_flag,
        "loudness_flag": loudness_flag,
        "waveform_side_png": waveform_side_png,
        "waveform_overlay_png": waveform_overlay_png,
        "spectrogram_png": spectrogram_png
    }
    return result

# Lightweight numeric comparison for "Quick Compare" tab
def compute_similarity(ref_bytes: bytes, target_bytes: bytes):
    """
    Compute simple similarity metrics between two audio byte objects.
    Returns a JSON-serializable dict with:
      - duration_ref, duration_target
      - peak_ref, peak_target
      - duration_ratio (target/ref)
      - peak_ratio (target/ref)
      - corr_coef (Pearson correlation on aligned, resampled mono signals)
    """
    # load
    y_ref, sr_ref = _load_audio_from_bytes(ref_bytes)
    y_tgt, sr_tgt = _load_audio_from_bytes(target_bytes)

    # Resample target to reference rate if needed
    if sr_ref != sr_tgt:
        if HAS_LIBROSA:
            y_tgt = librosa.resample(y_tgt, orig_sr=sr_tgt, target_sr=sr_ref)
            sr_tgt = sr_ref
        else:
            # scipy-based resample fallback
            from scipy.signal import resample
            num_samples = int(len(y_tgt) * (sr_ref / float(sr_tgt)))
            if num_samples <= 0:
                num_samples = 1
            y_tgt = resample(y_tgt, num_samples)
            sr_tgt = sr_ref

    # Compute durations and peaks
    dur_ref = len(y_ref) / float(sr_ref)
    dur_tgt = len(y_tgt) / float(sr_tgt)
    peak_ref = float(np.max(np.abs(y_ref))) if len(y_ref) else 0.0
    peak_tgt = float(np.max(np.abs(y_tgt))) if len(y_tgt) else 0.0

    # Align to same length for correlation
    n = min(len(y_ref), len(y_tgt))
    corr_coef = None
    if n >= 2:
        a = y_ref[:n]
        b = y_tgt[:n]
        # Avoid NaNs
        if np.std(a) > 0 and np.std(b) > 0:
            corr_coef = float(np.corrcoef(a, b)[0, 1])
        else:
            corr_coef = 0.0
    else:
        corr_coef = 0.0

    # Ratios (guard divide-by-zero)
    duration_ratio = float(dur_tgt / dur_ref) if dur_ref > 0 else None
    peak_ratio = float(peak_tgt / peak_ref) if peak_ref > 0 else None

    return {
        "duration_ref": float(dur_ref),
        "duration_target": float(dur_tgt),
        "peak_ref": float(peak_ref),
        "peak_target": float(peak_tgt),
        "duration_ratio": duration_ratio,
        "peak_ratio": peak_ratio,
        "corr_coef": corr_coef,
    }

# Optional: helper for local recording (development only)
def local_record(duration=3, fs=44100, channels=1, out_path=None):
    """
    Local-only helper that uses sounddevice to record audio and returns bytes.
    Not usable on Streamlit Cloud.
    """
    try:
        import sounddevice as sd
        from scipy.io.wavfile import write
    except Exception as e:
        raise RuntimeError("sounddevice or scipy not available: " + str(e))
    rec = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype='float32')
    sd.wait()
    rec_norm = rec / np.max(np.abs(rec))
    rec_int16 = (rec_norm * 32767).astype('int16')
    if out_path:
        write(out_path, fs, rec_int16)
        with open(out_path, "rb") as f:
            return f.read()
    else:
        fd, tmp = mkstemp(suffix=".wav")
        os.close(fd)
        write(tmp, fs, rec_int16)
        with open(tmp, "rb") as f:
            data = f.read()
        try:
            os.remove(tmp)
        except Exception:
            pass
        return data
