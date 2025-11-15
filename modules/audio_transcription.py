# modules/audio_transcription.py
"""
Improved audio transcription helper for Whisper (OpenAI/whisper).
Features:
- FFmpeg detection with configurable fallback path
- Cached model loading with automatic device selection (CUDA vs CPU)
- Transcribe from bytes or file path
- Return full text or segment-level results (timestamps + text + confidence)
- Optional Streamlit integration (set use_streamlit=False to disable st.* calls)
- Safer temp-file handling and cleanup
- Pydub fallback conversion for formats Whisper can't read directly
- Simple automatic chunking for long audio using Whisper segments (if available) or pydub silence split
"""
from __future__ import annotations

import io
import os
import shutil
import tempfile
import traceback
from typing import Dict, List, Optional, Tuple

# Optional Streamlit usage
try:
    import streamlit as st  # type: ignore
except Exception:
    st = None  # type: ignore

# Optional torch import for device detection
try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

# -------------------------------------------------------------------------
# Configuration: update this if you rely on local FFmpeg build
# -------------------------------------------------------------------------
_FFMPEG_PATH = r"C:\ffmpeg\ffmpeg-2025-10-21-git-535d4047d3-essentials_build\bin"

# -------------------------------------------------------------------------
# Streamlit-safe logging helpers
# -------------------------------------------------------------------------
def _st_info(msg: str, use_streamlit: bool):
    if use_streamlit and st:
        st.info(msg)
    else:
        print("[INFO]", msg)


def _st_warning(msg: str, use_streamlit: bool):
    if use_streamlit and st:
        st.warning(msg)
    else:
        print("[WARN]", msg)


def _st_error(msg: str, use_streamlit: bool):
    if use_streamlit and st:
        st.error(msg)
    else:
        print("[ERROR]", msg)


# -------------------------------------------------------------------------
# Ensure FFmpeg availability
# -------------------------------------------------------------------------
def ensure_ffmpeg_in_path(ffmpeg_path: Optional[str] = None, use_streamlit: bool = True) -> None:
    """
    Ensure ffmpeg executable is discoverable in PATH. If not found and ffmpeg_path
    is supplied (or _FFMPEG_PATH), it will be prepended to PATH.
    """
    fp = ffmpeg_path or _FFMPEG_PATH
    if shutil.which("ffmpeg") is None:
        if os.path.isdir(fp):
            os.environ["PATH"] = fp + os.pathsep + os.environ.get("PATH", "")
            _st_info(f"FFmpeg path added to environment: {fp}", use_streamlit)
        else:
            _st_warning(f"FFmpeg not found at {fp}. Audio conversion may fail.", use_streamlit)
    else:
        _st_info("FFmpeg found in PATH.", use_streamlit)


# Ensure FFmpeg available early (prevents pydub errors)
ensure_ffmpeg_in_path(use_streamlit=False)

# -------------------------------------------------------------------------
# Model caching & device detection
# -------------------------------------------------------------------------
_whisper_cache: Dict[str, Tuple[object, str, bool]] = {}  # model_name -> (model, device, fp16_ok)


def _detect_device() -> Tuple[str, bool]:
    if torch is not None and torch.cuda.is_available():
        return "cuda", True
    return "cpu", False


def load_whisper_model(model_name: str = "base", use_streamlit: bool = True):
    """Load Whisper model and cache it."""
    if model_name in _whisper_cache:
        return _whisper_cache[model_name]

    ensure_ffmpeg_in_path(use_streamlit=use_streamlit)
    try:
        import whisper  # type: ignore
    except Exception as e:
        _st_error("OpenAI Whisper not installed. Run: pip install openai-whisper", use_streamlit)
        raise RuntimeError("OpenAI Whisper not installed") from e

    device_str, fp16_ok = _detect_device()
    try:
        try:
            model = whisper.load_model(model_name, device=device_str)
        except TypeError:
            model = whisper.load_model(model_name)
            if device_str == "cuda":
                try:
                    model.to("cuda")
                except Exception:
                    fp16_ok = False

        _whisper_cache[model_name] = (model, device_str, fp16_ok)
        _st_info(f"Whisper model '{model_name}' loaded on {device_str}.", use_streamlit)
        return model, device_str, fp16_ok
    except Exception as e:
        _st_error(f"Failed to load Whisper model '{model_name}': {e}", use_streamlit)
        raise


# -------------------------------------------------------------------------
# Detect audio format
# -------------------------------------------------------------------------
def _detect_audio_format(audio_bytes: bytes) -> str:
    if len(audio_bytes) < 4:
        return ".wav"
    if audio_bytes[:4] == b"RIFF":
        return ".wav"
    if audio_bytes[:3] == b"ID3":
        return ".mp3"
    if audio_bytes[:4] == b"fLaC":
        return ".flac"
    if audio_bytes[:4] == b"OggS":
        return ".ogg"
    if len(audio_bytes) >= 12 and audio_bytes[4:8] == b"ftyp":
        return ".m4a"
    return ".wav"


# -------------------------------------------------------------------------
# Convert bytes -> WAV using pydub
# -------------------------------------------------------------------------
def _bytes_to_wav_with_pydub(audio_bytes: bytes, use_streamlit: bool = True) -> str:
    ensure_ffmpeg_in_path(use_streamlit=use_streamlit)
    try:
        from pydub import AudioSegment  # type: ignore
    except Exception as e:
        _st_error("pydub not installed. Run: pip install pydub", use_streamlit)
        raise RuntimeError("pydub not installed") from e

    wav_tmp = None
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wf:
            wav_tmp = wf.name
        audio.export(wav_tmp, format="wav")
        return wav_tmp
    except FileNotFoundError as e:
        _st_error("FFmpeg is not accessible to pydub. Ensure ffmpeg.exe is in PATH.", use_streamlit)
        raise RuntimeError("FFmpeg not accessible to pydub") from e
    except Exception as e:
        _st_error(f"pydub conversion failed: {e}", use_streamlit)
        raise


# -------------------------------------------------------------------------
# Core transcription functions
# -------------------------------------------------------------------------
def transcribe_from_bytes(
    audio_bytes: bytes,
    model_name: str = "base",
    return_segments: bool = False,
    use_streamlit: bool = True,
    max_chunk_duration_sec: Optional[int] = 30,
) -> str | List[Dict]:
    if not audio_bytes:
        _st_error("No audio data provided for transcription.", use_streamlit)
        return "" if not return_segments else []

    try:
        model, device_str, fp16_ok = load_whisper_model(model_name, use_streamlit=use_streamlit)
    except Exception:
        return "" if not return_segments else []

    tmp_path = None
    converted_wav = None
    try:
        ext = _detect_audio_format(audio_bytes)
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        _st_info(f"Starting transcription using model '{model_name}' on {device_str}.", use_streamlit)
        try:
            result = model.transcribe(tmp_path, fp16=fp16_ok)
        except Exception as primary_exc:
            _st_warning(f"Primary transcription failed: {primary_exc}. Trying pydub conversion.", use_streamlit)
            try:
                converted_wav = _bytes_to_wav_with_pydub(audio_bytes, use_streamlit=use_streamlit)
            except Exception:
                _st_error("Fallback conversion failed. See traceback.", use_streamlit)
                _st_error(traceback.format_exc(), use_streamlit)
                return "" if not return_segments else []
            try:
                result = model.transcribe(converted_wav, fp16=fp16_ok)
            except Exception as e2:
                _st_error(f"Transcription failed on converted WAV: {e2}", use_streamlit)
                _st_error(traceback.format_exc(), use_streamlit)
                return "" if not return_segments else []

        text = result.get("text", "").strip() if isinstance(result, dict) else ""
        segments = result.get("segments", None) if isinstance(result, dict) else None

        if return_segments:
            if segments:
                parsed = []
                for seg in segments:
                    parsed.append(
                        {
                            "start": float(seg.get("start", 0.0)),
                            "end": float(seg.get("end", 0.0)),
                            "text": seg.get("text", "").strip(),
                            "confidence": float(seg.get("avg_logprob", 0.0))
                            if seg.get("avg_logprob") is not None
                            else None,
                        }
                    )
                return parsed
            else:
                # fallback chunking
                if max_chunk_duration_sec and max_chunk_duration_sec > 0:
                    try:
                        from pydub import AudioSegment, silence  # type: ignore

                        audio_seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
                        total_seconds = len(audio_seg) / 1000.0
                        if total_seconds <= max_chunk_duration_sec:
                            return [{"start": 0.0, "end": total_seconds, "text": text, "confidence": None}]

                        chunks = silence.split_on_silence(
                            audio_seg, min_silence_len=500, silence_thresh=-40, keep_silence=250
                        )
                        parsed = []
                        cursor = 0.0
                        for chunk in chunks:
                            chunk_path = None
                            try:
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wf:
                                    chunk_path = wf.name
                                chunk.export(chunk_path, format="wav")
                                try:
                                    chunk_result = model.transcribe(chunk_path, fp16=fp16_ok)
                                    chunk_text = chunk_result.get("text", "").strip()
                                except Exception:
                                    chunk_text = ""
                                chunk_duration = len(chunk) / 1000.0
                                parsed.append(
                                    {"start": cursor, "end": cursor + chunk_duration, "text": chunk_text, "confidence": None}
                                )
                                cursor += chunk_duration
                            finally:
                                if chunk_path and os.path.exists(chunk_path):
                                    try:
                                        os.unlink(chunk_path)
                                    except Exception:
                                        pass
                        return parsed
                    except Exception as e:
                        _st_warning(f"Chunking attempt failed: {e}", use_streamlit)
                        return [{"start": 0.0, "end": None, "text": text, "confidence": None}]
                return [{"start": 0.0, "end": None, "text": text, "confidence": None}]
        else:
            return text
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        if converted_wav and os.path.exists(converted_wav):
            try:
                os.unlink(converted_wav)
            except Exception:
                pass


def transcribe_from_file(
    file_path: str,
    model_name: str = "base",
    return_segments: bool = False,
    use_streamlit: bool = True,
) -> str | List[Dict]:
    if not os.path.exists(file_path):
        _st_error(f"File not found: {file_path}", use_streamlit)
        return "" if not return_segments else []

    try:
        model, device_str, fp16_ok = load_whisper_model(model_name, use_streamlit=use_streamlit)
    except Exception:
        return "" if not return_segments else []

    try:
        result = model.transcribe(file_path, fp16=fp16_ok)
        text = result.get("text", "").strip() if isinstance(result, dict) else ""
        segments = result.get("segments", None) if isinstance(result, dict) else None
        if return_segments:
            if segments:
                parsed = []
                for seg in segments:
                    parsed.append(
                        {
                            "start": float(seg.get("start", 0.0)),
                            "end": float(seg.get("end", 0.0)),
                            "text": seg.get("text", "").strip(),
                            "confidence": float(seg.get("avg_logprob", 0.0))
                            if seg.get("avg_logprob") is not None
                            else None,
                        }
                    )
                return parsed
            else:
                return [{"start": 0.0, "end": None, "text": text, "confidence": None}]
        else:
            return text
    except Exception as e:
        _st_error(f"Transcription failed: {e}", use_streamlit)
        _st_error(traceback.format_exc(), use_streamlit)
        return "" if not return_segments else []


def transcribe_export(
    audio_bytes: Optional[bytes] = None,
    file_path: Optional[str] = None,
    model_name: str = "base",
    use_streamlit: bool = True,
) -> Dict:
    if audio_bytes is None and file_path is None:
        _st_error("Either audio_bytes or file_path must be provided.", use_streamlit)
        return {}

    if audio_bytes is not None:
        segments = transcribe_from_bytes(audio_bytes, model_name=model_name, return_segments=True, use_streamlit=use_streamlit)
    else:
        segments = transcribe_from_file(file_path, model_name=model_name, return_segments=True, use_streamlit=use_streamlit)

    full_text = " ".join(seg.get("text", "") for seg in segments) if isinstance(segments, list) else (segments or "")

    device_str = "unknown"
    try:
        _, device_str, _ = load_whisper_model(model_name, use_streamlit=use_streamlit)
    except Exception:
        pass

    return {"model": model_name, "device": device_str, "full_text": full_text, "segments": segments}


if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Transcribe audio file for quick debugging.")
    parser.add_argument("file", help="Path to audio file")
    parser.add_argument("--model", default="base", help="Whisper model name (tiny, base, small, medium, large)")
    parser.add_argument("--segments", action="store_true", help="Return segments instead of full text")
    args = parser.parse_args()

    ensure_ffmpeg_in_path(use_streamlit=False)
    out = transcribe_from_file(args.file, model_name=args.model, return_segments=args.segments, use_streamlit=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))

