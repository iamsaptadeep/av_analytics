import tempfile
import os
from typing import Union
import streamlit as st
import io

def transcribe_from_bytes(audio_bytes: bytes) -> str:
    """
    Transcribe audio from bytes using OpenAI Whisper.
    
    Args:
        audio_bytes: Audio data as bytes
        
    Returns:
        Transcribed text as string
    """
    try:
        import whisper
    except ImportError:
        st.error("OpenAI Whisper not installed. Run: pip install openai-whisper")
        return ""
    
    if not audio_bytes or len(audio_bytes) == 0:
        st.error("No audio data provided")
        return ""
    
    # Save bytes to temporary file with proper extension detection
    file_extension = _detect_audio_format(audio_bytes)
    
    primary_method_failed = False
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        # Load whisper model (base model for speed)
        st.info("Loading Whisper model... (this may take a moment)")
        model = whisper.load_model("base")
        
        # Transcribe with progress indication
        with st.spinner("Transcribing audio..."):
            result = model.transcribe(tmp_path, fp16=False)  # Force FP32 to avoid warning
            
        return result["text"].strip()
    
    except Exception as e:
        primary_method_failed = True
        st.warning(f"Primary transcription method failed: {e}")
        # Try alternative method for problematic files
        return _transcribe_alternative(audio_bytes, file_extension)
    
    finally:
        # Clean up temporary file
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception as e:
            if not primary_method_failed:  # Only warn if primary method didn't already fail
                st.warning(f"Could not clean up temporary file: {e}")



def _detect_audio_format(audio_bytes: bytes) -> str:
    """Detect audio format from bytes"""
    if len(audio_bytes) < 4:
        return ".wav"
    
    # Check for common file signatures
    if audio_bytes[:4] == b'RIFF':
        return ".wav"
    elif audio_bytes[:3] == b'ID3':
        return ".mp3"
    elif audio_bytes[:4] == b'fLaC':
        return ".flac"
    elif audio_bytes[:4] == b'OggS':
        return ".ogg"
    elif audio_bytes[4:8] == b'ftyp':
        return ".m4a"
    else:
        # Try to determine by content inspection
        return ".wav"

def _transcribe_alternative(audio_bytes: bytes, file_extension: str) -> str:
    """Alternative transcription method using different approach"""
    try:
        import whisper
        import librosa
        import numpy as np
        from io import BytesIO
        
        st.info("Trying alternative transcription method...")
        
        # Convert bytes to numpy array using librosa
        with st.spinner("Loading audio with librosa..."):
            audio_data, sample_rate = librosa.load(BytesIO(audio_bytes), sr=16000)
        
        # Load model
        model = whisper.load_model("base")
        
        # Transcribe from numpy array
        with st.spinner("Transcribing with alternative method..."):
            result = model.transcribe(audio_data, fp16=False)
        
        st.success("✅ Transcription successful with alternative method!")
        return result["text"].strip()
        
    except Exception as e:
        st.error(f"❌ All transcription methods failed: {e}")
        return ""


        
def transcribe_from_wavfile(file_path: str) -> str:
    """
    Transcribe audio from wav file path using OpenAI Whisper.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Transcribed text as string
    """
    try:
        import whisper
    except ImportError:
        st.error("OpenAI Whisper not installed. Run: pip install openai-whisper")
        return ""
    
    try:
        if not os.path.exists(file_path):
            st.error(f"Audio file not found: {file_path}")
            return ""
        
        # Load whisper model (base model for speed)
        model = whisper.load_model("base")
        
        # Transcribe
        result = model.transcribe(file_path, fp16=False)
        return result["text"].strip()
    
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        return ""



