streamlit_app.py                 # Main app with navigation
pages/
  1_Audio_Analysis.py           # Audio transcription & sentiment
  2_Audio_Comparison.py         # Audio comparison features  
  3_Video_Analysis.py           # YouTube comment analysis
modules/
  __init__.py
  audio_transcription.py        # Whisper transcription
  audio_comparison.py           # Audio comparison & visualization
  sentiment_analysis.py         # VADER sentiment
  youtube_advanced.py           # YouTube API integration
  utils.py                      # File utilities
.streamlit/
  secrets.toml                  # Local secrets (gitignored)
requirements.txt               # Python dependencies
assets/
  sample_media/
    hello.wav                  # Sample audio for comparison




    