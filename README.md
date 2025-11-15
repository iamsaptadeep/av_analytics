
## 🎧 EchoMind Analytics App

**Multimodal Audio & Video Intelligence Platform**
Built with **Streamlit**, **Whisper**, **Transformers**, and **VADER**, this app performs:

* 🎙️ **Audio-to-Text Transcription** (OpenAI Whisper)
* 🔊 **Audio-to-Audio Comparison** (Waveform, MFCC, Spectral Similarity)
* 💬 **YouTube Comment Sentiment & Sarcasm Analysis**
* 🧠 **Transformer-based Sarcasm Detection**
* ☁️ **Visual Word Clouds & Sentiment Insights**

---

## 🧩 Project Overview

### Key Features

| Module                       | Functionality                                                                                |
| ---------------------------- | -------------------------------------------------------------------------------------------- |
| **Audio Analysis**           | Transcribe audio files, detect language & extract speech-to-text insights                    |
| **Audio Comparison**         | Compare two audio clips — waveform correlation, cosine similarity, spectrogram visualization |
| **Video (YouTube) Analysis** | Fetch up to 5000 comments (incl. replies), run sentiment + sarcasm detection                 |
| **Sentiment Engine**         | Supports both **VADER** and **Transformer (DistilBERT)** models                              |
| **Sarcasm Classifier**       | Optional Transformer sarcasm model (`cardiffnlp/twitter-roberta-base-irony`)                 |
| **Word Clouds & Charts**     | Positive/Negative comment visualization and labeled bar charts                               |

---

## 🏗️ Project Structure

```
av_analytics/
│
├── modules/
│   ├── audio_transcription.py     # Whisper transcription logic (with ffmpeg fix)
│   ├── audio_comparison.py        # Advanced waveform, MFCC, spectral comparison
│   ├── sentiment_analysis.py      # VADER + Transformer sentiment utilities
│   ├── youtube_comments.py        # YouTube API fetch + sentiment/sarcasm logic
│
├── pages/
│   ├── Audio_Analysis.py          # Audio upload/transcribe Streamlit page
│   ├── Audio_Comparison.py        # Compare 2 audio files (visual + metrics)
│   ├── Video_Analysis.py          # YouTube comment sentiment dashboard
│
├── assets/                        # Sample audio/media files
│
├── streamlit_app.py               # Main entry point for Streamlit multipage app
├── requirements.txt               # All dependencies
└── README.md                      # You are here
```

---

## 📊 Features Showcase

### 🎙️ Audio Analysis

* Upload or record audio
* Real-time Whisper transcription
* Chunked fallback & segment-level output

### 🔊 Audio Comparison

* Compare reference & target audio
* Compute:

  * Waveform correlation
  * MFCC cosine similarity
  * Spectral + RMS energy similarity
* Spectrogram visualizations and summary metrics

### 💬 YouTube Comment Analysis

* Fetch up to **5000 comments** (with replies)
* Sentiment classification (positive / neutral / negative)
* Sarcasm detection probability
* Word cloud visualization
* Bar chart distribution
* Compact top 10 positive/negative comment tables with user, likes, sentiment, sarcasm

---

---

## 📘 Future Enhancements

* 🎯 Real-time audio stream comparison
* 📈 Interactive timeline of transcript segments
* 🌐 Multi-language sentiment detection
* 🧾 CSV export of YouTube analysis
* 🧊 Power BI or Streamlit Analytics dashboard integration

---

## 👨‍💻 Author

**Saptadeep**
Business Analytics & Data Science Enthusiast
PGDM – Globsyn Business School, Kolkata

* 💼 Data Science, Analytics, and AI Solutions Developer
* 🌐 Full-stack Web + ML Integration
* 📫 Reach out for collaborations or project demos

---

## 🪪 License

This project is licensed under the **MIT License**.
You are free to modify, extend, and use for research or educational purposes.

---

