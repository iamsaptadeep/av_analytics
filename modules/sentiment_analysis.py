# modules/sentiment_analysis.py
from typing import Dict, Any, List
import re
import streamlit as st

@st.cache_resource
def _get_vader_analyzer():
    """Load and cache a VADER SentimentIntensityAnalyzer instance."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        raise RuntimeError(
            "VADER sentiment not installed. Run: pip install vaderSentiment"
        )
    return SentimentIntensityAnalyzer()


def _clean_text(text: str) -> str:
    """Basic cleanup for filler words and Whisper artefacts."""
    text = re.sub(r"\b(uh+|um+|ah+|hmm+|mm+|erm+)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def analyze_sentiment_chunked(text: str, chunk_size: int = 3) -> Dict[str, Any]:
    """
    Chunk-based sentiment analysis using VADER.
    Breaks text into smaller chunks (sentences or word groups) and scores each.
    """
    if not text or not text.strip():
        return {"error": "No text provided for analysis"}

    try:
        analyzer = _get_vader_analyzer()
    except Exception as e:
        st.error(str(e))
        return {"error": str(e)}

    text = _clean_text(text)

    # --- sentence / word chunking helpers ---
    def split_into_sentences(txt: str) -> List[str]:
        sentences = re.split(r"[.!?]+", txt)
        return [s.strip() for s in sentences if s.strip()]

    def split_into_word_chunks(txt: str, words_per_chunk=10) -> List[str]:
        words = txt.split()
        return [" ".join(words[i : i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)]

    # --- chunking logic ---
    sentences = split_into_sentences(text)
    if len(sentences) <= 1:
        chunks = split_into_word_chunks(text, words_per_chunk=8)
    else:
        chunks = [" ".join(sentences[i : i + chunk_size]) for i in range(0, len(sentences), chunk_size)]

    chunk_details = []
    compound_scores = []

    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) < 3:
            continue

        scores = analyzer.polarity_scores(chunk)
        compound = scores["compound"]

        if compound >= 0.05:
            sentiment = "Positive"
        elif compound <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        chunk_details.append(
            {
                "chunk_number": i + 1,
                "text": chunk,
                "sentiment": sentiment,
                "compound_score": round(compound, 3),
                "positive": round(scores["pos"], 3),
                "neutral": round(scores["neu"], 3),
                "negative": round(scores["neg"], 3),
            }
        )
        compound_scores.append(compound)

    if not chunk_details:
        return {"error": "No analyzable content found"}

    # --- aggregate metrics ---
    total = len(chunk_details)
    counts = {
        "Positive": sum(1 for d in chunk_details if d["sentiment"] == "Positive"),
        "Neutral": sum(1 for d in chunk_details if d["sentiment"] == "Neutral"),
        "Negative": sum(1 for d in chunk_details if d["sentiment"] == "Negative"),
    }

    percentages = {k.lower(): counts[k] / total * 100 for k in counts}

    if counts["Positive"] > counts["Negative"] and counts["Positive"] > counts["Neutral"]:
        overall = "Positive"
    elif counts["Negative"] > counts["Positive"] and counts["Negative"] > counts["Neutral"]:
        overall = "Negative"
    else:
        overall = "Neutral"

    weighted = {
        "positive": round(sum(d["positive"] for d in chunk_details) / total, 3),
        "neutral": round(sum(d["neutral"] for d in chunk_details) / total, 3),
        "negative": round(sum(d["negative"] for d in chunk_details) / total, 3),
    }

    result = {
        "sentiment": overall,
        "compound_score": round(sum(compound_scores) / len(compound_scores), 3),
        **weighted,
        "chunk_analysis": chunk_details,
        "sentiment_distribution": counts,
        "sentiment_percentages": percentages,
        "total_chunks": total,
        "analysis_method": "chunk_based",
    }
    return result


def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Legacy wrapper for backward compatibility."""
    return analyze_sentiment_chunked(text)


def get_sentiment_breakdown(chunk_details: List[Dict]) -> Dict[str, Any]:
    """Generate lists of chunks by sentiment and strongest examples."""
    positive = [c for c in chunk_details if c["sentiment"] == "Positive"]
    negative = [c for c in chunk_details if c["sentiment"] == "Negative"]
    neutral = [c for c in chunk_details if c["sentiment"] == "Neutral"]

    strongest_positive = max(positive, key=lambda x: x["compound_score"], default=None)
    strongest_negative = min(negative, key=lambda x: x["compound_score"], default=None)

    return {
        "positive_chunks": positive,
        "negative_chunks": negative,
        "neutral_chunks": neutral,
        "strongest_positive": strongest_positive,
        "strongest_negative": strongest_negative,
    }



