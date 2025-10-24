from typing import Dict, Any, List, Tuple
import streamlit as st
import re

def analyze_sentiment_chunked(text: str, chunk_size: int = 3) -> Dict[str, Any]:
    """
    Analyze sentiment of text using VADER with chunk-based analysis.
    Breaks text into chunks of sentences/words for more granular analysis.
    
    Args:
        text: Text to analyze
        chunk_size: Number of sentences per chunk
        
    Returns:
        Dictionary with sentiment scores and classifications
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        st.error("VADER sentiment not installed. Run: pip install vaderSentiment")
        return {"error": "VADER sentiment not available"}
    
    if not text or not text.strip():
        return {"error": "No text provided for analysis"}
    
    # Initialize VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    
    # Clean and prepare text
    text = text.strip()
    
    # Method 1: Sentence-based chunking
    def split_into_sentences(text):
        # Simple sentence splitting using punctuation
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    # Method 2: Word-based chunking
    def split_into_word_chunks(text, words_per_chunk=10):
        words = text.split()
        chunks = []
        for i in range(0, len(words), words_per_chunk):
            chunk = ' '.join(words[i:i + words_per_chunk])
            chunks.append(chunk)
        return chunks
    
    # Try sentence-based chunking first
    sentences = split_into_sentences(text)
    
    if len(sentences) <= 1:
        # If few sentences, use word-based chunking
        chunks = split_into_word_chunks(text, words_per_chunk=8)
    else:
        # Group sentences into chunks
        chunks = []
        for i in range(0, len(sentences), chunk_size):
            chunk = ' '.join(sentences[i:i + chunk_size])
            chunks.append(chunk)
    
    # Analyze each chunk
    chunk_sentiments = []
    chunk_details = []
    
    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) < 3:  # Skip very short chunks
            continue
            
        scores = analyzer.polarity_scores(chunk)
        compound_score = scores['compound']
        
        # Determine sentiment classification
        if compound_score >= 0.05:
            sentiment = "Positive"
        elif compound_score <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        chunk_sentiments.append(compound_score)
        chunk_details.append({
            "chunk_number": i + 1,
            "text": chunk,
            "sentiment": sentiment,
            "compound_score": round(compound_score, 3),
            "positive": round(scores['pos'], 3),
            "neutral": round(scores['neu'], 3),
            "negative": round(scores['neg'], 3)
        })
    
    if not chunk_sentiments:
        return {"error": "No analyzable content found"}
    
    # Calculate overall metrics
    overall_compound = sum(chunk_sentiments) / len(chunk_sentiments)
    
    # Count sentiment distribution
    sentiment_counts = {
        "Positive": sum(1 for detail in chunk_details if detail["sentiment"] == "Positive"),
        "Neutral": sum(1 for detail in chunk_details if detail["sentiment"] == "Neutral"),
        "Negative": sum(1 for detail in chunk_details if detail["sentiment"] == "Negative")
    }
    
    total_chunks = len(chunk_details)
    sentiment_percentages = {
        "positive": sentiment_counts["Positive"] / total_chunks * 100,
        "neutral": sentiment_counts["Neutral"] / total_chunks * 100,
        "negative": sentiment_counts["Negative"] / total_chunks * 100
    }
    
    # Determine overall sentiment based on chunk majority
    if sentiment_counts["Positive"] > sentiment_counts["Negative"] and sentiment_counts["Positive"] > sentiment_counts["Neutral"]:
        overall_sentiment = "Positive"
    elif sentiment_counts["Negative"] > sentiment_counts["Positive"] and sentiment_counts["Negative"] > sentiment_counts["Neutral"]:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"
    
    # Calculate weighted scores
    weighted_positive = sum(detail["positive"] for detail in chunk_details) / total_chunks
    weighted_neutral = sum(detail["neutral"] for detail in chunk_details) / total_chunks
    weighted_negative = sum(detail["negative"] for detail in chunk_details) / total_chunks
    
    result = {
        "sentiment": overall_sentiment,
        "compound_score": round(overall_compound, 3),
        "positive": round(weighted_positive, 3),
        "neutral": round(weighted_neutral, 3),
        "negative": round(weighted_negative, 3),
        "chunk_analysis": chunk_details,
        "sentiment_distribution": sentiment_counts,
        "sentiment_percentages": sentiment_percentages,
        "total_chunks": total_chunks,
        "analysis_method": "chunk_based"
    }
    
    return result

def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Legacy function for backward compatibility.
    Uses chunk-based analysis by default.
    """
    return analyze_sentiment_chunked(text)

def get_sentiment_breakdown(chunk_details: List[Dict]) -> Dict[str, Any]:
    """
    Generate detailed breakdown of sentiment analysis.
    """
    positive_chunks = [chunk for chunk in chunk_details if chunk["sentiment"] == "Positive"]
    negative_chunks = [chunk for chunk in chunk_details if chunk["sentiment"] == "Negative"]
    neutral_chunks = [chunk for chunk in chunk_details if chunk["sentiment"] == "Neutral"]
    
    # Find strongest positive and negative chunks
    strongest_positive = max(positive_chunks, key=lambda x: x["compound_score"], default=None)
    strongest_negative = min(negative_chunks, key=lambda x: x["compound_score"], default=None)
    
    return {
        "positive_chunks": positive_chunks,
        "negative_chunks": negative_chunks,
        "neutral_chunks": neutral_chunks,
        "strongest_positive": strongest_positive,
        "strongest_negative": strongest_negative
    }

    