# modules/youtube_comments.py
"""
Advanced YouTube comments fetch + sentiment + sarcasm utilities.

Features:
- fetch_youtube_comments(video_url, max_comments=5000, include_replies=True)
- analyze_comments_sentiment(..., method="vader"|"transformer",
                              use_sarcasm_model=True, sarcasm_model_name=...)
    Adds keys: _sentiment_label, _sentiment_score, _sarcasm (bool), _sarcasm_score (0-1)
- build_wordcloud(comments)
- top_n_comments(comments, n=20, sort_by="like_count")
"""
from __future__ import annotations

import os
import re
import tempfile
from typing import List, Dict, Any, Optional
from collections import Counter

# optional streamlit import (module works without it)
try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    st = None  # type: ignore

# Google API client for YouTube Data API
from googleapiclient.discovery import build

# VADER Sentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Optional HuggingFace transformers
HF_AVAILABLE = False
try:
    from transformers import pipeline as hf_pipeline  # type: ignore
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# Optional wordcloud
WORDCLOUD_AVAILABLE = False
try:
    from wordcloud import WordCloud, STOPWORDS  # type: ignore
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False

# helper logger that works with or without Streamlit
def _info(msg: str):
    if st:
        st.info(msg)
    else:
        print("[INFO]", msg)


def _warn(msg: str):
    if st:
        st.warning(msg)
    else:
        print("[WARN]", msg)


def _error(msg: str):
    if st:
        st.error(msg)
    else:
        print("[ERROR]", msg)


# -------------------------
# Helper: caching decorator compatible with/without Streamlit
# -------------------------
def _cache_resource_decorator():
    """
    Return a decorator to cache resources:
    - uses st.cache_resource if streamlit present
    - else uses a simple lru_cache-like wrapper
    """
    if st:
        return st.cache_resource  # type: ignore
    else:
        from functools import lru_cache

        def deco(func):
            return lru_cache(maxsize=1)(func)

        return deco


_cache_resource = _cache_resource_decorator()

# -------------------------
# Video ID extractor
# -------------------------
def _extract_video_id(url: str) -> str:
    """
    Extract a YouTube video id from a URL or return the input if it already looks like an id.
    Supports:
      - https://www.youtube.com/watch?v=VIDEOID
      - https://youtu.be/VIDEOID
      - https://www.youtube.com/embed/VIDEOID
      - raw video id (11 chars)
    """
    if not url:
        return ""
    u = url.strip()
    # If looks like an ID already (11 chars typical)
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", u):
        return u
    # common patterns: v=VIDEOID
    m = re.search(r"[?&]v=([0-9A-Za-z_-]{11})", u)
    if m:
        return m.group(1)
    # youtu.be short link or /embed/
    m2 = re.search(r"(?:youtu\.be/|/embed/)([0-9A-Za-z_-]{11})", u)
    if m2:
        return m2.group(1)
    # fallback: try to find any 11-char token
    m3 = re.search(r"([0-9A-Za-z_-]{11})", u)
    if m3:
        return m3.group(1)
    return ""


# -------------------------
# Fetch comments (top-level + replies)
# -------------------------
def fetch_youtube_comments(video_url: str, max_comments: int = 5000, include_replies: bool = True) -> List[Dict[str, Any]]:
    """
    Fetch up to max_comments comments for a YouTube video (including replies if requested).
    Requires a YOUTUBE_API_KEY in st.secrets or environment variable.

    Returns list of dicts with keys:
      id, parent_id, author, text, like_count, published_at, is_reply
    """
    # find API key
    api_key = None
    if st and hasattr(st, "secrets") and "YOUTUBE_API_KEY" in st.secrets:
        api_key = st.secrets["YOUTUBE_API_KEY"]
    else:
        api_key = os.environ.get("YOUTUBE_API_KEY")

    if not api_key:
        raise RuntimeError("Missing YouTube API key. Set st.secrets['YOUTUBE_API_KEY'] or YOUTUBE_API_KEY env var.")

    try:
        youtube = build("youtube", "v3", developerKey=api_key)
    except Exception as e:
        _error(f"Failed to initialize YouTube client: {e}")
        raise

    video_id = _extract_video_id(video_url)
    if not video_id:
        raise ValueError("Could not parse video id from URL.")

    max_comments = max(0, min(5000, int(max_comments)))
    comments: List[Dict[str, Any]] = []
    fetched = 0
    next_page_token = None

    # Fetch top-level comment threads
    while fetched < max_comments:
        try:
            req = youtube.commentThreads().list(
                part="snippet,replies",
                videoId=video_id,
                pageToken=next_page_token,
                maxResults=min(100, max_comments - fetched),
                textFormat="plainText",
            )
            resp = req.execute()
        except Exception as e:
            _warn(f"Error fetching comment threads: {e}")
            break

        items = resp.get("items", [])
        for item in items:
            try:
                top = item["snippet"]["topLevelComment"]["snippet"]
                comment_obj = {
                    "id": item["snippet"]["topLevelComment"]["id"],
                    "parent_id": None,
                    "author": top.get("authorDisplayName"),
                    "text": top.get("textDisplay") or "",
                    "like_count": int(top.get("likeCount", 0) or 0),
                    "published_at": top.get("publishedAt"),
                    "is_reply": False,
                }
                comments.append(comment_obj)
                fetched += 1
                if include_replies and item.get("replies"):
                    for r in item["replies"].get("comments", []):
                        rs = r.get("snippet", {})
                        reply_obj = {
                            "id": r.get("id"),
                            "parent_id": item["snippet"]["topLevelComment"]["id"],
                            "author": rs.get("authorDisplayName"),
                            "text": rs.get("textDisplay") or "",
                            "like_count": int(rs.get("likeCount", 0) or 0),
                            "published_at": rs.get("publishedAt"),
                            "is_reply": True,
                        }
                        comments.append(reply_obj)
                        fetched += 1
                        if fetched >= max_comments:
                            break
                if fetched >= max_comments:
                    break
            except Exception:
                # ignore per-item errors to be robust
                continue

        next_page_token = resp.get("nextPageToken")
        if not next_page_token or fetched >= max_comments:
            break

    # For threads with more replies than included, optionally fetch remaining replies
    if include_replies and fetched < max_comments:
        extra_needed = max_comments - fetched
        parent_ids = [c["id"] for c in comments if not c.get("is_reply")]
        for pid in parent_ids:
            if extra_needed <= 0:
                break
            ptoken = None
            while extra_needed > 0:
                try:
                    creq = youtube.comments().list(
                        part="snippet",
                        parentId=pid,
                        pageToken=ptoken,
                        maxResults=min(100, extra_needed),
                        textFormat="plainText",
                    )
                    cresp = creq.execute()
                except Exception:
                    break
                for r in cresp.get("items", []):
                    try:
                        rs = r.get("snippet", {})
                        reply_obj = {
                            "id": r.get("id"),
                            "parent_id": pid,
                            "author": rs.get("authorDisplayName"),
                            "text": rs.get("textDisplay") or "",
                            "like_count": int(rs.get("likeCount", 0) or 0),
                            "published_at": rs.get("publishedAt"),
                            "is_reply": True,
                        }
                        comments.append(reply_obj)
                        extra_needed -= 1
                        fetched += 1
                        if extra_needed <= 0:
                            break
                    except Exception:
                        continue
                ptoken = cresp.get("nextPageToken")
                if not ptoken:
                    break

    return comments


# -------------------------
# Sentiment & Sarcasm
# -------------------------
_vader = SentimentIntensityAnalyzer()


@_cache_resource
def _load_transformer_sentiment(model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
    """
    Return a transformers sentiment-analysis pipeline.
    We do not pass truncation at pipeline creation to avoid tokenizer-max-length warnings;
    truncation will be supplied on call.
    """
    if not HF_AVAILABLE:
        raise RuntimeError("transformers not installed. Install: pip install transformers")
    try:
        return hf_pipeline("sentiment-analysis", model=model_name)
    except Exception as e:
        # propagate with clear message
        raise RuntimeError(f"Failed to create sentiment pipeline for '{model_name}': {e}")


@_cache_resource
def _load_sarcasm_transformer(model_name: str = "mrm8488/bert-tiny-finetuned-sarcasm"):
    """
    Return a transformers text-classification pipeline for sarcasm/irony detection.
    """
    if not HF_AVAILABLE:
        raise RuntimeError("transformers not installed. Install: pip install transformers")
    try:
        return hf_pipeline("text-classification", model=model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to create sarcasm pipeline for '{model_name}': {e}")


def _heuristic_sarcasm_check(text: str) -> bool:
    """
    Heuristic checks for likely sarcasm markers. Fast but shallow.
    """
    if not text or len(text) < 3:
        return False
    t = text.strip()
    low = t.lower()
    # explicit markers
    if "/s" in low or "#sarcasm" in low:
        return True
    # punctuation patterns
    if re.search(r"[!]{3,}", t) or re.search(r"\?\?+", t):
        return True
    # quotes used ironically
    if re.search(r'"\w+"', t) or re.search(r"‘\w+’", t):
        return True
    # ALL CAPS words combined with other words
    caps = re.findall(r"\b[A-Z]{3,}\b", t)
    if caps and len(t.split()) > 2:
        return True
    # emoji-heavy short text
    emojis = re.findall(r"[\U00010000-\U0010ffff\U0001F300-\U0001FAFF\U00002600-\U000027BF]+", t)
    if len(emojis) >= 2 and len(t.split()) <= 6:
        return True
    return False


def analyze_comments_sentiment(
    comments: List[Dict[str, Any]],
    method: str = "vader",
    transformer_model: Optional[str] = None,
    use_sarcasm_model: bool = False,
    sarcasm_model_name: str = "mrm8488/bert-tiny-finetuned-sarcasm",
    sarcasm_confidence_threshold: float = 0.6,
    batch_size: int = 32,
    invert_on_sarcasm: bool = True,
) -> List[Dict[str, Any]]:
    """
    Analyze comments for sentiment and sarcasm.

    Augments each comment dict with:
      _sentiment_label: "Positive"/"Neutral"/"Negative"
      _sentiment_score: (VADER compound or transformer score)
      _sarcasm: bool
      _sarcasm_score: float 0..1 (0 if not computed)
    Params:
      method: "vader" or "transformer"
      use_sarcasm_model: whether to use HF sarcasm model in addition to heuristics
    """
    if method not in ("vader", "transformer"):
        raise ValueError("method must be 'vader' or 'transformer'")

    transformer_pipeline = None
    sarcasm_pipeline = None

    if method == "transformer":
        transformer_model = transformer_model or "distilbert-base-uncased-finetuned-sst-2-english"
        try:
            transformer_pipeline = _load_transformer_sentiment(transformer_model)
        except Exception as e:
            _warn(f"Failed to load transformer sentiment model '{transformer_model}': {e}")
            transformer_pipeline = None
            # fallback to vader
            method = "vader"

    if use_sarcasm_model:
        try:
            sarcasm_pipeline = _load_sarcasm_transformer(sarcasm_model_name)
        except Exception as e:
            _warn(f"Failed to load sarcasm model '{sarcasm_model_name}': {e}")
            sarcasm_pipeline = None

    texts = [c.get("text", "") or "" for c in comments]
    total = len(comments)
    # process in batches
    for idx in range(0, total, batch_size):
        batch_texts = texts[idx: idx + batch_size]
        batch_comments = comments[idx: idx + batch_size]

        # Sentiment analysis
        if method == "vader" or transformer_pipeline is None:
            for t, c in zip(batch_texts, batch_comments):
                vs = _vader.polarity_scores(t or "")
                comp = vs["compound"]
                if comp >= 0.05:
                    lab = "Positive"
                elif comp <= -0.05:
                    lab = "Negative"
                else:
                    lab = "Neutral"
                c["_sentiment_label"] = lab
                c["_sentiment_score"] = round(float(comp), 4)
        else:
            # try batch call with truncation / max_length to avoid tokenizer warning
            try:
                preds = transformer_pipeline(batch_texts, truncation=True, max_length=512)
            except Exception:
                # fallback to per-item calls to be robust
                preds = []
                for t in batch_texts:
                    try:
                        p = transformer_pipeline(t, truncation=True, max_length=512)
                        # pipeline returns list for single string; normalize
                        if isinstance(p, list) and len(p) > 0:
                            preds.append(p[0])
                        else:
                            preds.append(p)
                    except Exception:
                        preds.append({"label": "NEUTRAL", "score": 0.0})

            # normalize results
            if not isinstance(preds, list):
                preds = list(preds)

            for p, c in zip(preds, batch_comments):
                label_raw = p.get("label", "")
                score = float(p.get("score", 0.0))
                if label_raw and isinstance(label_raw, str):
                    lr = label_raw.lower()
                    if lr.startswith("pos"):
                        lab = "Positive"
                    elif lr.startswith("neg"):
                        lab = "Negative"
                    else:
                        lab = "Neutral"
                else:
                    # fallback heuristic
                    lab = "Positive" if score >= 0.5 else "Negative"
                c["_sentiment_label"] = lab
                c["_sentiment_score"] = round(score, 4)

        # Sarcasm detection: heuristic + optional model
        if sarcasm_pipeline is not None:
            try:
                sar_preds = sarcasm_pipeline(batch_texts, truncation=True, max_length=512)
            except Exception:
                # fallback to per-item
                sar_preds = []
                for t in batch_texts:
                    try:
                        p = sarcasm_pipeline(t, truncation=True, max_length=512)
                        if isinstance(p, list) and len(p) > 0:
                            sar_preds.append(p[0])
                        else:
                            sar_preds.append(p)
                    except Exception:
                        sar_preds.append(None)
        else:
            sar_preds = [None] * len(batch_texts)

        for t, c, sp in zip(batch_texts, batch_comments, sar_preds):
            heuristic_flag = _heuristic_sarcasm_check(t)
            sar_label = False
            sar_score = 0.0
            if sp is not None:
                lab = (sp.get("label", "") or "").lower()
                score = float(sp.get("score", 0.0) or 0.0)
                if ("sar" in lab or "iron" in lab or "irony" in lab) and score >= sarcasm_confidence_threshold:
                    sar_label = True
                    sar_score = score
            final_sar = bool(heuristic_flag or sar_label)
            c["_sarcasm"] = final_sar
            c["_sarcasm_score"] = round(float(sar_score), 4)

            # Optional inversion: flip sentiment if sarcasm detected and sentiment is clear
            if invert_on_sarcasm and final_sar and c.get("_sentiment_label") in ("Positive", "Negative"):
                orig = c["_sentiment_label"]
                c["_sentiment_label"] = "Negative" if orig == "Positive" else "Positive"

    return comments


# -------------------------
# Wordcloud builder
# -------------------------
def build_wordcloud(comments: List[Dict[str, Any]], max_words: int = 200) -> Optional[str]:
    """
    Build wordcloud image from comment texts. Returns path to PNG file or None if WordCloud not available.
    """
    if not WORDCLOUD_AVAILABLE:
        _warn("wordcloud package not available (pip install wordcloud). Skipping wordcloud.")
        return None

    texts = " ".join([c.get("text", "") or "" for c in comments])
    if not texts.strip():
        return None

    stopwords = set(STOPWORDS)
    stopwords.update(["https", "http", "amp", "RT"])
    wc = WordCloud(width=1200, height=600, background_color="white", stopwords=stopwords, max_words=max_words)
    wc.generate(texts)
    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    wc.to_file(tmpf.name)
    return tmpf.name


# -------------------------
# Top N comments helper
# -------------------------
def top_n_comments(comments: List[Dict[str, Any]], n: int = 20, sort_by: str = "like_count") -> List[Dict[str, Any]]:
    """
    Return top-n comments sorted by sort_by (like_count by default). Each returned dict includes:
      id, author, text, like_count, published_at, sentiment, sentiment_score, sarcasm, sarcasm_score, is_reply, parent_id
    """
    valid = [c for c in comments if c.get("text")]
    sorted_list = sorted(valid, key=lambda x: x.get(sort_by, 0) or 0, reverse=True)
    out = []
    for c in sorted_list[:n]:
        out.append({
            "id": c.get("id"),
            "author": c.get("author"),
            "text": c.get("text"),
            "like_count": c.get("like_count", 0),
            "published_at": c.get("published_at"),
            "sentiment": c.get("_sentiment_label"),
            "sentiment_score": c.get("_sentiment_score"),
            "sarcasm": c.get("_sarcasm", False),
            "sarcasm_score": c.get("_sarcasm_score", 0.0),
            "is_reply": c.get("is_reply", False),
            "parent_id": c.get("parent_id"),
        })
    return out
