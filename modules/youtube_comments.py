# modules/youtube_comments.py
"""
YouTube comments fetch + process for Streamlit.
- Uses st.secrets['YOUTUBE_API_KEY'] for API key.
- Exposes `run_analysis(video_input, max_comments=5000)` which:
    returns (df_processed, kpi_dict, wordcloud_pos_png_bytes_or_None, wordcloud_neg_png_bytes_or_None)
"""
import time
import json
import re
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
from io import BytesIO

import pandas as pd
import streamlit as st

# Google client
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Sentiment + lang detection + wordcloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect, DetectorFactory
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Setup
DetectorFactory.seed = 0
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Local data dirs (optional; used if you want to save files to disk)
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
OUT_DIR = DATA_DIR / "outputs"
for p in (RAW_DIR, PROC_DIR, OUT_DIR):
    p.mkdir(parents=True, exist_ok=True)

def _get_api_key() -> str:
    key = None
    # prefer Streamlit secrets
    try:
        key = st.secrets.get("YOUTUBE_API_KEY")  # returns None if not set
    except Exception:
        key = None
    if not key:
        raise RuntimeError("YOUTUBE_API_KEY not found in Streamlit secrets. Add it in Streamlit Cloud -> Settings -> Secrets.")
    return key

def youtube_client():
    key = _get_api_key()
    return build("youtube", "v3", developerKey=key)

def clean_text(t: str) -> str:
    t = str(t)
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"&amp;|&lt;|&gt;", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def safe_lang(s: str) -> str:
    try:
        return detect(s) if s and s.strip() else "unknown"
    except Exception:
        return "unknown"

def _extract_video_id(url_or_id: str) -> str:
    # robust extraction for youtube urls and short urls
    s = str(url_or_id).strip()
    if "youtu" in s:
        # classic v= extraction
        if "v=" in s:
            return s.split("v=")[-1].split("&")[0]
        # short youtu.be/
        if "youtu.be/" in s:
            return s.split("youtu.be/")[-1].split("?")[0]
        # embed urls
        if "/embed/" in s:
            return s.split("/embed/")[-1].split("?")[0]
    return s  # assume it's already an id

def fetch_youtube_comments(video_id: str, max_pages: Optional[int] = None, max_comments: int = 5000) -> List[dict]:
    """
    Fetch comments and replies using YouTube Data API v3.
    Logic is adapted from your original script; returns list of dicts.
    """
    yt = youtube_client()
    comments: List[dict] = []
    token = None
    page = 0

    while True:
        try:
            resp = yt.commentThreads().list(
                part="snippet,replies",
                videoId=video_id,
                maxResults=100,
                pageToken=token,
                textFormat="plainText"
            ).execute()
        except HttpError as e:
            logging.warning(f"HttpError while fetching comments: {e}. Sleeping 5s and retrying once.")
            time.sleep(5)
            try:
                resp = yt.commentThreads().list(
                    part="snippet,replies",
                    videoId=video_id,
                    maxResults=100,
                    pageToken=token,
                    textFormat="plainText"
                ).execute()
            except Exception as e2:
                logging.error("Second attempt failed: %s", e2)
                break

        items = resp.get("items", [])
        for item in items:
            tlc = item.get("snippet", {}).get("topLevelComment", {})
            top_snip = tlc.get("snippet", {})
            top_id = tlc.get("id") or top_snip.get("id")
            comments.append({
                "comment_id": top_id,
                "author": top_snip.get("authorDisplayName"),
                "comment": top_snip.get("textDisplay"),
                "likes": top_snip.get("likeCount", 0),
                "published": top_snip.get("publishedAt"),
                "video_id": video_id,
                "parent_id": None,
                "raw": top_snip
            })
            if item.get("replies"):
                for r in item["replies"].get("comments", []):
                    rs_snip = r.get("snippet", {})
                    reply_id = r.get("id") or rs_snip.get("id")
                    comments.append({
                        "comment_id": reply_id,
                        "author": rs_snip.get("authorDisplayName"),
                        "comment": rs_snip.get("textDisplay"),
                        "likes": rs_snip.get("likeCount", 0),
                        "published": rs_snip.get("publishedAt"),
                        "video_id": video_id,
                        "parent_id": top_id,
                        "raw": rs_snip
                    })

        page += 1
        logging.info("Fetched page %d, total comments so far: %d", page, len(comments))

        if len(comments) >= max_comments:
            logging.info("Reached limit of %d comments, stopping.", max_comments)
            break

        token = resp.get("nextPageToken")
        if not token or (max_pages and page >= max_pages):
            break

        time.sleep(0.1)

    return comments[:max_comments]

def vader_label(c: float) -> str:
    if c >= 0.05:
        return "Positive"
    elif c <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def _make_wordcloud_bytes(text_series, sentiment_label: str) -> Optional[bytes]:
    stopwords = set(STOPWORDS)
    stopwords.update(["https", "http", "amp", "video", "youtube", "watch", "com"])
    texts = [str(t).lower() for t in text_series if isinstance(t, str) and len(str(t).strip()) > 1]
    text = " ".join(texts)
    if not text.strip():
        return None
    sentiment = sentiment_label.lower()
    if sentiment == "positive":
        colormap = "Greens"
    elif sentiment == "negative":
        colormap = "Reds"
    else:
        colormap = "Greys"

    wc = WordCloud(width=1000, height=600, background_color="white",
                   stopwords=stopwords, colormap=colormap, max_words=200).generate(text)
    fig = plt.figure(figsize=(10, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Most Frequent Words in {sentiment_label} Comments", fontsize=14)
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def run_analysis(video_input: str, max_pages: Optional[int] = None, max_comments: int = 2000) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[bytes], Optional[bytes]]:
    """
    Main entry for Streamlit pages.
    video_input: youtube url or id
    returns: (df_processed, kpi_dict, wordcloud_pos_bytes, wordcloud_neg_bytes)
    """
    video_id = _extract_video_id(video_input)
    ts = int(time.time())

    comments = fetch_youtube_comments(video_id, max_pages=max_pages, max_comments=max_comments)
    if not comments:
        raise RuntimeError("No comments fetched - check video id or API quota.")

    # optionally save raw json
    raw_path = RAW_DIR / f"comments_{video_id}_{ts}.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(comments, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(comments).drop_duplicates(subset=["comment_id"])
    df["clean_comment"] = df["comment"].fillna("").apply(clean_text)
    df["lang"] = df["clean_comment"].apply(safe_lang)
    df["published"] = pd.to_datetime(df["published"], errors="coerce")

    analyzer = SentimentIntensityAnalyzer()
    scores = df["clean_comment"].apply(lambda x: analyzer.polarity_scores(str(x)))
    scores_df = pd.DataFrame(list(scores))
    df = pd.concat([df.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1)
    df["sentiment_vader"] = df["compound"].apply(vader_label)

    out_csv = PROC_DIR / f"processed_{video_id}_{ts}.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")

    kpi = {
        "video_id": video_id,
        "timestamp": datetime.utcnow().isoformat(),
        "n_comments": int(len(df)),
        "pct_negative": float((df["sentiment_vader"] == "Negative").mean() * 100),
        "pct_positive": float((df["sentiment_vader"] == "Positive").mean() * 100),
        "pct_neutral": float((df["sentiment_vader"] == "Neutral").mean() * 100),
        "avg_compound": float(df["compound"].mean())
    }

    # wordcloud bytes
    pos_wc = _make_wordcloud_bytes(df[df["sentiment_vader"] == "Positive"]["clean_comment"], "Positive")
    neg_wc = _make_wordcloud_bytes(df[df["sentiment_vader"] == "Negative"]["clean_comment"], "Negative")

    # save KPI quick file (optional)
    pd.DataFrame([kpi]).to_csv(OUT_DIR / f"kpi_{video_id}_{ts}.csv", index=False)

    return df, kpi, pos_wc, neg_wc



    