"""
collector.py — Data collection via PipeScraper + report generation via Groq/LangChain
"""

import os
import json
import re
from datetime import datetime
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from src.report_store import save_report

# ── Topic → search keyword mapping ───────────────────────────────────────────────
TOPIC_KEYWORDS = {
    "AI Regulation":      ["AI regulation policy", "artificial intelligence law"],
    "Technology Updates": ["latest technology news", "tech innovation 2025"],
    "UK Economy":         ["UK economy news", "Bank of England inflation"],
    "Cybersecurity":      ["cybersecurity threats", "data breach news"],
    "Climate & Energy":   ["climate change policy", "renewable energy news"],
}

# ── BBC fallback URLs (keyword-agnostic) ─────────────────────────────────────────
BBC_FEEDS = {
    "BBC Technology": "https://www.bbc.co.uk/news/technology",
    "BBC Science":    "https://www.bbc.co.uk/news/science-environment",
    "UK Gov":         "https://www.gov.uk/search/news-and-communications",
}


def _scrape_articles(topic: str, sources: list[str], max_articles: int = 12) -> list[dict]:
    """
    Use PipeScraper to collect articles. Falls back gracefully if unavailable.
    Returns a list of article dicts with keys: title, url, text, source.
    """
    import concurrent.futures
    articles = []

    def fetch_google_news():
        results = []
        try:
            from pipescraper import FetchGoogleNews, ExtractArticles, ToDataFrame

            keywords = TOPIC_KEYWORDS.get(topic, [topic])
            raw = (
                FetchGoogleNews(search=keywords, period="1d", max_results=8, print_url=False)
                >> ExtractArticles(workers=4, skip_errors=True, print_url=False, timeout=15)
                >> ToDataFrame(include_text=True)
            )
            if raw is not None and len(raw) > 0:
                for _, row in raw.iterrows():
                    results.append({
                        "title":  str(row.get("title", "")).strip() or "No title",
                        "url":    str(row.get("url", "")),
                        "text":   str(row.get("text", ""))[:3000],
                        "source": str(row.get("source", "Google News")),
                    })
        except Exception as e:
            print(f"[PipeScraper/GoogleNews] {e}")
        return results

    def fetch_bbc_feed(source_label):
        results = []
        try:
            from pipescraper import FetchLinks, ExtractArticles, ToDataFrame, FilterArticles

            url = BBC_FEEDS[source_label]
            raw = (
                url
                >> FetchLinks(max_links=10, respect_robots=True, delay=0.5, timeout=10)
                >> ExtractArticles(workers=3, skip_errors=True, print_url=False, timeout=15)
                >> FilterArticles(lambda a: len(a.text or "") > 200)
                >> ToDataFrame(include_text=True)
            )
            if raw is not None and len(raw) > 0:
                for _, row in raw.iterrows():
                    results.append({
                        "title":  str(row.get("title", "")).strip() or "No title",
                        "url":    str(row.get("url", "")),
                        "text":   str(row.get("text", ""))[:3000],
                        "source": source_label,
                    })
        except Exception as e:
            print(f"[PipeScraper/{source_label}] {e}")
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []

        # ── 1. Google News via FetchGoogleNews (PipeScraper v0.3.0) ──────────────────
        if "Google News" in sources:
            futures.append(executor.submit(fetch_google_news))

        # ── 2. BBC / Gov sites via FetchLinks + ExtractArticles ──────────────────────
        for source_label in sources:
            if source_label in BBC_FEEDS:
                futures.append(executor.submit(fetch_bbc_feed, source_label))

        for future in concurrent.futures.as_completed(futures):
            articles.extend(future.result())

    # ── 3. Fallback: BBC RSS via requests if PipeScraper fully unavailable ────────
    if not articles:
        articles = _fallback_rss(topic)

    # Deduplicate by URL and cap
    seen, deduped = set(), []
    for a in articles:
        if a["url"] not in seen:
            seen.add(a["url"])
            deduped.append(a)
    return deduped[:max_articles]


def _fallback_rss(topic: str) -> list[dict]:
    """Minimal fallback: pull BBC RSS and parse with feedparser or requests+bs4."""
    fallback_articles = []
    rss_urls = {
        "AI Regulation":      "http://feeds.bbci.co.uk/news/technology/rss.xml",
        "Technology Updates": "http://feeds.bbci.co.uk/news/technology/rss.xml",
        "UK Economy":         "http://feeds.bbci.co.uk/news/business/rss.xml",
        "Cybersecurity":      "http://feeds.bbci.co.uk/news/technology/rss.xml",
        "Climate & Energy":   "http://feeds.bbci.co.uk/news/science-environment/rss.xml",
    }
    feed_url = rss_urls.get(topic, "http://feeds.bbci.co.uk/news/technology/rss.xml")
    try:
        import requests
        from bs4 import BeautifulSoup
        resp = requests.get(feed_url, timeout=10, headers={"User-Agent": "GenAIAgent/1.0"})
        soup = BeautifulSoup(resp.content, "xml")
        for item in soup.find_all("item")[:10]:
            title = item.find("title")
            link  = item.find("link")
            desc  = item.find("description")
            fallback_articles.append({
                "title":  title.text.strip() if title else "No title",
                "url":    link.text.strip()  if link  else "",
                "text":   desc.text.strip()  if desc  else "",
                "source": "BBC RSS (fallback)",
            })
    except Exception as e:
        print(f"[RSS Fallback] {e}")
    return fallback_articles


import time

# Token tracking to pace requests within the Groq 6000 TPM limit
_SESSION_TOKENS = 0

def _generate_report(articles: list[dict], topic: str) -> dict:
    """
    Use LangChain + Groq to produce a structured JSON report, completely pacing and handling Groq Rate Limits.
    """
    global _SESSION_TOKENS
    
    api_key = os.getenv("GROQ_API_KEY", "")
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=1024,
        groq_api_key=api_key,
    )

    # Build a compact digest of article content for the prompt
    digest = ""
    for i, art in enumerate(articles[:10], 1):
        digest += f"\n[{i}] {art['title']} (Source: {art['source']})\n{art['text'][:600]}\n"

    # --- TOKEN PACING LOGIC ---
    # Roughly estimate tokens (4 chars/token input + 1000 buffer for generated output)
    estimated_tokens = len(digest) // 4 + 1000
    
    if _SESSION_TOKENS + estimated_tokens > 6000:
        print(f"[{datetime.now():%H:%M:%S}] ⚠️ Reaching 6000 token limit ({_SESSION_TOKENS} + {estimated_tokens}). Pausing for 60s to refresh Groq RPM bucket...")
        time.sleep(60)
        _SESSION_TOKENS = 0  # Reset token bucket after wait
        
    _SESSION_TOKENS += estimated_tokens

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert news analyst. Given a set of recent articles,
produce a structured intelligence report as valid JSON (no markdown fences).

Return ONLY this JSON structure:
{{
  "summary": "<100-150 word paragraph summarising the key developments>",
  "takeaways": ["<takeaway 1>", "<takeaway 2>", "<takeaway 3>", "<takeaway 4>", "<takeaway 5>"],
  "entities": ["<org or person 1>", "<org or person 2>", ...],
  "key_topics": ["<topic 1>", "<topic 2>", ...]
}}

Rules:
- summary must be 100-150 words, written in professional analyst style
- takeaways: exactly 3-5 concise bullet strings
- entities: organisations, governments, companies, or notable persons mentioned
- key_topics: 4-8 short topic tags (e.g. "Machine Learning", "Data Privacy")
- Return ONLY the JSON object, nothing else"""),
        ("human", "Topic: {topic}\n\nArticles:\n{digest}"),
    ])

    chain = prompt | llm
    
    # --- RATE LIMIT FALLBACK LOOP ---
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = chain.invoke({"topic": topic, "digest": digest})
            break
        except Exception as e:
            err_msg = str(e).lower()
            if "rate limit" in err_msg or "429" in err_msg or "too many requests" in err_msg:
                # Attempt to parse specific Groq wait time from error e.g. "Please try again in 12m30.816s"
                wait_time = 60.0
                match = re.search(r"try again in (?:(\d+)m)?([\d\.]+)s", err_msg)
                if match:
                    mins = int(match.group(1)) if match.group(1) else 0
                    secs = float(match.group(2))
                    wait_time = (mins * 60) + secs + 2.0  # Add 2s safe buffer
                
                print(f"[{datetime.now():%H:%M:%S}] 🕒 Rate Limit Error 429 caught! Sleeping for {wait_time:.1f} seconds before retrying...")
                time.sleep(wait_time)
                _SESSION_TOKENS = estimated_tokens  # Start new bucket window
                if attempt == max_retries - 1:
                    raise e
            else:
                raise e

    raw = result.content.strip()

    # Strip markdown fences if model adds them anyway
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Graceful degradation
        return {
            "summary": raw[:500],
            "takeaways": ["Could not parse structured output — see raw summary above."],
            "entities": [],
            "key_topics": [topic],
        }


def collect_and_report(
    topic: str = "AI Regulation",
    sources: list[str] | None = None,
) -> dict:
    """
    Full pipeline: scrape → generate report → persist.
    Returns the report dict.
    """
    if sources is None:
        sources = ["BBC Technology", "Google News"]

    print(f"[{datetime.now():%H:%M:%S}] Collecting articles for topic: '{topic}'")
    articles = _scrape_articles(topic=topic, sources=sources)
    print(f"[{datetime.now():%H:%M:%S}] Collected {len(articles)} articles — generating report...")

    llm_report = _generate_report(articles, topic)

    report = {
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "topic":         topic,
        "article_count": len(articles),
        "summary":       llm_report.get("summary", ""),
        "takeaways":     llm_report.get("takeaways", []),
        "entities":      llm_report.get("entities", []),
        "key_topics":    llm_report.get("key_topics", []),
        "articles": [
            {"title": a["title"], "url": a["url"], "source": a["source"]}
            for a in articles
        ],
    }

    save_report(report)
    print(f"[{datetime.now():%H:%M:%S}] Report saved")
    return report
