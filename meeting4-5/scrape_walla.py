"""
Walla Article Scraper  —  Test Set
====================================
Scrapes Hebrew news articles from walla.co.il across 4 categories
to serve as a held-out test set (trained on Ynet, tested on Walla).

Categories and RSS feeds used:
  news    (חדשות)      feed/1  → news.walla.co.il
  sport   (ספורט)      feed/7  → sports.walla.co.il
  tech    (טכנולוגיה)  feed/4  → tech.walla.co.il
  economy (כלכלה)      feed/3  → finance.walla.co.il

Important note on Walla's architecture
---------------------------------------
Walla renders article body content via React (client-side), so a plain
requests.get() call may not return the article paragraphs.

Strategy:
  1. Parse each category's RSS feed with feedparser — this gives article
     URLs and a short Hebrew summary (2-4 sentences) per article.
  2. For each URL, attempt to extract the full article body from the
     static HTML using BeautifulSoup (.article-content > p).
  3. If extraction yields < MIN_TEXT_LENGTH chars, fall back to the RSS
     description (the summary).  Mark the article with source="rss_summary"
     so the notebook can filter or flag short articles.

Usage:
    pip install feedparser requests beautifulsoup4

    python scrape_walla.py                   # ~15 articles per category
    python scrape_walla.py --per-class 25    # more (needs multiple RSS pages)

Output:
    meeting4-5/data/walla_articles.json
"""

import re
import json
import time
import argparse
import datetime
from pathlib import Path

import requests
import feedparser
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CATEGORIES = {
    "news":    {"he": "חדשות",      "rss_feeds": ["https://rss.walla.co.il/feed/1"]},
    "sport":   {"he": "ספורט",      "rss_feeds": ["https://rss.walla.co.il/feed/7"]},
    "tech":    {"he": "טכנולוגיה",  "rss_feeds": ["https://rss.walla.co.il/feed/4"]},
    "economy": {"he": "כלכלה",      "rss_feeds": ["https://rss.walla.co.il/feed/3"]},
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "he-IL,he;q=0.9,en-US;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

SLEEP_BETWEEN_REQUESTS = 1.5
MIN_TEXT_LENGTH        = 60    # chars — below this, fall back to RSS summary
OUTPUT_PATH = Path(__file__).parent / "data" / "walla_articles.json"


# ---------------------------------------------------------------------------
# Step 1: Parse RSS feed  →  list of {url, title, rss_summary}
# ---------------------------------------------------------------------------

def parse_rss_feed(feed_url: str) -> list[dict]:
    """
    Use feedparser to get article URLs and their RSS description text.
    Strips HTML tags from the description (Walla wraps summaries in <p>).
    """
    print(f"  Parsing RSS: {feed_url}")
    parsed = feedparser.parse(feed_url)

    entries = []
    for entry in parsed.entries:
        url   = entry.get("link", "")
        title = entry.get("title", "").strip()

        # description may contain inline HTML — strip it
        raw_desc = entry.get("summary", "") or entry.get("description", "")
        soup_desc = BeautifulSoup(raw_desc, "html.parser")
        summary = soup_desc.get_text(separator=" ", strip=True)

        if url and (title or summary):
            entries.append({"url": url, "title": title, "rss_summary": summary})

    print(f"  Found {len(entries)} entries in feed")
    return entries


# ---------------------------------------------------------------------------
# Step 2: Attempt full article extraction from the article page
# ---------------------------------------------------------------------------

def extract_article_page(url: str) -> str:
    """
    Try to extract body text from the article page HTML.
    Walla uses .article-content > p for the article body.
    Returns empty string if extraction fails (JS-rendered page).
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"    [WARN] Fetch failed for {url}: {e}")
        return ""

    soup = BeautifulSoup(resp.content, "html.parser")
    parts = []

    # Author + breadcrumb (adds a little noise — good for the lesson)
    for cls in ("itemAuthor", "writer-and-time", "categoryName", "tags-and-breadcrumbs"):
        el = soup.find(class_=cls)
        if el:
            parts.append(el.get_text(separator=" ", strip=True))

    # Primary body selector
    body = soup.find(class_="article-content")
    if body:
        for p in body.find_all("p"):
            t = p.get_text(separator=" ", strip=True)
            if t:
                parts.append(t)

    # Fallback: any <article> tag
    if not parts or sum(len(p) for p in parts) < MIN_TEXT_LENGTH:
        article_tag = soup.find("article")
        if article_tag:
            parts.append(article_tag.get_text(separator=" ", strip=True))

    return " ".join(parts).strip()


# ---------------------------------------------------------------------------
# Step 3: Main scraping loop
# ---------------------------------------------------------------------------

def scrape(articles_per_category: int = 15) -> list[dict]:
    all_articles = []

    for category, meta in CATEGORIES.items():
        print(f"\n{'='*60}")
        print(f"Category: {category} ({meta['he']})")
        print(f"Target: {articles_per_category} articles")
        print(f"{'='*60}")

        # Gather candidate entries from all RSS feeds for this category
        candidates = []
        for rss_url in meta["rss_feeds"]:
            candidates.extend(parse_rss_feed(rss_url))
            time.sleep(SLEEP_BETWEEN_REQUESTS)

        # De-duplicate by URL
        seen, unique = set(), []
        for c in candidates:
            if c["url"] not in seen:
                seen.add(c["url"])
                unique.append(c)

        print(f"  {len(unique)} unique articles available in feed(s)")

        if len(unique) < articles_per_category:
            print(
                f"  [WARN] Only {len(unique)} articles in RSS — "
                f"reduce --per-class or add more feed IDs."
            )

        collected = []
        for entry in unique[:articles_per_category]:
            url     = entry["url"]
            title   = entry["title"]
            summary = entry["rss_summary"]

            print(f"  [{len(collected)+1}] {url[-40:]}")

            # Attempt full-page extraction
            page_text = extract_article_page(url)
            time.sleep(SLEEP_BETWEEN_REQUESTS)

            if len(page_text) >= MIN_TEXT_LENGTH:
                text   = page_text
                source = "article_page"
            else:
                # Fall back to RSS summary
                text   = f"{title}. {summary}".strip()
                source = "rss_summary"
                print(f"    → page empty, using RSS summary ({len(text)} chars)")

            if len(text) < MIN_TEXT_LENGTH // 2:
                print(f"    [SKIP] text too short")
                continue

            collected.append({
                "url":         url,
                "title":       title,
                "text":        text,
                "category":    category,
                "category_he": meta["he"],
                "source":      source,        # "article_page" or "rss_summary"
                "scraped_at":  datetime.date.today().isoformat(),
            })

        print(f"  Collected {len(collected)} articles for '{category}'")
        print(f"  Sources: { {s: sum(1 for a in collected if a['source']==s) for s in ('article_page','rss_summary')} }")
        all_articles.extend(collected)

    return all_articles


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scrape Walla articles for use as a cross-site test set."
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=15,
        help=(
            "Articles per category (default: 15). "
            "Each RSS feed has ~20-30 items, so values above 25 "
            "may require adding extra feed IDs in CATEGORIES."
        ),
    )
    args = parser.parse_args()

    print(f"Scraping {args.per_class} articles per category from Walla...")
    articles = scrape(articles_per_category=args.per_class)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    print(f"\nDone.  {len(articles)} articles → {OUTPUT_PATH}")

    from collections import Counter
    cats    = Counter(a["category"] for a in articles)
    sources = Counter(a["source"]   for a in articles)
    print("\nArticles per category:")
    for cat, n in cats.items():
        print(f"  {cat:10s}: {n}")
    print(f"\nSources: {dict(sources)}")
    print(
        "\nNote: 'rss_summary' articles are shorter (2-4 sentences). "
        "They still work for classification but produce weaker signal. "
        "If most articles fell back to RSS summary, Walla is fully "
        "JS-rendered on this server — consider using a headless browser "
        "(playwright / selenium) for full article text."
    )


if __name__ == "__main__":
    main()
