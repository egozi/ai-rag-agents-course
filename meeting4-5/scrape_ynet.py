"""
Ynet Article Scraper
====================
Scrapes Hebrew news articles from ynet.co.il across 4 categories:
  - news    (חדשות)
  - sport   (ספורט)
  - tech    (טכנולוגיה)
  - economy (כלכלה)

Usage:
    python scrape_ynet.py                  # 35 articles per category (default)
    python scrape_ynet.py --per-class 10   # quick test run

Output:
    meeting4-5/data/ynet_articles.json

Notes:
- Sleeps 1.5s between requests to avoid rate limiting.
- Skips articles where extraction fails (short text < 50 chars).
- Raw text is intentionally minimally cleaned so the notebook can
  demonstrate the full preprocessing pipeline.
"""

import re
import json
import time
import argparse
import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CATEGORIES = {
    "news":    {
        "he": "חדשות",
        "listing_urls": [
            "https://www.ynet.co.il/news/category/184",
        ],
    },
    "sport":   {
        "he": "ספורט",
        "listing_urls": [
            "https://www.ynet.co.il/sport",
            "https://www.ynet.co.il/sport/category/3",
        ],
    },
    "tech":    {
        "he": "טכנולוגיה",
        "listing_urls": [
            "https://www.ynet.co.il/digital/technews",
        ],
    },
    "economy": {
        "he": "כלכלה",
        "listing_urls": [
            "https://www.ynet.co.il/economy",
            "https://www.ynet.co.il/economy/category/429",
        ],
    },
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "he-IL,he;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

SLEEP_BETWEEN_REQUESTS = 1.5   # seconds
MIN_TEXT_LENGTH = 80            # skip articles shorter than this (chars)
OUTPUT_PATH = Path(__file__).parent / "data" / "ynet_articles.json"


# ---------------------------------------------------------------------------
# Step 1: Collect article URLs from a category listing page
# ---------------------------------------------------------------------------

def get_article_urls_from_listing(listing_url: str) -> list[str]:
    """
    Ynet category pages inject article metadata as JSON inside <script> tags
    (the YITSiteWidgets pattern).  We also fall back to a simple regex over
    all href attributes.  Returns de-duplicated absolute URLs.
    """
    try:
        resp = requests.get(listing_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [WARN] Could not fetch listing {listing_url}: {e}")
        return []

    html = resp.text
    urls = set()

    # Strategy 1: parse shareUrl values embedded in JSON blobs
    for match in re.finditer(r'"shareUrl"\s*:\s*"(https?://[^"]+/article/[^"]+)"', html):
        urls.add(match.group(1))

    # Strategy 2: plain href scan for /article/ paths
    for match in re.finditer(
        r'href=["\']?(https?://www\.ynet\.co\.il/[^"\'>\s]+/article/[^"\'>\s]+)',
        html
    ):
        urls.add(match.group(1))

    # Strategy 3: relative paths
    for match in re.finditer(r'href=["\']?(/[^"\'>\s]+/article/[^"\'>\s]+)', html):
        urls.add("https://www.ynet.co.il" + match.group(1))

    return list(urls)


# ---------------------------------------------------------------------------
# Step 2: Extract title + body text from a single article page
# ---------------------------------------------------------------------------

def extract_article(url: str) -> dict | None:
    """
    Fetches an article page and extracts:
      - title  : from class 'mainTitle'
      - text   : from ArticleBodyComponent (with intentional noise:
                 author, breadcrumb, and any stray nav text that slips
                 through imperfect selectors)

    Returns None if extraction fails or text is too short.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"    [WARN] Could not fetch article {url}: {e}")
        return None

    soup = BeautifulSoup(resp.content, "html.parser")

    # --- Title ---
    title_el = (
        soup.find(class_="mainTitle")
        or soup.find("h1")
    )
    title = title_el.get_text(separator=" ", strip=True) if title_el else ""

    # --- Body text ---
    # We intentionally cast a slightly wider net than just text_editor_paragraph
    # so that author bylines, breadcrumbs, and related-article labels
    # bleed into the raw text. This gives the notebook realistic noise to clean.
    raw_parts = []

    # Breadcrumb / channel name
    breadcrumb = soup.find(class_="dcPath") or soup.find(class_="channelName")
    if breadcrumb:
        raw_parts.append(breadcrumb.get_text(separator=" ", strip=True))

    # Author byline
    author = soup.find(class_="authorName") or soup.find(class_="author")
    if author:
        raw_parts.append(author.get_text(separator=" ", strip=True))

    # Article body
    body_el = (
        soup.find(id="ArticleBodyComponent")
        or soup.find(class_="ArticleBodyComponent")
    )
    if body_el:
        # Grab all paragraph-like elements inside the body
        for el in body_el.find_all(
            class_=re.compile(r"text_editor_paragraph|article_body|paragraphs")
        ):
            raw_parts.append(el.get_text(separator=" ", strip=True))

        # Fallback: if no paragraphs found, take all text from the container
        if not raw_parts or sum(len(p) for p in raw_parts) < MIN_TEXT_LENGTH:
            raw_parts.append(body_el.get_text(separator=" ", strip=True))
    else:
        # Last resort: take the full page text (very noisy — good for lesson)
        article_tag = soup.find("article") or soup.find(class_=re.compile(r"article"))
        if article_tag:
            raw_parts.append(article_tag.get_text(separator=" ", strip=True))

    text = " ".join(raw_parts).strip()

    if len(text) < MIN_TEXT_LENGTH:
        print(f"    [SKIP] Too short ({len(text)} chars): {url}")
        return None

    return {
        "url": url,
        "title": title,
        "text": text,          # raw — intentionally noisy
        "scraped_at": datetime.date.today().isoformat(),
    }


# ---------------------------------------------------------------------------
# Step 3: Main scraping loop
# ---------------------------------------------------------------------------

def scrape(articles_per_category: int = 35) -> list[dict]:
    all_articles = []

    for category, meta in CATEGORIES.items():
        print(f"\n{'='*60}")
        print(f"Category: {category} ({meta['he']})")
        print(f"Target: {articles_per_category} articles")
        print(f"{'='*60}")

        # Collect candidate URLs from all listing pages for this category
        candidate_urls = []
        for listing_url in meta["listing_urls"]:
            print(f"  Fetching listing: {listing_url}")
            urls = get_article_urls_from_listing(listing_url)
            print(f"  Found {len(urls)} candidate URLs")
            candidate_urls.extend(urls)
            time.sleep(SLEEP_BETWEEN_REQUESTS)

        # De-duplicate while preserving order
        seen = set()
        unique_urls = []
        for u in candidate_urls:
            if u not in seen:
                seen.add(u)
                unique_urls.append(u)

        print(f"  Total unique URLs: {len(unique_urls)}")

        # Fetch articles until we have enough
        collected = []
        for url in unique_urls:
            if len(collected) >= articles_per_category:
                break

            print(f"  [{len(collected)+1}/{articles_per_category}] {url[:80]}")
            article = extract_article(url)
            time.sleep(SLEEP_BETWEEN_REQUESTS)

            if article is None:
                continue

            article["category"] = category
            article["category_he"] = meta["he"]
            collected.append(article)

        print(f"  Collected {len(collected)} articles for '{category}'")
        all_articles.extend(collected)

    return all_articles


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scrape Ynet articles by category.")
    parser.add_argument(
        "--per-class",
        type=int,
        default=35,
        help="Number of articles to collect per category (default: 35)",
    )
    args = parser.parse_args()

    print(f"Scraping {args.per_class} articles per category...")
    articles = scrape(articles_per_category=args.per_class)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    print(f"\nDone. {len(articles)} articles saved to {OUTPUT_PATH}")

    # Summary
    from collections import Counter
    counts = Counter(a["category"] for a in articles)
    print("\nArticles per category:")
    for cat, n in counts.items():
        print(f"  {cat:10s}: {n}")


if __name__ == "__main__":
    main()
