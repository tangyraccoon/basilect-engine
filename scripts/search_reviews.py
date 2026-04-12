"""Automated review/feature URL discovery for artists using configurable search backends.

Mirrors search.py but targets critic-written content about artists.
Searches music review sites for reviews, features, and critical essays.
"""

import sys
import json
import re
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

# Import search provider abstraction from search.py
import importlib.util
spec = importlib.util.spec_from_file_location("search", "scripts/search.py")
search_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(search_module)

SearchProvider = search_module.SearchProvider
BraveSearchProvider = search_module.BraveSearchProvider
SerpAPIProvider = search_module.SerpAPIProvider
GoogleCSEProvider = search_module.GoogleCSEProvider
get_provider = search_module.get_provider
normalize_artist_id = search_module.normalize_artist_id
load_blocked_domains = search_module.load_blocked_domains
is_blocked_domain = search_module.is_blocked_domain

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)


# Review-focused domains to prioritize
REVIEW_SITES = {
    'pitchfork.com',
    'thequietus.com',
    'stereogum.com',
    'residentadvisor.net',
    'tinymixtapes.com',
    'drownedinsound.com',
    'noisey.vice.com',
    'albumoftheyear.org',
    'sputnikmusic.com',
    'nme.com',
    'thelineofbestfit.com',
    'popmatters.com',
    'consequence.net',
}

# Video/audio platforms to skip (same as search.py)
VIDEO_AUDIO_DOMAINS = {
    'youtube.com', 'youtu.be', 'vimeo.com', 'soundcloud.com',
    'spotify.com', 'bandcamp.com', 'tiktok.com', 'instagram.com',
    'twitch.tv', 'dailymotion.com', 'apple.com/music', 'music.amazon.com'
}


def is_video_audio_url(url: str) -> bool:
    """Check if URL is a video or audio platform."""
    domain = urlparse(url).netloc.lower()
    for blocked_domain in VIDEO_AUDIO_DOMAINS:
        if blocked_domain in domain:
            return True
    return False


def score_review_result(result: dict) -> float:
    """Score result by likely review quality."""
    snippet = result.get("snippet", "").lower()
    title = result.get("title", "").lower()
    url = result.get("url", "").lower()

    score = 0.0

    # Longer snippets suggest more substantial content
    score += len(snippet) / 100.0

    # Review keywords
    keywords = ["review", "feature", "interview", "essay", "critique",
                "analysis", "album review", "music criticism", "music critic",
                "artist profile", "critical assessment"]
    for kw in keywords:
        if kw in snippet or kw in title:
            score += 2.0

    # Bonus for review site domains
    domain = urlparse(url).netloc.lower()
    for review_site in REVIEW_SITES:
        if review_site in domain:
            score += 3.0
            break

    # Avoid fluff and non-reviews
    avoid = ["top 10", "best of", "ranking", "best moments", "countdown"]
    for word in avoid:
        if word in title:
            score -= 1.0

    return max(0, score)


def deduplicate_by_domain(results: list, max_per_domain: int = 2) -> list:
    """Deduplicate results, keeping max N per publication domain."""
    seen = {}
    deduped = []

    # Sort by domain first to group them
    by_domain = {}
    for r in results:
        domain = urlparse(r["url"]).netloc.lower()
        if domain not in by_domain:
            by_domain[domain] = []
        by_domain[domain].append(r)

    # Keep top N from each domain
    for domain, items in sorted(by_domain.items()):
        deduped.extend(items[:max_per_domain])

    return deduped


def search_for_reviews(artist_name: str, provider: SearchProvider,
                      blocked_domains: set, num_results: int = 6) -> list:
    """Search for review/feature URLs about an artist."""

    # Multiple queries to find critic discourse
    queries = [
        f'"{artist_name}" review album music',
        f'"{artist_name}" music criticism "music critic" OR "music journalism"',
        f'"{artist_name}" feature "music review" OR "music journalism"',
        f'"{artist_name}" review site:pitchfork.com OR site:thequietus.com OR site:stereogum.com',
    ]

    all_results = []

    for query in queries:
        print(f"  Querying: {query}")
        try:
            results = provider.search(query, num_results=10)
        except Exception as e:
            print(f"    Search error: {e}")
            continue

        for r in results:
            url = r.get("url", "")

            # Filter
            if not url:
                continue
            if is_video_audio_url(url):
                print(f"    SKIP (video/audio): {url}")
                continue
            if is_blocked_domain(url, blocked_domains):
                print(f"    SKIP (blocked domain): {url}")
                continue

            all_results.append(r)

        time.sleep(0.5)  # Rate limiting

    # Score and sort
    for r in all_results:
        r["score"] = score_review_result(r)

    all_results.sort(key=lambda x: x["score"], reverse=True)

    # Deduplicate and limit
    deduped = deduplicate_by_domain(all_results, max_per_domain=2)
    final = deduped[:num_results]

    return final


def search_reviews(artist_name: str, provider_name: str = "brave"):
    """Main search function for reviews."""

    print(f"Searching for reviews: {artist_name}")

    # Normalize artist ID
    artist_id = normalize_artist_id(artist_name)
    print(f"Artist ID: {artist_id}")

    # Create artist directory
    artist_dir = Path(f"data/artists/{artist_id}")
    artist_dir.mkdir(parents=True, exist_ok=True)

    # Load existing review sources if present
    review_sources_path = artist_dir / "review_sources.json"
    if review_sources_path.exists():
        review_sources = json.loads(review_sources_path.read_text())
        existing_urls = {s["url"] for s in review_sources}
        print(f"Found {len(existing_urls)} existing review source(s)")
    else:
        review_sources = []
        existing_urls = set()

    # Get provider
    try:
        provider = get_provider(provider_name)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Load blocked domains
    blocked_domains = load_blocked_domains()

    # Search
    results = search_for_reviews(artist_name, provider, blocked_domains, num_results=8)

    if not results:
        print("No results found.")
        return

    # Add to review sources (skip duplicates)
    added = 0
    for result in results:
        url = result["url"]
        if url in existing_urls:
            print(f"Already in review sources: {url}")
            continue

        entry = {
            "url": url,
            "publication": urlparse(url).netloc.replace("www.", "").replace(".com", "").title(),
            "date": "",
            "title": result.get("title", ""),
            "text": None,
        }
        review_sources.append(entry)
        added += 1
        print(f"Added: {url}")

    # Save
    review_sources_path.write_text(json.dumps(review_sources, indent=2, ensure_ascii=False))
    print(f"\nSaved {added} new review source(s) to {review_sources_path}")
    print(f"Total review sources for {artist_id}: {len(review_sources)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/search_reviews.py <artist_name> [--provider brave|serpapi|google]")
        sys.exit(1)

    artist_name = sys.argv[1]
    provider = "brave"

    for i, arg in enumerate(sys.argv[2:]):
        if arg == "--provider" and i + 3 < len(sys.argv):
            provider = sys.argv[i + 3]

    search_reviews(artist_name, provider)
