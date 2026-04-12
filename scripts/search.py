"""Automated interview URL discovery for artists using configurable search backends.

Supports Brave Search (default), SerpAPI, and Google Custom Search.
"""

import sys
import json
import re
import os
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional
from urllib.parse import urlparse
import time

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

# Artist ID normalization
def normalize_artist_id(name: str) -> str:
    """Normalize artist name to ID: lowercase, spaces to underscores, strip punctuation."""
    id_str = name.lower().strip()
    id_str = re.sub(r'[^\w\s\-]', '', id_str)
    id_str = re.sub(r'\s+', '_', id_str)
    id_str = re.sub(r'_+', '_', id_str)
    return id_str.strip('_')


# Blocked domains
def load_blocked_domains() -> set:
    """Load blocked domains from data/blocked_domains.md."""
    path = Path("data/blocked_domains.md")
    blocked = set()
    if path.exists():
        content = path.read_text()
        for line in content.split('\n'):
            if '`' in line:
                match = re.search(r'`([^`]+)`', line)
                if match:
                    domain = match.group(1)
                    blocked.add(domain.lower())
    return blocked


# Video/audio domain patterns
VIDEO_AUDIO_DOMAINS = {
    'youtube.com', 'youtu.be', 'vimeo.com', 'soundcloud.com',
    'spotify.com', 'bandcamp.com', 'tiktok.com', 'instagram.com',
    'twitch.tv', 'dailymotion.com', 'apple.com/music', 'music.amazon.com'
}


class SearchProvider(ABC):
    """Base class for search providers."""

    @abstractmethod
    def search(self, query: str, num_results: int = 10) -> list:
        """Search and return list of {title, url, snippet} dicts."""
        pass


class BraveSearchProvider(SearchProvider):
    """Brave Search API provider."""

    def __init__(self):
        self.api_key = os.getenv('BRAVE_API_KEY')
        if not self.api_key:
            raise ValueError("BRAVE_API_KEY environment variable not set")
        self.base_url = "https://api.search.brave.com/res/v1/web/search"

    def search(self, query: str, num_results: int = 10) -> list:
        """Query Brave Search API."""
        headers = {"Accept": "application/json", "X-Subscription-Token": self.api_key}
        params = {"q": query, "count": num_results}

        try:
            resp = requests.get(self.base_url, headers=headers, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            results = []
            for item in data.get("web", {}).get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("description", "")
                })
            return results
        except requests.RequestException as e:
            print(f"Brave Search error: {e}")
            return []


class SerpAPIProvider(SearchProvider):
    """SerpAPI provider."""

    def __init__(self):
        self.api_key = os.getenv('SERPAPI_KEY')
        if not self.api_key:
            raise ValueError("SERPAPI_KEY environment variable not set")
        self.base_url = "https://serpapi.com/search"

    def search(self, query: str, num_results: int = 10) -> list:
        """Query SerpAPI."""
        params = {
            "q": query,
            "api_key": self.api_key,
            "num": num_results,
            "engine": "google"
        }

        try:
            resp = requests.get(self.base_url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            results = []
            for item in data.get("organic_results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", "")
                })
            return results
        except requests.RequestException as e:
            print(f"SerpAPI error: {e}")
            return []


class GoogleCSEProvider(SearchProvider):
    """Google Custom Search Engine provider."""

    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY')
        self.cse_id = os.getenv('GOOGLE_CSE_ID')
        if not self.api_key or not self.cse_id:
            raise ValueError("GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables required")
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def search(self, query: str, num_results: int = 10) -> list:
        """Query Google Custom Search."""
        params = {
            "q": query,
            "key": self.api_key,
            "cx": self.cse_id,
            "num": min(num_results, 10)
        }

        try:
            resp = requests.get(self.base_url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            results = []
            for item in data.get("items", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", "")
                })
            return results
        except requests.RequestException as e:
            print(f"Google CSE error: {e}")
            return []


def get_provider(provider_name: str = "brave") -> SearchProvider:
    """Factory function to get search provider by name."""
    providers = {
        "brave": BraveSearchProvider,
        "serpapi": SerpAPIProvider,
        "google": GoogleCSEProvider,
    }

    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}. Must be one of: {list(providers.keys())}")

    return providers[provider_name]()


def is_video_audio_url(url: str) -> bool:
    """Check if URL is a video or audio platform."""
    domain = urlparse(url).netloc.lower()
    for blocked_domain in VIDEO_AUDIO_DOMAINS:
        if blocked_domain in domain:
            return True
    return False


def is_blocked_domain(url: str, blocked_domains: set) -> bool:
    """Check if URL is in blocked domains list."""
    domain = urlparse(url).netloc.lower()
    for blocked in blocked_domains:
        if blocked in domain:
            return True
    return False


def score_result(result: dict) -> float:
    """Score result by likely interview quality (length of snippet, presence of keywords)."""
    snippet = result.get("snippet", "").lower()
    title = result.get("title", "").lower()

    score = 0.0

    # Longer snippets suggest more substantial content
    score += len(snippet) / 100.0

    # Interview keywords
    keywords = ["interview", "talks", "conversation", "discusses", "creative process",
                "how i make", "making music", "behind the scenes", "in conversation"]
    for kw in keywords:
        if kw in snippet or kw in title:
            score += 2.0

    # Avoid fluff
    avoid = ["top 10", "best of", "ranking", "review", "best moments"]
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


def search_for_artist(artist_name: str, provider: SearchProvider,
                     blocked_domains: set, num_results: int = 6) -> list:
    """Search for interview URLs for an artist."""

    # Two main queries
    queries = [
        f'"{artist_name}" interview "creative process" OR "making music" OR "how i make"',
        f'"{artist_name}" interview musician "in their own words"',
    ]

    all_results = []

    for query in queries:
        print(f"  Querying: {query}")
        results = provider.search(query, num_results=10)

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
        r["score"] = score_result(r)

    all_results.sort(key=lambda x: x["score"], reverse=True)

    # Deduplicate and limit
    deduped = deduplicate_by_domain(all_results, max_per_domain=2)
    final = deduped[:num_results]

    return final


def search(artist_name: str, provider_name: str = "brave"):
    """Main search function."""

    print(f"Searching for interviews: {artist_name}")

    # Normalize artist ID
    artist_id = normalize_artist_id(artist_name)
    print(f"Artist ID: {artist_id}")

    # Create artist directory
    artist_dir = Path(f"data/artists/{artist_id}")
    artist_dir.mkdir(parents=True, exist_ok=True)

    # Load existing sources if present
    sources_path = artist_dir / "sources.json"
    if sources_path.exists():
        sources = json.loads(sources_path.read_text())
        existing_urls = {s["url"] for s in sources}
        print(f"Found {len(existing_urls)} existing source(s)")
    else:
        sources = []
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
    results = search_for_artist(artist_name, provider, blocked_domains, num_results=6)

    if not results:
        print("No results found.")
        return

    # Add to sources (skip duplicates)
    added = 0
    for result in results:
        url = result["url"]
        if url in existing_urls:
            print(f"Already in sources: {url}")
            continue

        entry = {
            "url": url,
            "publication": urlparse(url).netloc.replace("www.", "").replace(".com", "").title(),
            "date": "",
            "title": result.get("title", ""),
            "text": None,
        }
        sources.append(entry)
        added += 1
        print(f"Added: {url}")

    # Save
    sources_path.write_text(json.dumps(sources, indent=2, ensure_ascii=False))
    print(f"\nSaved {added} new source(s) to {sources_path}")
    print(f"Total sources for {artist_id}: {len(sources)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/search.py <artist_name> [--provider brave|serpapi|google]")
        sys.exit(1)

    artist_name = sys.argv[1]
    provider = "brave"

    for i, arg in enumerate(sys.argv[2:]):
        if arg == "--provider" and i + 3 < len(sys.argv):
            provider = sys.argv[i + 3]

    search(artist_name, provider)
