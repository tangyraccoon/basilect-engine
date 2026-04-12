"""Scrape review text from URLs.

Identical to scrape.py but reads from review_sources.json and writes back to the same.
"""

import sys
import json
import signal
from pathlib import Path
import trafilatura


def _timeout_handler(signum, frame):
    raise TimeoutError("fetch timed out")


def scrape_reviews(artist_id):
    """Scrape text from review URLs."""
    review_sources_path = Path(f"data/artists/{artist_id}/review_sources.json")

    if not review_sources_path.exists():
        print(f"Not found: {review_sources_path}")
        sys.exit(1)

    review_sources = json.loads(review_sources_path.read_text())

    for entry in review_sources:
        if entry.get("text"):
            continue

        url = entry["url"]
        pub = entry.get("publication", url)

        try:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(15)
            downloaded = trafilatura.fetch_url(url)
            signal.alarm(0)
        except TimeoutError:
            entry["text"] = None
            print(f"FAIL {pub} — fetch timed out — {url}")
            continue

        if not downloaded:
            entry["text"] = None
            print(f"FAIL {pub} — fetch failed (network/blocked) — {url}")
            continue
        text = trafilatura.extract(downloaded)
        if not text:
            entry["text"] = None
            print(f"FAIL {pub} — no text extracted (paywall/JS) — {url}")
            continue
        entry["text"] = text
        print(f"OK  {pub}")

    review_sources_path.write_text(json.dumps(review_sources, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/scrape_reviews.py <artist_id>")
        sys.exit(1)
    scrape_reviews(sys.argv[1])
