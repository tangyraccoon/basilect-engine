import sys
import json
import signal
from pathlib import Path
import trafilatura

def _timeout_handler(signum, frame):
    raise TimeoutError("fetch timed out")

def scrape(artist_id):
    path = Path(f"data/artists/{artist_id}/sources.json")
    if not path.exists():
        print(f"Not found: {path}")
        sys.exit(1)

    sources = json.loads(path.read_text())

    for entry in sources:
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

    path.write_text(json.dumps(sources, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/scrape.py <artist_id>")
        sys.exit(1)
    scrape(sys.argv[1])
