"""Single-artist full pipeline orchestration.

Runs: search → scrape → extract → validate
"""

import sys
import json
import re
import subprocess
from pathlib import Path

# Import the necessary functions from other scripts
from search import search, normalize_artist_id
import importlib.util


def load_module(script_name: str):
    """Dynamically load a module from scripts/."""
    spec = importlib.util.spec_from_file_location(script_name, f"scripts/{script_name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_scrape(artist_id: str) -> bool:
    """Run scrape.py for artist."""
    print(f"\n[SCRAPE] Processing {artist_id}...")
    try:
        scrape_module = load_module("scrape")
        scrape_module.scrape(artist_id)
        return True
    except Exception as e:
        print(f"Scrape failed: {e}")
        return False


def run_extract(artist_id: str, model: str = "claude-sonnet-4-20250514") -> bool:
    """Run extract_llm.py for artist."""
    print(f"\n[EXTRACT] Processing {artist_id}...")
    try:
        extract_module = load_module("extract_llm")
        extract_module.extract(artist_id, model)
        return True
    except SystemExit:
        # extract_llm calls sys.exit on error, catch it
        return False
    except Exception as e:
        print(f"Extract failed: {e}")
        return False


def is_corpus_valid(artist_id: str) -> bool:
    """Check if artist has a valid corpus."""
    quotes_path = Path(f"data/artists/{artist_id}/quotes.json")
    if not quotes_path.exists():
        return False

    try:
        data = json.loads(quotes_path.read_text())
        return data.get("corpus_meta", {}).get("corpus_valid", False)
    except:
        return False


def ingest(artist_name: str, search_provider: str = "brave", skip_search: bool = False,
           skip_scrape: bool = False, skip_extract: bool = False):
    """Run full ingestion pipeline for a single artist."""

    artist_id = normalize_artist_id(artist_name)
    print(f"Ingesting: {artist_name}")
    print(f"Artist ID: {artist_id}\n")

    artist_dir = Path(f"data/artists/{artist_id}")

    # Step 1: Search
    if not skip_search:
        print("[SEARCH] Finding interview URLs...")
        try:
            search(artist_name, search_provider)
        except Exception as e:
            print(f"Search failed: {e}")
            return False
    else:
        sources_path = artist_dir / "sources.json"
        if not sources_path.exists():
            print("Error: sources.json not found and --skip-search was used")
            return False
        sources = json.loads(sources_path.read_text())
        print(f"Using existing sources.json ({len(sources)} source(s))")

    # Step 2: Scrape
    if not skip_scrape:
        if not run_scrape(artist_id):
            return False
    else:
        sources_path = artist_dir / "sources.json"
        if sources_path.exists():
            sources = json.loads(sources_path.read_text())
            sources_with_text = [s for s in sources if s.get("text")]
            print(f"Using existing sources with text ({len(sources_with_text)} source(s))")
        else:
            print("Error: sources.json not found")
            return False

    # Step 3: Extract
    if not skip_extract:
        if not run_extract(artist_id):
            return False
    else:
        quotes_path = artist_dir / "quotes.json"
        if quotes_path.exists():
            data = json.loads(quotes_path.read_text())
            quote_count = data.get("corpus_meta", {}).get("quote_count", 0)
            print(f"Using existing quotes.json ({quote_count} quote(s))")
        else:
            print("Error: quotes.json not found")
            return False

    # Step 4: Report
    print("\n[REPORT]")
    if is_corpus_valid(artist_id):
        print(f"SUCCESS: {artist_id} has a VALID corpus")
        return True
    else:
        quotes_path = artist_dir / "quotes.json"
        if quotes_path.exists():
            data = json.loads(quotes_path.read_text())
            meta = data.get("corpus_meta", {})
            quote_count = meta.get("quote_count", 0)
            source_count = meta.get("source_count", 0)
            date_range = meta.get("date_range", [])
            year_count = len(set(date_range))
            print(f"INVALID corpus: {artist_id}")
            print(f"  {quote_count}/5 quotes, {source_count}/3 sources, {year_count}/2 years")
        else:
            print(f"FAILED: {artist_id} — no quotes.json")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Single-artist full ingestion pipeline")
    parser.add_argument("artist", help="Artist name")
    parser.add_argument("--search-provider", choices=["brave", "serpapi", "google"],
                       default="brave", help="Search provider (default: brave)")
    parser.add_argument("--skip-search", action="store_true", help="Skip search step")
    parser.add_argument("--skip-scrape", action="store_true", help="Skip scrape step")
    parser.add_argument("--skip-extract", action="store_true", help="Skip extract step")

    args = parser.parse_args()

    success = ingest(args.artist, args.search_provider, args.skip_search,
                     args.skip_scrape, args.skip_extract)

    sys.exit(0 if success else 1)
