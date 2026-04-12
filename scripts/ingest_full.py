"""Full pipeline for one artist (both interview quotes and critic discourse).

Orchestrates:
1. Search interviews → scrape → extract quotes (existing pipeline)
2. Search reviews → scrape reviews → extract critic passages
3. Report both corpus validities
"""

import sys
import json
import re
import subprocess
from pathlib import Path

# Import the necessary functions from other scripts
from search import normalize_artist_id
import importlib.util


def load_module(script_name: str):
    """Dynamically load a module from scripts/."""
    spec = importlib.util.spec_from_file_location(script_name, f"scripts/{script_name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_scrape(artist_id: str) -> bool:
    """Run scrape.py for artist."""
    print(f"\n[SCRAPE INTERVIEWS] Processing {artist_id}...")
    try:
        scrape_module = load_module("scrape")
        scrape_module.scrape(artist_id)
        return True
    except Exception as e:
        print(f"Scrape interviews failed: {e}")
        return False


def run_extract(artist_id: str, model: str = "claude-haiku-4-5-20251001") -> bool:
    """Run extract_llm.py for artist."""
    print(f"\n[EXTRACT QUOTES] Processing {artist_id}...")
    try:
        extract_module = load_module("extract_llm")
        extract_module.extract(artist_id, model)
        return True
    except SystemExit:
        return False
    except Exception as e:
        print(f"Extract quotes failed: {e}")
        return False


def run_scrape_reviews(artist_id: str) -> bool:
    """Run scrape_reviews.py for artist."""
    print(f"\n[SCRAPE REVIEWS] Processing {artist_id}...")
    try:
        scrape_reviews_module = load_module("scrape_reviews")
        scrape_reviews_module.scrape_reviews(artist_id)
        return True
    except Exception as e:
        print(f"Scrape reviews failed: {e}")
        return False


def run_extract_critic(artist_id: str, model: str = "claude-haiku-4-5-20251001") -> bool:
    """Run extract_critic.py for artist."""
    print(f"\n[EXTRACT CRITIC DISCOURSE] Processing {artist_id}...")
    try:
        extract_critic_module = load_module("extract_critic")
        extract_critic_module.extract_critic(artist_id, model)
        return True
    except SystemExit:
        return False
    except Exception as e:
        print(f"Extract critic failed: {e}")
        return False


def is_corpus_valid(artist_id: str, corpus_type: str = "quotes") -> bool:
    """Check if artist has a valid corpus."""
    if corpus_type == "quotes":
        path = Path(f"data/artists/{artist_id}/quotes.json")
    elif corpus_type == "critic":
        path = Path(f"data/artists/{artist_id}/critic_quotes.json")
    else:
        return False

    if not path.exists():
        return False

    try:
        data = json.loads(path.read_text())
        return data.get("corpus_meta", {}).get("corpus_valid", False)
    except:
        return False


def ingest_full(artist_name: str, search_provider: str = "brave", skip_search: bool = False,
                skip_scrape: bool = False, skip_extract: bool = False):
    """Run full dual ingestion pipeline (interviews + reviews) for a single artist."""

    artist_id = normalize_artist_id(artist_name)
    print(f"\n{'='*70}")
    print(f"Ingesting (FULL DUAL PIPELINE): {artist_name}")
    print(f"Artist ID: {artist_id}")
    print(f"{'='*70}")

    artist_dir = Path(f"data/artists/{artist_id}")

    # ============ INTERVIEW PIPELINE ============
    print(f"\n{'─'*70}")
    print("PHASE 1: INTERVIEW QUOTES")
    print(f"{'─'*70}")

    if not skip_search:
        print("[SEARCH] Finding interview URLs...")
        try:
            search_module = load_module("search")
            search_module.search(artist_name, search_provider)
        except Exception as e:
            print(f"Search interviews failed: {e}")
            return False
    else:
        sources_path = artist_dir / "sources.json"
        if not sources_path.exists():
            print("Error: sources.json not found and --skip-search was used")
            return False
        sources = json.loads(sources_path.read_text())
        print(f"Using existing sources.json ({len(sources)} source(s))")

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

    # ============ REVIEW/CRITIC PIPELINE ============
    print(f"\n{'─'*70}")
    print("PHASE 2: CRITIC DISCOURSE")
    print(f"{'─'*70}")

    # Search reviews
    print("[SEARCH REVIEWS] Finding review URLs...")
    try:
        search_reviews_module = load_module("search_reviews")
        search_reviews_module.search_reviews(artist_name, search_provider)
    except Exception as e:
        print(f"Search reviews failed: {e}")
        return False

    # Scrape reviews
    if not run_scrape_reviews(artist_id):
        return False

    # Extract critic discourse
    if not run_extract_critic(artist_id):
        return False

    # ============ REPORT ============
    print(f"\n{'='*70}")
    print("[REPORT]")
    print(f"{'='*70}")

    quote_valid = is_corpus_valid(artist_id, "quotes")
    critic_valid = is_corpus_valid(artist_id, "critic")

    print(f"\nInterview Quotes: {'VALID' if quote_valid else 'INVALID'}")
    if not quote_valid:
        quotes_path = artist_dir / "quotes.json"
        if quotes_path.exists():
            data = json.loads(quotes_path.read_text())
            meta = data.get("corpus_meta", {})
            print(f"  {meta.get('quote_count', 0)}/5 quotes, "
                  f"{meta.get('source_count', 0)}/3 sources, "
                  f"{len(set(meta.get('date_range', [])))}/2 years")

    print(f"Critic Discourse: {'VALID' if critic_valid else 'INVALID'}")
    if not critic_valid:
        critic_path = artist_dir / "critic_quotes.json"
        if critic_path.exists():
            data = json.loads(critic_path.read_text())
            meta = data.get("corpus_meta", {})
            print(f"  {meta.get('quote_count', 0)}/5 passages, "
                  f"{meta.get('source_count', 0)}/2 sources, "
                  f"{len(set(meta.get('date_range', [])))}/1 years")

    success = quote_valid or critic_valid
    if success:
        print(f"\nSUCCESS: {artist_id} has at least one valid corpus")
    else:
        print(f"\nFAILED: {artist_id} has no valid corpora")

    return success


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dual-signal (interviews + reviews) full ingestion pipeline")
    parser.add_argument("artist", help="Artist name")
    parser.add_argument("--search-provider", choices=["brave", "serpapi", "google"],
                       default="brave", help="Search provider (default: brave)")
    parser.add_argument("--skip-search", action="store_true", help="Skip search step")
    parser.add_argument("--skip-scrape", action="store_true", help="Skip scrape step")
    parser.add_argument("--skip-extract", action="store_true", help="Skip extract step")

    args = parser.parse_args()

    success = ingest_full(args.artist, args.search_provider, args.skip_search,
                          args.skip_scrape, args.skip_extract)

    sys.exit(0 if success else 1)
