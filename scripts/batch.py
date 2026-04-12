"""Batch processing orchestrator for multiple artists.

Processes artists sequentially, respecting API rate limits.
Runs global pipeline (embed → compute → discover) when complete.
"""

import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime

from ingest import ingest, is_corpus_valid, normalize_artist_id


def load_artist_list(file_path: str = None, artists_str: str = None) -> list:
    """Load artist names from file or comma-separated string."""

    names = []

    if file_path:
        path = Path(file_path)
        if not path.exists():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
        content = path.read_text().strip()
        names = [line.strip() for line in content.split('\n') if line.strip()]
    elif artists_str:
        names = [name.strip() for name in artists_str.split(',')]
    else:
        print("Error: Provide --file or --artists")
        sys.exit(1)

    print(f"Loaded {len(names)} artist(s)")
    return names


def run_global_pipeline():
    """Run embed → compute → discover globally."""
    print("\n[GLOBAL PIPELINE]")
    scripts = ["embed.py", "compute.py", "discover.py"]

    for script in scripts:
        print(f"\nRunning {script}...")
        try:
            result = subprocess.run([sys.executable, f"scripts/{script}"],
                                  capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                print(f"Error in {script}:")
                print(result.stderr)
            else:
                print(result.stdout)
        except subprocess.TimeoutExpired:
            print(f"Timeout: {script} took too long")
        except Exception as e:
            print(f"Error running {script}: {e}")


def batch_ingest(artists: list, search_provider: str = "brave",
                delay: float = 2.0, force: bool = False) -> dict:
    """Process multiple artists with delay between them."""

    results = {
        "timestamp": datetime.now().isoformat(),
        "search_provider": search_provider,
        "artists_requested": len(artists),
        "succeeded": [],
        "failed": [],
        "skipped": [],
    }

    for i, artist_name in enumerate(artists, 1):
        artist_id = normalize_artist_id(artist_name)

        print(f"\n{'='*70}")
        print(f"[{i}/{len(artists)}] {artist_name}")
        print(f"{'='*70}")

        # Check if already valid
        if not force and is_corpus_valid(artist_id):
            print(f"Already has valid corpus, skipping.")
            results["skipped"].append(artist_id)
            continue

        # Ingest
        try:
            success = ingest(artist_name, search_provider, skip_search=False,
                           skip_scrape=False, skip_extract=False)
            if success:
                results["succeeded"].append(artist_id)
            else:
                results["failed"].append(artist_id)
        except Exception as e:
            print(f"Exception during ingest: {e}")
            results["failed"].append(artist_id)

        # Rate limit (except after last)
        if i < len(artists):
            print(f"Waiting {delay}s before next artist...")
            time.sleep(delay)

    return results


def batch(file_path: str = None, artists_str: str = None,
         search_provider: str = "brave", delay: float = 2.0,
         force: bool = False, run_pipeline: bool = True):
    """Main batch processing function."""

    artists = load_artist_list(file_path, artists_str)

    print(f"Starting batch ingestion of {len(artists)} artist(s)")
    print(f"Search provider: {search_provider}")
    print(f"Delay between artists: {delay}s")
    print(f"Force re-ingest: {force}")
    print(f"Run global pipeline after: {run_pipeline}")

    # Run ingestion
    results = batch_ingest(artists, search_provider, delay, force)

    # Log results
    log_path = Path("data/batch_log.json")
    existing_logs = []
    if log_path.exists():
        existing_logs = json.loads(log_path.read_text())
        if not isinstance(existing_logs, list):
            existing_logs = [existing_logs]

    existing_logs.append(results)
    log_path.write_text(json.dumps(existing_logs, indent=2, ensure_ascii=False))

    # Summary
    print(f"\n{'='*70}")
    print("[SUMMARY]")
    print(f"{'='*70}")
    print(f"Succeeded: {len(results['succeeded'])}")
    for aid in results["succeeded"]:
        print(f"  + {aid}")
    print(f"\nFailed: {len(results['failed'])}")
    for aid in results["failed"]:
        print(f"  - {aid}")
    print(f"\nSkipped: {len(results['skipped'])}")
    for aid in results["skipped"]:
        print(f"  ~ {aid}")

    print(f"\nLog saved to: {log_path}")

    # Run global pipeline
    if run_pipeline and results["succeeded"]:
        print(f"\nRunning global pipeline...")
        run_global_pipeline()
    else:
        print("\nSkipping global pipeline (no new artists added).")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch ingestion for multiple artists")
    parser.add_argument("--file", help="Path to file with artist names (one per line)")
    parser.add_argument("--artists", help="Comma-separated artist names")
    parser.add_argument("--search-provider", choices=["brave", "serpapi", "google"],
                       default="brave", help="Search provider (default: brave)")
    parser.add_argument("--delay", type=float, default=2.0,
                       help="Delay between artists in seconds (default: 2)")
    parser.add_argument("--force", action="store_true",
                       help="Re-ingest even if artist has valid corpus")
    parser.add_argument("--no-pipeline", action="store_true",
                       help="Don't run global pipeline after batch")

    args = parser.parse_args()

    batch(args.file, args.artists, args.search_provider, args.delay,
         args.force, not args.no_pipeline)
