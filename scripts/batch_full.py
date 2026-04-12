"""Batch processing orchestrator for multiple artists with dual signals.

Processes artists sequentially, running both interview and critic pipelines.
Runs global dual-signal pipeline (embed, compute, discover, compare) when complete.
"""

import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime

from search import normalize_artist_id
import importlib.util

CHECKPOINT_PATH = Path("data/batch_checkpoint.json")


def load_checkpoint() -> set:
    """Return set of artist_ids already processed in the current session."""
    if not CHECKPOINT_PATH.exists():
        return set()
    try:
        data = json.loads(CHECKPOINT_PATH.read_text())
        return set(data.get("processed", []))
    except Exception:
        return set()


def checkpoint_add(artist_id: str) -> None:
    """Record an artist as processed (append-safe, survives concurrent reads)."""
    processed = load_checkpoint()
    processed.add(artist_id)
    CHECKPOINT_PATH.write_text(json.dumps({
        "updated": datetime.now().isoformat(),
        "processed": sorted(processed),
    }, indent=2))


def checkpoint_clear() -> None:
    """Delete checkpoint once a full run completes."""
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()


def load_module(script_name: str):
    """Dynamically load a module from scripts/."""
    spec = importlib.util.spec_from_file_location(script_name, f"scripts/{script_name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def run_ingest_full(artist_name: str, search_provider: str = "brave") -> bool:
    """Run the full dual pipeline for one artist."""
    try:
        ingest_full_module = load_module("ingest_full")
        success = ingest_full_module.ingest_full(artist_name, search_provider,
                                                 skip_search=False, skip_scrape=False,
                                                 skip_extract=False)
        return success
    except SystemExit:
        return False
    except Exception as e:
        print(f"Exception during ingest_full: {e}")
        return False


def run_global_dual_pipeline():
    """Run the full global pipeline for both signals.

    Sequence:
    1. embed.py (quotes)
    2. compute.py (quotes)
    3. discover.py (quotes)
    4. embed_critics.py (critic)
    5. compute_critics.py (critic)
    6. discover_critics.py (critic)
    7. compare_signals.py (comparison)
    """
    print(f"\n{'='*70}")
    print("[GLOBAL DUAL-SIGNAL PIPELINE]")
    print(f"{'='*70}")

    scripts = [
        "embed.py",
        "compute.py",
        "discover.py",
        "embed_critics.py",
        "compute_critics.py",
        "discover_critics.py",
        "compare_signals.py",
    ]

    for script in scripts:
        print(f"\nRunning {script}...")
        try:
            result = subprocess.run([sys.executable, f"scripts/{script}"],
                                  capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                print(f"Error in {script}:")
                print(result.stderr)
            else:
                print(result.stdout)
        except subprocess.TimeoutExpired:
            print(f"Timeout: {script} took too long")
        except Exception as e:
            print(f"Error running {script}: {e}")


def batch_full(artists: list, search_provider: str = "brave",
               delay: float = 2.0, force: bool = False,
               resume: bool = False) -> dict:
    """Process multiple artists with dual pipelines."""

    already_done = load_checkpoint() if resume else set()
    if resume and already_done:
        print(f"Resuming: {len(already_done)} artist(s) already processed this session, skipping them.")

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

        # Resume: skip artists already touched this session
        if resume and artist_id in already_done:
            print(f"Already processed this session, skipping.")
            results["skipped"].append(artist_id)
            continue

        # Check if already has any valid corpus
        has_valid_quote = is_corpus_valid(artist_id, "quotes")
        has_valid_critic = is_corpus_valid(artist_id, "critic")

        if not force and (has_valid_quote or has_valid_critic):
            print(f"Already has valid corpus (quotes: {has_valid_quote}, critic: {has_valid_critic}), skipping.")
            results["skipped"].append(artist_id)
            checkpoint_add(artist_id)
            continue

        # Ingest
        try:
            success = run_ingest_full(artist_name, search_provider)
            if success:
                results["succeeded"].append(artist_id)
            else:
                results["failed"].append(artist_id)
        except Exception as e:
            print(f"Exception during batch ingest: {e}")
            results["failed"].append(artist_id)

        # Mark as processed so a restart won't repeat this artist
        checkpoint_add(artist_id)

        # Rate limit (except after last)
        if i < len(artists):
            print(f"Waiting {delay}s before next artist...")
            time.sleep(delay)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch ingestion with dual signals (interviews + reviews)")
    parser.add_argument("--file", help="Path to file with artist names (one per line)")
    parser.add_argument("--artists", help="Comma-separated artist names")
    parser.add_argument("--search-provider", choices=["brave", "serpapi", "google"],
                       default="brave", help="Search provider (default: brave)")
    parser.add_argument("--delay", type=float, default=2.0,
                       help="Delay between artists in seconds (default: 2)")
    parser.add_argument("--force", action="store_true",
                       help="Re-ingest even if artist has valid corpus")
    parser.add_argument("--resume", action="store_true",
                       help="Skip artists already processed in the last interrupted run")
    parser.add_argument("--no-pipeline", action="store_true",
                       help="Don't run global pipeline after batch")

    args = parser.parse_args()

    # Load artists
    artists = load_artist_list(args.file, args.artists)

    print(f"Starting batch dual-signal ingestion of {len(artists)} artist(s)")
    print(f"Search provider: {args.search_provider}")
    print(f"Delay between artists: {args.delay}s")
    print(f"Force re-ingest: {args.force}")
    print(f"Resume from checkpoint: {args.resume}")
    print(f"Run global pipeline after: {not args.no_pipeline}")

    # Run ingestion
    results = batch_full(artists, args.search_provider, args.delay, args.force, args.resume)

    # Log results
    log_path = Path("data/batch_full_log.json")
    existing_logs = []
    if log_path.exists():
        try:
            existing_logs = json.loads(log_path.read_text())
            if not isinstance(existing_logs, list):
                existing_logs = [existing_logs]
        except:
            existing_logs = []

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
    if not args.no_pipeline and results["succeeded"]:
        print(f"\nRunning global dual-signal pipeline...")
        run_global_dual_pipeline()
    else:
        print("\nSkipping global pipeline (no new artists added).")

    # Clear checkpoint — full run finished, next run starts fresh
    checkpoint_clear()
    print("\nCheckpoint cleared.")
