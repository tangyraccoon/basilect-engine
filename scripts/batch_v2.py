"""Batch processing orchestrator v2 — state-machine-driven, crash-resilient.

Replaces batch.py and batch_full.py. Key improvements:
- Per-artist state.json tracks each step (search/scrape/extract × interview/review)
- On restart, resumes from first incomplete step per artist
- Never re-runs completed steps
- Supports --phase interview|critic|both, --max-credits, --dry-run
- Writes live batch_progress.json for monitoring
- Sorts work queue: partial artists first, then new artists
"""

import sys
import os
import json
import time
import importlib.util
from pathlib import Path
from datetime import datetime, timezone

# Add scripts dir to path for imports
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from state import (
    load_state, save_state, mark_step_running, mark_step_done,
    mark_step_failed, mark_step_skipped, is_step_done,
    next_incomplete_step, is_artist_complete, scan_all_artist_states,
    load_batch_state, save_batch_state, STEPS,
)
from prefilter import prefilter


# ── Cost estimation ──────────────────────────────────────────────────────────

# Haiku 4.5 pricing (per million tokens)
HAIKU_INPUT_PER_M = 0.80
HAIKU_OUTPUT_PER_M = 4.00
# After truncation to 6K chars: ~7,500 input tokens + ~600 output tokens per call
AVG_INPUT_TOKENS = 7500
AVG_OUTPUT_TOKENS = 600

COST_PER_EXTRACT_CALL = (
    AVG_INPUT_TOKENS / 1_000_000 * HAIKU_INPUT_PER_M +
    AVG_OUTPUT_TOKENS / 1_000_000 * HAIKU_OUTPUT_PER_M
)

# 4 interview + 4 review sources (capped) = 8 extraction calls per artist
EST_CALLS_PER_ARTIST = 8
EST_COST_PER_ARTIST = EST_CALLS_PER_ARTIST * COST_PER_EXTRACT_CALL


# ── Module loading ───────────────────────────────────────────────────────────

_module_cache = {}

def load_module(script_name: str):
    """Dynamically load a module from scripts/."""
    if script_name in _module_cache:
        return _module_cache[script_name]
    spec = importlib.util.spec_from_file_location(
        script_name, str(SCRIPTS_DIR / f"{script_name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _module_cache[script_name] = module
    return module


def normalize_artist_id(name: str) -> str:
    import re
    id_str = name.lower().strip()
    id_str = re.sub(r'[^\w\s\-]', '', id_str)
    id_str = re.sub(r'\s+', '_', id_str)
    id_str = re.sub(r'_+', '_', id_str)
    return id_str.strip('_')


# ── Per-step execution ───────────────────────────────────────────────────────

def run_interview_search(artist_name: str, artist_id: str, state: dict,
                         search_provider: str) -> dict:
    """Step: interview_search"""
    if is_step_done(state, "interview_search"):
        return state

    mark_step_running(state, "interview_search")
    try:
        search_module = load_module("search")
        search_module.search(artist_name, search_provider)

        sources_path = Path(f"data/artists/{artist_id}/sources.json")
        count = 0
        if sources_path.exists():
            count = len(json.loads(sources_path.read_text()))

        return mark_step_done(state, "interview_search", f"{count} urls")
    except (Exception, SystemExit) as e:
        return mark_step_failed(state, "interview_search", str(e))


def run_interview_scrape(artist_id: str, state: dict) -> dict:
    """Step: interview_scrape"""
    if is_step_done(state, "interview_scrape"):
        return state

    mark_step_running(state, "interview_scrape")
    try:
        scrape_module = load_module("scrape")
        scrape_module.scrape(artist_id)

        sources_path = Path(f"data/artists/{artist_id}/sources.json")
        scraped = 0
        total = 0
        if sources_path.exists():
            sources = json.loads(sources_path.read_text())
            total = len(sources)
            scraped = sum(1 for s in sources if s.get("text"))

        return mark_step_done(state, "interview_scrape", f"{scraped}/{total} scraped")
    except (Exception, SystemExit) as e:
        return mark_step_failed(state, "interview_scrape", str(e))


def run_interview_extract(artist_id: str, artist_name: str, state: dict,
                          model: str) -> dict:
    """Step: interview_extract — with prefiltering"""
    if is_step_done(state, "interview_extract"):
        return state

    mark_step_running(state, "interview_extract")
    try:
        extract_module = load_module("extract_llm")
        extract_module.extract(artist_id, model)

        quotes_path = Path(f"data/artists/{artist_id}/quotes.json")
        count = 0
        if quotes_path.exists():
            data = json.loads(quotes_path.read_text())
            count = data.get("corpus_meta", {}).get("quote_count", 0)

        return mark_step_done(state, "interview_extract", f"{count} quotes")
    except SystemExit:
        return mark_step_failed(state, "interview_extract", "no sources")
    except (Exception, SystemExit) as e:
        return mark_step_failed(state, "interview_extract", str(e))


def run_review_search(artist_name: str, artist_id: str, state: dict,
                      search_provider: str) -> dict:
    """Step: review_search"""
    if is_step_done(state, "review_search"):
        return state

    mark_step_running(state, "review_search")
    try:
        search_reviews_module = load_module("search_reviews")
        search_reviews_module.search_reviews(artist_name, search_provider)

        sources_path = Path(f"data/artists/{artist_id}/review_sources.json")
        count = 0
        if sources_path.exists():
            count = len(json.loads(sources_path.read_text()))

        return mark_step_done(state, "review_search", f"{count} urls")
    except (Exception, SystemExit) as e:
        return mark_step_failed(state, "review_search", str(e))


def run_review_scrape(artist_id: str, state: dict) -> dict:
    """Step: review_scrape"""
    if is_step_done(state, "review_scrape"):
        return state

    mark_step_running(state, "review_scrape")
    try:
        scrape_reviews_module = load_module("scrape_reviews")
        scrape_reviews_module.scrape_reviews(artist_id)

        sources_path = Path(f"data/artists/{artist_id}/review_sources.json")
        scraped = 0
        total = 0
        if sources_path.exists():
            sources = json.loads(sources_path.read_text())
            total = len(sources)
            scraped = sum(1 for s in sources if s.get("text"))

        return mark_step_done(state, "review_scrape", f"{scraped}/{total} scraped")
    except (Exception, SystemExit) as e:
        return mark_step_failed(state, "review_scrape", str(e))


def run_review_extract(artist_id: str, artist_name: str, state: dict,
                       model: str) -> dict:
    """Step: review_extract"""
    if is_step_done(state, "review_extract"):
        return state

    mark_step_running(state, "review_extract")
    try:
        extract_critic_module = load_module("extract_critic")
        extract_critic_module.extract_critic(artist_id, model)

        critic_path = Path(f"data/artists/{artist_id}/critic_quotes.json")
        count = 0
        if critic_path.exists():
            data = json.loads(critic_path.read_text())
            count = data.get("corpus_meta", {}).get("quote_count", 0)

        return mark_step_done(state, "review_extract", f"{count} passages")
    except SystemExit:
        return mark_step_failed(state, "review_extract", "no sources")
    except (Exception, SystemExit) as e:
        return mark_step_failed(state, "review_extract", str(e))


def run_influence_extract(artist_id: str, state: dict, model: str) -> dict:
    """Step: influence_extract"""
    if is_step_done(state, "influence_extract"):
        return state

    # Skip if no interview sources exist
    sources_path = Path(f"data/artists/{artist_id}/sources.json")
    if not sources_path.exists():
        return mark_step_skipped(state, "influence_extract", "no sources.json")

    sources = json.loads(sources_path.read_text())
    if not any(s.get("text") for s in sources):
        return mark_step_skipped(state, "influence_extract", "no scraped text")

    mark_step_running(state, "influence_extract")
    try:
        influence_module = load_module("extract_influences")
        result = influence_module.extract_influences(artist_id, model)
        count = result.get("meta", {}).get("count", 0)
        return mark_step_done(state, "influence_extract", f"{count} influences")
    except (Exception, SystemExit) as e:
        return mark_step_failed(state, "influence_extract", str(e))


def run_tag_extract(artist_id: str, state: dict, model: str) -> dict:
    """Step: tag_extract"""
    if is_step_done(state, "tag_extract"):
        return state

    # Check for any text sources
    has_text = False
    for fname in ("sources.json", "review_sources.json"):
        p = Path(f"data/artists/{artist_id}/{fname}")
        if p.exists():
            sources = json.loads(p.read_text())
            if any(s.get("text") for s in sources):
                has_text = True
                break

    if not has_text:
        return mark_step_skipped(state, "tag_extract", "no text sources")

    mark_step_running(state, "tag_extract")
    try:
        tag_module = load_module("extract_tags")
        result = tag_module.extract_tags(artist_id, model)
        count = result.get("meta", {}).get("count", 0)
        return mark_step_done(state, "tag_extract", f"{count} tags")
    except (Exception, SystemExit) as e:
        return mark_step_failed(state, "tag_extract", str(e))


# Step dispatch table
STEP_RUNNERS = {
    "interview_search": lambda ctx: run_interview_search(
        ctx["artist_name"], ctx["artist_id"], ctx["state"], ctx["search_provider"]),
    "interview_scrape": lambda ctx: run_interview_scrape(
        ctx["artist_id"], ctx["state"]),
    "interview_extract": lambda ctx: run_interview_extract(
        ctx["artist_id"], ctx["artist_name"], ctx["state"], ctx["model"]),
    "review_search": lambda ctx: run_review_search(
        ctx["artist_name"], ctx["artist_id"], ctx["state"], ctx["search_provider"]),
    "review_scrape": lambda ctx: run_review_scrape(
        ctx["artist_id"], ctx["state"]),
    "review_extract": lambda ctx: run_review_extract(
        ctx["artist_id"], ctx["artist_name"], ctx["state"], ctx["model"]),
    "influence_extract": lambda ctx: run_influence_extract(
        ctx["artist_id"], ctx["state"], ctx["model"]),
    "tag_extract": lambda ctx: run_tag_extract(
        ctx["artist_id"], ctx["state"], ctx["model"]),
}


# ── Progress tracking ────────────────────────────────────────────────────────

PROGRESS_PATH = Path("data/batch_progress.json")
# Each worker writes to its own shard to avoid race conditions
_WORKER_PROGRESS_PATH = Path(f"data/batch_progress_{os.getpid()}.json")


def update_progress(progress: dict):
    """Write live progress — per-worker shard + merged summary."""
    progress["last_updated"] = datetime.now(timezone.utc).isoformat()
    _WORKER_PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Write per-worker shard (no contention)
    tmp = _WORKER_PROGRESS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(progress, indent=2))
    tmp.rename(_WORKER_PROGRESS_PATH)

    # Merge all worker shards into the shared progress file (best-effort)
    try:
        _merge_progress_shards()
    except Exception:
        pass


def _merge_progress_shards():
    """Aggregate all per-worker progress shards into batch_progress.json."""
    import glob as _glob
    shards = _glob.glob("data/batch_progress_*.json")
    if not shards:
        return
    totals = {"completed": 0, "failed": 0, "total_artists": 0, "queued": 0,
              "skipped_complete": 0, "estimated_cost": 0.0, "artist_times": []}
    latest_updated = ""
    in_progress_list = []
    started = None
    for path in shards:
        try:
            d = json.loads(Path(path).read_text())
            totals["completed"] += d.get("completed", 0)
            totals["failed"] += d.get("failed", 0)
            totals["total_artists"] += d.get("total_artists", 0)
            totals["queued"] += d.get("queued", 0)
            totals["skipped_complete"] += d.get("skipped_complete", 0)
            totals["estimated_cost"] += d.get("estimated_cost", 0.0)
            totals["artist_times"].extend(d.get("artist_times", []))
            lu = d.get("last_updated", "")
            if lu > latest_updated:
                latest_updated = lu
            if d.get("in_progress"):
                in_progress_list.append(d["in_progress"])
            if not started:
                started = d.get("started")
        except Exception:
            pass
    merged = {**totals,
              "started": started,
              "in_progress": in_progress_list[0] if in_progress_list else "",
              "current_step": "",
              "last_updated": latest_updated,
              "worker_count": len(shards)}
    tmp = PROGRESS_PATH.parent / f"batch_progress.merge_{os.getpid()}.tmp"
    tmp.write_text(json.dumps(merged, indent=2))
    try:
        tmp.rename(PROGRESS_PATH)
    except Exception:
        tmp.unlink(missing_ok=True)


# ── Main batch orchestrator ──────────────────────────────────────────────────

def build_work_queue(artists: list, phase: str) -> list:
    """Build ordered work queue: partial artists first, then new ones."""
    scan = scan_all_artist_states()

    partial = []
    new = []

    for artist_name in artists:
        artist_id = normalize_artist_id(artist_name)
        state = load_state(artist_id, artist_name)

        if is_artist_complete(state, phase):
            continue  # Already done

        # Check if any steps are done (partial work)
        has_work = any(
            state["steps"].get(s, {}).get("status") in ("done", "skipped")
            for s in STEPS
        )

        entry = {"artist_name": artist_name, "artist_id": artist_id, "state": state}
        if has_work:
            partial.append(entry)
        else:
            new.append(entry)

    # Partial first (finish what we started), then new
    return partial + new


def batch_v2(artists: list, search_provider: str = "brave",
             delay: float = 1.5, model: str = "claude-haiku-4-5-20251001",
             phase: str = "both", max_credits: float = 0,
             dry_run: bool = False) -> dict:
    """Process artists with state-machine-driven pipeline."""

    work_queue = build_work_queue(artists, phase)

    progress = {
        "started": datetime.now(timezone.utc).isoformat(),
        "total_artists": len(artists),
        "queued": len(work_queue),
        "skipped_complete": len(artists) - len(work_queue),
        "completed": 0,
        "failed": 0,
        "in_progress": "",
        "current_step": "",
        "estimated_cost": 0.0,
        "artist_times": [],
    }

    if dry_run:
        print(f"\n[DRY RUN] Would process {len(work_queue)} artists")
        print(f"  Already complete: {progress['skipped_complete']}")
        print(f"  Partial (resume): {sum(1 for w in work_queue if any(w['state']['steps'].get(s, {}).get('status') in ('done', 'skipped') for s in STEPS))}")
        print(f"  New: {sum(1 for w in work_queue if not any(w['state']['steps'].get(s, {}).get('status') in ('done', 'skipped') for s in STEPS))}")
        est = len(work_queue) * EST_COST_PER_ARTIST
        print(f"  Estimated cost: ${est:.2f}")
        return progress

    print(f"\nBatch v2: {len(work_queue)} artists to process "
          f"({progress['skipped_complete']} already complete)")
    print(f"Model: {model} | Phase: {phase} | Search: {search_provider}")

    if max_credits > 0:
        print(f"Budget cap: ${max_credits:.2f}")

    update_progress(progress)
    credits_spent = 0.0

    batch_state = load_batch_state()

    for i, entry in enumerate(work_queue, 1):
        artist_name = entry["artist_name"]
        artist_id = entry["artist_id"]
        state = entry["state"]

        # Budget check
        if max_credits > 0 and credits_spent >= max_credits:
            print(f"\n[BUDGET] Reached ${max_credits:.2f} cap after {i-1} artists")
            break

        print(f"\n{'='*70}")
        print(f"[{i}/{len(work_queue)}] {artist_name} ({artist_id})")
        next_step = next_incomplete_step(state, phase)
        print(f"  Resuming from: {next_step}")
        print(f"{'='*70}")

        progress["in_progress"] = artist_name
        update_progress(progress)

        artist_start = time.time()
        api_calls_this_artist = 0

        # Context for step runners
        ctx = {
            "artist_name": artist_name,
            "artist_id": artist_id,
            "state": state,
            "search_provider": search_provider,
            "model": model,
        }

        # Run each incomplete step
        while True:
            step = next_incomplete_step(state, phase)
            if step is None:
                break

            progress["current_step"] = step
            update_progress(progress)

            print(f"\n  [{step}]")

            runner = STEP_RUNNERS.get(step)
            if not runner:
                print(f"  WARNING: No runner for step {step}, skipping")
                mark_step_skipped(state, step, "no runner")
                continue

            ctx["state"] = state
            state = runner(ctx)

            # Track cost for extraction steps
            if "extract" in step and state["steps"][step]["status"] == "done":
                # 4 sources max per extraction step (capped in extract_llm/extract_critic)
                api_calls_this_artist += 4
                credits_spent += 4 * COST_PER_EXTRACT_CALL

            if state["steps"][step]["status"] == "failed":
                print(f"  FAILED: {state.get('error', 'unknown error')}")
                # Mark as skipped so next_incomplete_step advances past it
                mark_step_skipped(state, step, "failed — skipping to unblock")
                continue

            time.sleep(0.5)  # Brief pause between steps

        # Artist complete
        elapsed = time.time() - artist_start
        progress["artist_times"].append(round(elapsed, 1))

        if is_artist_complete(state, phase):
            progress["completed"] += 1
            print(f"\n  COMPLETE ({elapsed:.0f}s)")
        else:
            progress["failed"] += 1
            print(f"\n  PARTIAL ({elapsed:.0f}s) — some steps failed")

        progress["estimated_cost"] = round(credits_spent, 4)
        update_progress(progress)

        # Update batch state
        if artist_id not in batch_state["artists_visited"]:
            batch_state["artists_visited"].append(artist_id)
        if is_artist_complete(state, phase):
            if artist_id not in batch_state["artists_complete"]:
                batch_state["artists_complete"].append(artist_id)
        save_batch_state(batch_state)

        # Delay between artists
        if i < len(work_queue):
            time.sleep(delay)

    # Final summary
    progress["finished"] = datetime.now(timezone.utc).isoformat()
    update_progress(progress)

    avg_time = (sum(progress["artist_times"]) / len(progress["artist_times"])
                if progress["artist_times"] else 0)

    print(f"\n{'='*70}")
    print(f"[BATCH COMPLETE]")
    print(f"{'='*70}")
    print(f"  Completed: {progress['completed']}")
    print(f"  Failed:    {progress['failed']}")
    print(f"  Skipped:   {progress['skipped_complete']}")
    print(f"  Avg time:  {avg_time:.0f}s per artist")
    print(f"  Est. cost: ${credits_spent:.4f}")

    return progress


# ── CLI ──────────────────────────────────────────────────────────────────────

def load_artist_list(file_path: str = None, artists_str: str = None) -> list:
    """Load artist names from file or comma-separated string."""
    if file_path:
        path = Path(file_path)
        if not path.exists():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    elif artists_str:
        return [name.strip() for name in artists_str.split(',')]
    else:
        # Default: read seed_artists.txt
        seed = Path("data/seed_artists.txt")
        if seed.exists():
            return [line.strip() for line in seed.read_text().splitlines() if line.strip()]
        print("Error: Provide --file, --artists, or have data/seed_artists.txt")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch v2: state-machine-driven, crash-resilient artist ingestion"
    )
    parser.add_argument("--file", help="Path to artist list file (one per line)")
    parser.add_argument("--artists", help="Comma-separated artist names")
    parser.add_argument("--search-provider", choices=["brave", "serpapi", "google"],
                       default="brave", help="Search provider (default: brave)")
    parser.add_argument("--delay", type=float, default=1.5,
                       help="Delay between artists in seconds (default: 1.5)")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001",
                       help="Anthropic model for extraction (default: haiku 4.5)")
    parser.add_argument("--phase", choices=["interview", "critic", "both"],
                       default="both", help="Which signal pipelines to run (default: both)")
    parser.add_argument("--max-credits", type=float, default=0,
                       help="Stop after spending ~this many dollars (0 = unlimited)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print what would be done without making any API calls")
    parser.add_argument("--no-pipeline", action="store_true",
                       help="Don't run global pipeline after batch")

    args = parser.parse_args()

    artists = load_artist_list(args.file, args.artists)
    print(f"Loaded {len(artists)} artist(s)")

    progress = batch_v2(
        artists=artists,
        search_provider=args.search_provider,
        delay=args.delay,
        model=args.model,
        phase=args.phase,
        max_credits=args.max_credits,
        dry_run=args.dry_run,
    )

    if not args.dry_run and not args.no_pipeline and progress.get("completed", 0) > 0:
        print(f"\nTo run global pipeline:")
        print(f"  python scripts/embed.py && python scripts/compute.py && python scripts/discover.py")
