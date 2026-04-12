"""Per-artist state machine for crash-resilient pipeline processing.

Each artist gets a state.json tracking per-step completion status.
On restart, the pipeline resumes from the first incomplete step —
never re-runs search, scrape, or extraction that already succeeded.

State writes are atomic (write to .tmp, rename) to survive mid-write crashes.
"""

import json
import os
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional


# All pipeline steps in execution order
STEPS = [
    "interview_search",
    "interview_scrape",
    "interview_extract",
    "review_search",
    "review_scrape",
    "review_extract",
    "influence_extract",
    "tag_extract",
]

# Valid step statuses
STEP_PENDING = "pending"
STEP_RUNNING = "running"
STEP_DONE = "done"
STEP_FAILED = "failed"
STEP_SKIPPED = "skipped"

# Valid artist-level statuses
STATUS_PENDING = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_EXTRACT_COMPLETE = "extract_complete"
STATUS_FAILED = "failed"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: Path, data: dict):
    """Write JSON atomically: write to .tmp then rename."""
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp_path.rename(path)


def _default_state(artist_id: str, artist_name: str) -> dict:
    """Create a fresh state dict with all steps pending."""
    return {
        "artist_id": artist_id,
        "artist_name": artist_name,
        "status": STATUS_PENDING,
        "steps": {
            step: {"status": STEP_PENDING, "timestamp": None, "result": None}
            for step in STEPS
        },
        "error": None,
        "last_updated": _now_iso(),
    }


def state_path(artist_id: str) -> Path:
    """Return the path to an artist's state.json."""
    return Path(f"data/artists/{artist_id}/state.json")


def load_state(artist_id: str, artist_name: str = "") -> dict:
    """Load artist state from disk, creating default if missing."""
    path = state_path(artist_id)
    if path.exists():
        try:
            state = json.loads(path.read_text())
            # Ensure any new steps are present (forward-compat)
            for step in STEPS:
                if step not in state.get("steps", {}):
                    state.setdefault("steps", {})[step] = {
                        "status": STEP_PENDING,
                        "timestamp": None,
                        "result": None,
                    }
            return state
        except (json.JSONDecodeError, KeyError):
            pass

    # Create fresh state
    display_name = artist_name or artist_id.replace("_", " ").title()
    return _default_state(artist_id, display_name)


def save_state(state: dict):
    """Persist artist state atomically."""
    artist_id = state["artist_id"]
    path = state_path(artist_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    state["last_updated"] = _now_iso()
    _atomic_write(path, state)


def mark_step_running(state: dict, step: str) -> dict:
    """Mark a step as currently running."""
    state["steps"][step]["status"] = STEP_RUNNING
    state["steps"][step]["timestamp"] = _now_iso()
    state["status"] = STATUS_IN_PROGRESS
    save_state(state)
    return state


def mark_step_done(state: dict, step: str, result: str = "") -> dict:
    """Mark a step as successfully completed."""
    state["steps"][step]["status"] = STEP_DONE
    state["steps"][step]["timestamp"] = _now_iso()
    state["steps"][step]["result"] = result
    state["error"] = None
    _update_overall_status(state)
    save_state(state)
    return state


def mark_step_failed(state: dict, step: str, error: str = "") -> dict:
    """Mark a step as failed."""
    state["steps"][step]["status"] = STEP_FAILED
    state["steps"][step]["timestamp"] = _now_iso()
    state["steps"][step]["result"] = None
    state["error"] = error
    state["status"] = STATUS_FAILED
    save_state(state)
    return state


def mark_step_skipped(state: dict, step: str, reason: str = "") -> dict:
    """Mark a step as skipped (e.g., no sources to scrape)."""
    state["steps"][step]["status"] = STEP_SKIPPED
    state["steps"][step]["timestamp"] = _now_iso()
    state["steps"][step]["result"] = reason
    _update_overall_status(state)
    save_state(state)
    return state


def _update_overall_status(state: dict):
    """Recompute artist-level status from step statuses."""
    steps = state["steps"]
    all_terminal = all(
        steps[s]["status"] in (STEP_DONE, STEP_SKIPPED)
        for s in STEPS
    )
    any_failed = any(steps[s]["status"] == STEP_FAILED for s in STEPS)

    if all_terminal:
        state["status"] = STATUS_EXTRACT_COMPLETE
    elif any_failed:
        state["status"] = STATUS_FAILED
    else:
        state["status"] = STATUS_IN_PROGRESS


def is_step_done(state: dict, step: str) -> bool:
    """Check if a step has already completed successfully."""
    return state["steps"].get(step, {}).get("status") in (STEP_DONE, STEP_SKIPPED)


def next_incomplete_step(state: dict, phase: str = "both") -> Optional[str]:
    """Return the first incomplete step, or None if all done.

    phase: 'interview' = only interview steps
           'critic'    = only review/critic steps
           'both'      = all steps (default)
    """
    for step in STEPS:
        # Phase filtering
        if phase == "interview" and not step.startswith("interview_"):
            continue
        if phase == "critic" and not step.startswith("review_"):
            continue

        if not is_step_done(state, step):
            return step

    return None


def is_artist_complete(state: dict, phase: str = "both") -> bool:
    """Check if all relevant steps for an artist are done."""
    return next_incomplete_step(state, phase) is None


# ── Batch-level state ────────────────────────────────────────────────────────

BATCH_STATE_PATH = Path("data/batch_state.json")


def load_batch_state() -> dict:
    """Load global batch state tracking which artists have been visited."""
    if BATCH_STATE_PATH.exists():
        try:
            return json.loads(BATCH_STATE_PATH.read_text())
        except (json.JSONDecodeError, KeyError):
            pass
    return {
        "started": _now_iso(),
        "artists_visited": [],
        "artists_complete": [],
        "artists_failed": [],
        "last_updated": _now_iso(),
    }


def save_batch_state(batch: dict):
    """Persist batch state atomically."""
    batch["last_updated"] = _now_iso()
    BATCH_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(BATCH_STATE_PATH, batch)


def scan_all_artist_states() -> dict:
    """Scan all artist directories and return a summary of states.

    Returns dict with:
        complete: list of artist_ids fully done
        partial:  list of artist_ids with some work done
        pending:  list of artist_ids with no state.json
        failed:   list of artist_ids in failed state
        states:   dict of artist_id -> full state dict
    """
    data_dir = Path("data/artists")
    result = {
        "complete": [],
        "partial": [],
        "pending": [],
        "failed": [],
        "states": {},
    }

    if not data_dir.exists():
        return result

    for artist_dir in sorted(data_dir.iterdir()):
        if not artist_dir.is_dir():
            continue

        artist_id = artist_dir.name
        sp = artist_dir / "state.json"

        if not sp.exists():
            # Check if there's any data at all (legacy artists without state.json)
            has_data = (artist_dir / "quotes.json").exists() or (artist_dir / "sources.json").exists()
            if has_data:
                result["partial"].append(artist_id)
            else:
                result["pending"].append(artist_id)
            continue

        try:
            state = json.loads(sp.read_text())
            result["states"][artist_id] = state

            status = state.get("status", STATUS_PENDING)
            if status == STATUS_EXTRACT_COMPLETE:
                result["complete"].append(artist_id)
            elif status == STATUS_FAILED:
                result["failed"].append(artist_id)
            elif status == STATUS_IN_PROGRESS:
                result["partial"].append(artist_id)
            else:
                result["pending"].append(artist_id)
        except (json.JSONDecodeError, KeyError):
            result["pending"].append(artist_id)

    return result
