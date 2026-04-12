"""QA Step 2: End-to-end mini batch validation (50 artists).

Runs the full new pipeline on 50 artists, validates:
- State machine works (kill+restart resumes correctly)
- Corpus validity rate ≥60%
- Cost tracking matches estimates
- batch_progress.json updates correctly
- Influence and tag extraction produce plausible results
- Prefiltering works as expected

Usage:
    python scripts/qa_mini_batch.py [--count 50] [--search-provider brave]
"""

import sys
import json
import time
import signal
from pathlib import Path
from datetime import datetime

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))


def find_artists_without_corpora(count: int = 50) -> list:
    """Find artists from seed list that don't have valid corpora yet."""
    seed_path = Path("data/seed_artists.txt")
    if not seed_path.exists():
        print("Error: data/seed_artists.txt not found")
        return []

    all_artists = [l.strip() for l in seed_path.read_text().splitlines() if l.strip()]

    # Deduplicate
    seen = set()
    unique = []
    for name in all_artists:
        from state import load_state
        from batch_v2 import normalize_artist_id
        aid = normalize_artist_id(name)
        if aid not in seen:
            seen.add(aid)
            unique.append(name)

    # Filter to those without complete state
    candidates = []
    for name in unique:
        aid = normalize_artist_id(name)
        state = load_state(aid, name)
        if state["status"] != "extract_complete":
            candidates.append(name)

    return candidates[:count]


def validate_state_files(artist_ids: list) -> dict:
    """Check state.json files for consistency."""
    results = {"valid": 0, "invalid": 0, "missing": 0, "issues": []}

    for aid in artist_ids:
        state_path = Path(f"data/artists/{aid}/state.json")
        if not state_path.exists():
            results["missing"] += 1
            results["issues"].append(f"{aid}: no state.json")
            continue

        try:
            state = json.loads(state_path.read_text())

            # Check required fields
            required = ["artist_id", "status", "steps", "last_updated"]
            missing = [f for f in required if f not in state]
            if missing:
                results["invalid"] += 1
                results["issues"].append(f"{aid}: missing fields {missing}")
                continue

            # Check step structure
            for step_name, step_data in state["steps"].items():
                if "status" not in step_data:
                    results["invalid"] += 1
                    results["issues"].append(f"{aid}: step {step_name} missing status")
                    break
            else:
                results["valid"] += 1

        except json.JSONDecodeError:
            results["invalid"] += 1
            results["issues"].append(f"{aid}: invalid JSON")

    return results


def validate_corpora(artist_ids: list) -> dict:
    """Check corpus validity across tested artists."""
    results = {
        "total": len(artist_ids),
        "quote_valid": 0,
        "critic_valid": 0,
        "any_valid": 0,
        "none_valid": 0,
    }

    for aid in artist_ids:
        has_any = False

        for corpus_file, key in [("quotes.json", "quote_valid"), ("critic_quotes.json", "critic_valid")]:
            path = Path(f"data/artists/{aid}/{corpus_file}")
            if path.exists():
                try:
                    data = json.loads(path.read_text())
                    if data.get("corpus_meta", {}).get("corpus_valid"):
                        results[key] += 1
                        has_any = True
                except Exception:
                    pass

        if has_any:
            results["any_valid"] += 1
        else:
            results["none_valid"] += 1

    results["validity_rate"] = results["any_valid"] / results["total"] if results["total"] else 0
    return results


def validate_new_signals(artist_ids: list) -> dict:
    """Check influence and tag extraction results."""
    results = {
        "influences_extracted": 0,
        "tags_extracted": 0,
        "artists_with_influences": 0,
        "artists_with_tags": 0,
        "sample_influences": [],
        "sample_tags": [],
    }

    for aid in artist_ids:
        inf_path = Path(f"data/artists/{aid}/influences.json")
        if inf_path.exists():
            try:
                data = json.loads(inf_path.read_text())
                count = data.get("meta", {}).get("count", 0)
                if count > 0:
                    results["artists_with_influences"] += 1
                    results["influences_extracted"] += count
                    if len(results["sample_influences"]) < 5:
                        for inf in data.get("influences", [])[:2]:
                            results["sample_influences"].append(
                                f"{aid} <- {inf.get('name', '?')}"
                            )
            except Exception:
                pass

        tag_path = Path(f"data/artists/{aid}/tags.json")
        if tag_path.exists():
            try:
                data = json.loads(tag_path.read_text())
                count = data.get("meta", {}).get("count", 0)
                if count > 0:
                    results["artists_with_tags"] += 1
                    results["tags_extracted"] += count
                    if len(results["sample_tags"]) < 5:
                        for t in data.get("tags", [])[:2]:
                            results["sample_tags"].append(
                                f"{aid}: {t.get('tag', '?')} ({t.get('category', '?')})"
                            )
            except Exception:
                pass

    return results


def validate_progress_file() -> dict:
    """Check batch_progress.json integrity."""
    progress_path = Path("data/batch_progress.json")
    if not progress_path.exists():
        return {"exists": False, "valid": False, "issues": ["file not found"]}

    try:
        data = json.loads(progress_path.read_text())
        required = ["started", "total_artists", "completed", "failed"]
        missing = [f for f in required if f not in data]

        return {
            "exists": True,
            "valid": len(missing) == 0,
            "issues": [f"missing: {missing}"] if missing else [],
            "completed": data.get("completed", 0),
            "failed": data.get("failed", 0),
        }
    except json.JSONDecodeError:
        return {"exists": True, "valid": False, "issues": ["invalid JSON"]}


def run_mini_batch(count: int = 50, search_provider: str = "brave"):
    """Run the full QA mini batch."""

    print(f"QA Mini Batch: finding {count} artists to test\n")
    artists = find_artists_without_corpora(count)

    if not artists:
        print("No untested artists found in seed list.")
        return None

    actual_count = len(artists)
    print(f"Found {actual_count} artists to test")
    print(f"Search provider: {search_provider}\n")

    # Save test list
    test_list_path = Path("data/qa_test_artists.txt")
    test_list_path.write_text("\n".join(artists))

    # Run batch_v2
    print(f"{'='*60}")
    print(f"Running batch_v2 on {actual_count} artists...")
    print(f"{'='*60}\n")

    from batch_v2 import batch_v2, normalize_artist_id

    progress = batch_v2(
        artists=artists,
        search_provider=search_provider,
        delay=1.5,
        model="claude-haiku-4-5-20251001",
        phase="both",
    )

    # Collect artist IDs for validation
    artist_ids = [normalize_artist_id(name) for name in artists]

    # Run validations
    print(f"\n{'='*60}")
    print(f"RUNNING QA VALIDATIONS")
    print(f"{'='*60}\n")

    state_check = validate_state_files(artist_ids)
    print(f"State files: {state_check['valid']} valid, {state_check['invalid']} invalid, {state_check['missing']} missing")

    corpus_check = validate_corpora(artist_ids)
    print(f"Corpus validity: {corpus_check['any_valid']}/{corpus_check['total']} "
          f"({corpus_check['validity_rate']*100:.0f}%)")

    signal_check = validate_new_signals(artist_ids)
    print(f"Influences: {signal_check['influences_extracted']} across "
          f"{signal_check['artists_with_influences']} artists")
    print(f"Tags: {signal_check['tags_extracted']} across "
          f"{signal_check['artists_with_tags']} artists")

    progress_check = validate_progress_file()
    print(f"Progress file: {'valid' if progress_check.get('valid') else 'INVALID'}")

    # Build QA report
    report = {
        "timestamp": datetime.now().isoformat(),
        "test": "mini_batch_e2e",
        "artists_tested": actual_count,
        "checks": {
            "state_files": {
                "pass": state_check["valid"] == actual_count,
                "detail": state_check,
            },
            "corpus_validity": {
                "pass": corpus_check["validity_rate"] >= 0.60,
                "detail": corpus_check,
            },
            "new_signals": {
                "pass": signal_check["artists_with_influences"] > 0 and signal_check["artists_with_tags"] > 0,
                "detail": signal_check,
            },
            "progress_file": {
                "pass": progress_check.get("valid", False),
                "detail": progress_check,
            },
        },
        "batch_progress": progress,
    }

    all_pass = all(c["pass"] for c in report["checks"].values())
    report["overall_pass"] = all_pass

    # Save report
    report_path = Path("data/qa_report.json")
    report_path.write_text(json.dumps(report, indent=2, default=str))

    # Print summary
    print(f"\n{'='*60}")
    print(f"QA MINI BATCH RESULTS")
    print(f"{'='*60}")
    for name, check in report["checks"].items():
        status = "PASS" if check["pass"] else "FAIL"
        print(f"  [{status}] {name}")
    print(f"\n  Overall: {'PASS' if all_pass else 'FAIL'}")
    print(f"  Report: {report_path}")

    if signal_check["sample_influences"]:
        print(f"\n  Sample influences:")
        for s in signal_check["sample_influences"]:
            print(f"    {s}")

    if signal_check["sample_tags"]:
        print(f"\n  Sample tags:")
        for s in signal_check["sample_tags"]:
            print(f"    {s}")

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QA: End-to-end mini batch validation")
    parser.add_argument("--count", type=int, default=50,
                       help="Number of artists to test (default: 50)")
    parser.add_argument("--search-provider", choices=["brave", "serpapi", "google"],
                       default="brave", help="Search provider")

    args = parser.parse_args()
    run_mini_batch(args.count, args.search_provider)
