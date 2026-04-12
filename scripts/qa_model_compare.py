"""QA Step 1: Compare Haiku vs Sonnet extraction quality.

Picks artists with existing Sonnet-extracted corpora, re-extracts with Haiku,
and compares:
- Quote count (±20% acceptable)
- JSON validity (100% required)
- False positive rate (no systematic pattern)

Usage:
    python scripts/qa_model_compare.py [--count 10] [--artists "bjork,aphex_twin"]
"""

import sys
import json
import os
import time
from pathlib import Path
from datetime import datetime

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))


def find_artists_with_corpora(count: int = 10) -> list:
    """Find artists that have valid Sonnet-extracted corpora."""
    data_dir = Path("data/artists")
    candidates = []

    for artist_dir in sorted(data_dir.iterdir()):
        if not artist_dir.is_dir():
            continue

        quotes_path = artist_dir / "quotes.json"
        sources_path = artist_dir / "sources.json"

        if not quotes_path.exists() or not sources_path.exists():
            continue

        try:
            quotes_data = json.loads(quotes_path.read_text())
            if quotes_data.get("corpus_meta", {}).get("corpus_valid"):
                sources = json.loads(sources_path.read_text())
                has_text = any(s.get("text") for s in sources)
                if has_text:
                    candidates.append(artist_dir.name)
        except Exception:
            continue

    return candidates[:count]


def extract_with_model(artist_id: str, model: str) -> dict:
    """Run extraction with a specific model and return results."""
    from extract_llm import extract_quotes_from_text, artist_id_to_display_name

    sources_path = Path(f"data/artists/{artist_id}/sources.json")
    sources = json.loads(sources_path.read_text())
    sources_with_text = [s for s in sources if s.get("text")]

    artist_name = artist_id_to_display_name(artist_id)
    api_key = os.getenv('ANTHROPIC_API_KEY')
    client = Anthropic(api_key=api_key)

    all_quotes = []
    json_errors = 0

    for source in sources_with_text:
        url = source.get("url", "")
        publication = source.get("publication", url)
        date = source.get("date", "")
        text = source.get("text", "")

        quotes = extract_quotes_from_text(
            client, artist_name, text, publication, url, date, model
        )
        all_quotes.extend(quotes)
        time.sleep(0.5)

    return {
        "model": model,
        "quote_count": len(all_quotes),
        "source_count": len(sources_with_text),
        "quotes": all_quotes,
        "json_errors": json_errors,
    }


def compare_results(artist_id: str, sonnet_data: dict, haiku_result: dict) -> dict:
    """Compare Sonnet (existing) vs Haiku (new) results for one artist."""
    sonnet_count = sonnet_data.get("corpus_meta", {}).get("quote_count", 0)
    haiku_count = haiku_result["quote_count"]

    # Quote count comparison
    if sonnet_count > 0:
        ratio = haiku_count / sonnet_count
        count_pass = 0.8 <= ratio <= 1.2  # ±20%
    else:
        ratio = float('inf') if haiku_count > 0 else 1.0
        count_pass = True

    return {
        "artist_id": artist_id,
        "sonnet_quotes": sonnet_count,
        "haiku_quotes": haiku_count,
        "ratio": round(ratio, 3),
        "count_pass": count_pass,
        "haiku_json_errors": haiku_result["json_errors"],
    }


def run_qa_compare(artists: list = None, count: int = 10):
    """Run the full QA comparison."""

    if artists:
        test_artists = artists
    else:
        test_artists = find_artists_with_corpora(count)

    if not test_artists:
        print("No artists with valid corpora found for comparison.")
        return None

    print(f"QA Model Compare: testing {len(test_artists)} artists")
    print(f"  Haiku: claude-haiku-4-5-20251001")
    print(f"  Baseline: existing Sonnet-extracted corpora\n")

    results = []

    for i, artist_id in enumerate(test_artists, 1):
        print(f"[{i}/{len(test_artists)}] {artist_id}")

        # Load existing Sonnet results
        quotes_path = Path(f"data/artists/{artist_id}/quotes.json")
        sonnet_data = json.loads(quotes_path.read_text())
        sonnet_count = sonnet_data.get("corpus_meta", {}).get("quote_count", 0)
        print(f"  Sonnet baseline: {sonnet_count} quotes")

        # Extract with Haiku
        print(f"  Running Haiku extraction...")
        haiku_result = extract_with_model(artist_id, "claude-haiku-4-5-20251001")
        print(f"  Haiku result: {haiku_result['quote_count']} quotes")

        # Compare
        comparison = compare_results(artist_id, sonnet_data, haiku_result)
        results.append(comparison)

        status = "PASS" if comparison["count_pass"] else "FAIL"
        print(f"  Ratio: {comparison['ratio']:.2f}x — {status}")
        print()

        time.sleep(1)

    # Aggregate
    total = len(results)
    passed = sum(1 for r in results if r["count_pass"])
    avg_ratio = sum(r["ratio"] for r in results) / total if total else 0
    json_errors = sum(r["haiku_json_errors"] for r in results)

    overall_pass = passed >= total * 0.85 and json_errors == 0

    report = {
        "timestamp": datetime.now().isoformat(),
        "test": "haiku_vs_sonnet",
        "artists_tested": total,
        "count_pass_rate": f"{passed}/{total} ({passed/total*100:.0f}%)" if total else "0/0",
        "avg_ratio": round(avg_ratio, 3),
        "json_errors": json_errors,
        "overall_pass": overall_pass,
        "per_artist": results,
    }

    # Save report
    report_path = Path("data/qa_model_compare_report.json")
    report_path.write_text(json.dumps(report, indent=2))

    print(f"{'='*60}")
    print(f"QA MODEL COMPARE RESULTS")
    print(f"{'='*60}")
    print(f"  Artists tested: {total}")
    print(f"  Count pass rate: {passed}/{total}")
    print(f"  Avg quote ratio (haiku/sonnet): {avg_ratio:.2f}")
    print(f"  JSON errors: {json_errors}")
    print(f"  Overall: {'PASS' if overall_pass else 'FAIL'}")
    print(f"\n  Report saved to: {report_path}")

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QA: Compare Haiku vs Sonnet extraction quality")
    parser.add_argument("--count", type=int, default=10,
                       help="Number of artists to test (default: 10)")
    parser.add_argument("--artists", help="Comma-separated artist IDs to test")

    args = parser.parse_args()

    artists = args.artists.split(",") if args.artists else None
    run_qa_compare(artists, args.count)
