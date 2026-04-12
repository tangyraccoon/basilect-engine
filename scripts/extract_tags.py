"""Extract thematic creative-approach tags from interview + review text.

Tags capture process, philosophy, and approach dimensions:
- Process: texture, improvisation, sampling, field_recording, layering, etc.
- Philosophy: anti-commercial, DIY, collaborative, isolation, spiritual, etc.
- Approach: studio_as_instrument, live_performance, digital, analog, hybrid, etc.

Cheap to extract — Haiku handles this in a single short call per source.
Output: data/artists/{artist_id}/tags.json
"""

import sys
import json
import os
import time
from pathlib import Path
from collections import Counter

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def artist_id_to_display_name(artist_id: str) -> str:
    return artist_id.replace('_', ' ').title()


# Canonical tag vocabulary — the LLM should prefer these but can suggest new ones
CANONICAL_TAGS = {
    "process": [
        "texture", "improvisation", "sampling", "field_recording", "layering",
        "minimalism", "maximalism", "repetition", "drone", "noise",
        "collage", "loop_based", "generative", "algorithmic", "chance_operations",
        "found_sound", "vocal_processing", "synthesis", "orchestration", "polyrhythm",
    ],
    "philosophy": [
        "anti_commercial", "diy", "collaborative", "isolation", "spiritual",
        "political", "conceptual", "intuitive", "intellectual", "emotional",
        "confrontational", "meditative", "playful", "serious", "autobiographical",
        "abstract", "narrative", "experimental", "traditional", "hybrid",
    ],
    "approach": [
        "studio_as_instrument", "live_performance", "digital", "analog",
        "acoustic", "electronic", "hybrid_production", "lo_fi", "hi_fi",
        "self_produced", "collaborative_production", "multi_instrumental",
        "single_instrument", "vocal_focused", "instrumental_focused",
        "long_form", "short_form", "immersive", "sparse",
    ],
}


def extract_tags_from_text(client: Anthropic, artist_name: str, text: str,
                           model: str = DEFAULT_MODEL) -> list:
    """Extract creative-approach tags from a single article."""

    tag_list = []
    for category, tags in CANONICAL_TAGS.items():
        tag_list.append(f"  {category}: {', '.join(tags)}")
    tag_reference = "\n".join(tag_list)

    prompt = f"""You are tagging a music article about {artist_name}'s creative approach.

Read the text and assign tags that describe {artist_name}'s creative process, philosophy, and approach to making music. Use the canonical vocabulary below when possible, but you may add 1-2 new tags if something important is clearly discussed but not in the list.

CANONICAL TAGS:
{tag_reference}

RULES:
- Only tag what is explicitly discussed or strongly implied in the text
- Each tag should have a confidence score: high (clearly discussed), medium (implied), low (mentioned in passing)
- Include the category (process/philosophy/approach) for each tag
- Return 3-15 tags per article
- Do NOT tag genre labels (rock, jazz, electronic) — we want creative approach, not genre

Return a JSON array: [{{"tag": "tag_name", "category": "process|philosophy|approach", "confidence": "high|medium|low"}}]
Return ONLY the JSON array, no other text.

TEXT:
{text[:6000]}"""

    try:
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = message.content[0].text
        import re as _re
        response_text = response_text.strip()
        if response_text.startswith("```"):
            response_text = _re.sub(r'^```\w*\n?', '', response_text)
            response_text = _re.sub(r'\n?```$', '', response_text)
            response_text = response_text.strip()
        try:
            tags = json.loads(response_text)
            if not isinstance(tags, list):
                return []
            return tags
        except json.JSONDecodeError:
            return []
    except Exception as e:
        print(f"  ERROR: API call failed: {e}")
        return []


def aggregate_tags(all_tag_lists: list) -> dict:
    """Aggregate tags across multiple sources into weighted tag set.

    Tags that appear in multiple sources with high confidence get higher weight.
    """
    tag_scores = Counter()
    tag_categories = {}

    confidence_weights = {"high": 3, "medium": 2, "low": 1}

    for tag_list in all_tag_lists:
        for t in tag_list:
            tag = t.get("tag", "").lower().strip()
            category = t.get("category", "unknown")
            confidence = t.get("confidence", "medium")

            if not tag:
                continue

            weight = confidence_weights.get(confidence, 1)
            tag_scores[tag] += weight
            tag_categories[tag] = category

    # Normalize scores to 0-1 range
    max_score = max(tag_scores.values()) if tag_scores else 1
    weighted_tags = []

    for tag, score in tag_scores.most_common():
        weighted_tags.append({
            "tag": tag,
            "category": tag_categories.get(tag, "unknown"),
            "weight": round(score / max_score, 3),
            "raw_score": score,
        })

    return weighted_tags


def extract_tags(artist_id: str, model: str = DEFAULT_MODEL):
    """Extract thematic tags for an artist from all available text sources."""

    print(f"Extracting tags for: {artist_id}")

    # Gather text from both interview and review sources
    source_files = [
        ("interview", Path(f"data/artists/{artist_id}/sources.json")),
        ("review", Path(f"data/artists/{artist_id}/review_sources.json")),
    ]

    all_sources = []
    for source_type, path in source_files:
        if path.exists():
            try:
                sources = json.loads(path.read_text())
                for s in sources:
                    if s.get("text"):
                        s["_source_type"] = source_type
                        all_sources.append(s)
            except json.JSONDecodeError:
                pass

    if not all_sources:
        print("No sources with text found.")
        return {"tags": [], "meta": {"count": 0}}

    print(f"Found {len(all_sources)} source(s) with text")

    artist_name = artist_id_to_display_name(artist_id)
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = Anthropic(api_key=api_key)
    all_tag_lists = []

    for i, source in enumerate(all_sources, 1):
        text = source.get("text", "")
        pub = source.get("publication", "unknown")
        print(f"  [{i}/{len(all_sources)}] {pub} ({source.get('_source_type', '?')})")

        tags = extract_tags_from_text(client, artist_name, text, model)
        if tags:
            all_tag_lists.append(tags)
            tag_names = [t.get("tag", "?") for t in tags[:5]]
            print(f"    {len(tags)} tags: {', '.join(tag_names)}...")

        time.sleep(0.3)

    # Aggregate across all sources
    weighted_tags = aggregate_tags(all_tag_lists)

    output = {
        "tags": weighted_tags,
        "meta": {
            "count": len(weighted_tags),
            "sources_checked": len(all_sources),
        }
    }

    out_path = Path(f"data/artists/{artist_id}/tags.json")
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    print(f"\n{len(weighted_tags)} unique tag(s) extracted")
    # Show top tags
    for t in weighted_tags[:10]:
        print(f"  [{t['category']}] {t['tag']} (weight: {t['weight']})")

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract thematic creative-approach tags")
    parser.add_argument("artist_id", help="Artist ID")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                       help=f"Anthropic model (default: {DEFAULT_MODEL})")

    args = parser.parse_args()
    extract_tags(args.artist_id, args.model)
