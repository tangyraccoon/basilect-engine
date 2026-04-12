"""Extract influence citations from interview text.

From already-scraped interview text, extracts statements like:
"I was inspired by X", "I grew up listening to Y", "Z changed how I think about music."

No additional scraping needed — reuses sources.json text.
Output: data/artists/{artist_id}/influences.json
"""

import sys
import json
import os
import time
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def normalize_artist_id(name: str) -> str:
    import re
    id_str = name.lower().strip()
    id_str = re.sub(r'[^\w\s\-]', '', id_str)
    id_str = re.sub(r'\s+', '_', id_str)
    id_str = re.sub(r'_+', '_', id_str)
    return id_str.strip('_')


def artist_id_to_display_name(artist_id: str) -> str:
    return artist_id.replace('_', ' ').title()


def extract_influences_from_text(client: Anthropic, artist_name: str, text: str,
                                  model: str = DEFAULT_MODEL) -> list:
    """Extract influence citations from a single article."""

    prompt = f"""You are extracting influence citations from a music interview with {artist_name}.

Find every statement where {artist_name} mentions being influenced by, inspired by, or learning from another artist, musician, composer, producer, or creative figure.

Look for patterns like:
- "I was inspired by X"
- "I grew up listening to Y"
- "Z changed how I think about music"
- "I learned a lot from watching W perform"
- "My biggest influence was V"
- "I was really into [artist] when I started"

RULES:
- Only extract influences where {artist_name} is the one being influenced (direction = "influenced_by")
- The influencer must be a specific named person or group, not a genre or abstract concept
- Include the surrounding context/quote where the influence is mentioned
- Each influence should have the influencer's name and the verbatim quote context
- Do NOT include: genre influences without a specific artist, self-references, vague mentions

Return a JSON array: [{{"name": "Influencer Name", "context": "verbatim quote mentioning the influence", "direction": "influenced_by"}}]
Return ONLY the JSON array, no other text. Return [] if no influence citations found.

TEXT:
{text}"""

    try:
        message = client.messages.create(
            model=model,
            max_tokens=2048,
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
            influences = json.loads(response_text)
            if not isinstance(influences, list):
                return []
            return influences
        except json.JSONDecodeError:
            return []
    except Exception as e:
        print(f"  ERROR: API call failed: {e}")
        return []


def extract_influences(artist_id: str, model: str = DEFAULT_MODEL):
    """Extract influence citations for an artist from all interview sources."""

    print(f"Extracting influences for: {artist_id}")
    sources_path = Path(f"data/artists/{artist_id}/sources.json")

    if not sources_path.exists():
        print(f"No sources.json found for {artist_id}")
        return {"influences": [], "meta": {"count": 0}}

    sources = json.loads(sources_path.read_text())
    sources_with_text = [s for s in sources if s.get("text")]

    if not sources_with_text:
        print("No sources with text found.")
        return {"influences": [], "meta": {"count": 0}}

    print(f"Found {len(sources_with_text)} source(s) with text")

    artist_name = artist_id_to_display_name(artist_id)
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = Anthropic(api_key=api_key)
    all_influences = []
    seen_names = set()

    for i, source in enumerate(sources_with_text, 1):
        text = source.get("text", "")
        print(f"  [{i}/{len(sources_with_text)}] {source.get('publication', 'unknown')}")

        if not text:
            continue

        influences = extract_influences_from_text(client, artist_name, text, model)

        for inf in influences:
            name = inf.get("name", "").strip()
            if name and name.lower() not in seen_names:
                seen_names.add(name.lower())
                all_influences.append(inf)
                print(f"    + {name}")
            elif name:
                # Duplicate, but might have different context — skip for dedup
                pass

        time.sleep(0.5)

    output = {
        "influences": all_influences,
        "meta": {
            "count": len(all_influences),
            "sources_checked": len(sources_with_text),
        }
    }

    out_path = Path(f"data/artists/{artist_id}/influences.json")
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    print(f"\n{len(all_influences)} unique influence(s) extracted")
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract influence citations from interview text")
    parser.add_argument("artist_id", help="Artist ID")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                       help=f"Anthropic model (default: {DEFAULT_MODEL})")

    args = parser.parse_args()
    extract_influences(args.artist_id, args.model)
