"""LLM-powered critic discourse extraction using Anthropic API.

Extracts passages where CRITICS describe an artist's creative approach, philosophy,
and artistic identity. This is the critic's voice about the artist — not quotes from the artist.
"""

import sys
import json
import re
import os
import time
from pathlib import Path
from typing import Optional

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

# Thresholds for corpus validation
MIN_QUOTES = 5
MIN_SOURCES = 2
MIN_YEARS = 1

# Model to use — Haiku 4.5 is sufficient for structured extraction and ~12-15x cheaper
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# Cost controls
MAX_SOURCES = 4
MAX_TEXT_CHARS = 6000


def normalize_artist_id(name: str) -> str:
    """Normalize artist name to ID: lowercase, spaces to underscores, strip punctuation."""
    id_str = name.lower().strip()
    id_str = re.sub(r'[^\w\s\-]', '', id_str)
    id_str = re.sub(r'\s+', '_', id_str)
    id_str = re.sub(r'_+', '_', id_str)
    return id_str.strip('_')


def artist_id_to_display_name(artist_id: str) -> str:
    """Convert artist_id back to display name (underscores to spaces, title case)."""
    return artist_id.replace('_', ' ').title()


def extract_critic_discourse_from_text(client: Anthropic, artist_name: str, text: str,
                                       publication: str, url: str, date: str,
                                       model: str = DEFAULT_MODEL) -> list:
    """Use Claude to extract critic discourse about an artist's creative approach."""

    prompt = f"""You are extracting passages from a music review/feature about {artist_name}.

Extract every passage where the critic describes {artist_name}'s creative approach, artistic philosophy, musical identity, or how they make music. These are the CRITIC's words about the artist's creative world — not quotes from the artist themselves.

Look for passages about:
- How the artist approaches songwriting, production, or performance
- Their artistic evolution or creative trajectory
- Their relationship to genre, tradition, or innovation
- What makes their creative process distinctive
- Their aesthetic sensibility or artistic vision
- Their influence on others or position in music history
- Technical or compositional choices

RULES:
- Keep the critic's exact wording — typos and all
- Each passage should be 1-3 sentences, self-contained
- EXCLUDE: track-by-track reviews, star ratings, biographical facts without creative insight, promotional language, pure comparisons to other artists, listener experience descriptions that don't illuminate the artist's creative approach
- INCLUDE: surrounding context if it clarifies the creative meaning
- The critic should be describing THE ARTIST'S CREATIVE WORLD, not the listener's experience

Return a JSON array: [{{"text": "passage...", "publication": "{publication}", "date": "{date}"}}]
Return ONLY the JSON array, no other text.

TEXT TO EXTRACT FROM:
{text}"""

    try:
        message = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        response_text = message.content[0].text

        # Strip markdown code fences if present (Haiku tends to add these)
        response_text = response_text.strip()
        if response_text.startswith("```"):
            response_text = re.sub(r'^```\w*\n?', '', response_text)
            response_text = re.sub(r'\n?```$', '', response_text)
            response_text = response_text.strip()

        # Try to parse JSON
        try:
            passages = json.loads(response_text)
            if not isinstance(passages, list):
                print(f"  WARNING: Response is not a list, got {type(passages)}")
                return []
            return passages
        except json.JSONDecodeError as e:
            print(f"  ERROR: Failed to parse JSON response: {e}")
            print(f"  Response: {response_text[:200]}")
            return []

    except Exception as e:
        print(f"  ERROR: API call failed: {e}")
        return []


def extract_critic(artist_id: str, model: str = DEFAULT_MODEL):
    """Extract critic discourse for an artist from all review sources with text."""

    print(f"Extracting critic discourse for: {artist_id}")
    review_sources_path = Path(f"data/artists/{artist_id}/review_sources.json")

    if not review_sources_path.exists():
        print(f"Error: Not found: {review_sources_path}")
        sys.exit(1)

    review_sources = json.loads(review_sources_path.read_text())
    sources_with_text = [s for s in review_sources if s.get("text")]

    if not sources_with_text:
        print("No review sources with text found. Run scrape_reviews.py first.")
        sys.exit(1)

    # Cap sources to control cost
    if len(sources_with_text) > MAX_SOURCES:
        sources_with_text = sorted(sources_with_text, key=lambda s: len(s.get("text", "")), reverse=True)[:MAX_SOURCES]

    print(f"Found {len(sources_with_text)} review source(s) with text (capped at {MAX_SOURCES})")

    # Get artist display name
    artist_name = artist_id_to_display_name(artist_id)

    # Initialize Anthropic client
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    client = Anthropic(api_key=api_key)

    all_passages = []

    for i, source in enumerate(sources_with_text, 1):
        url = source.get("url", "")
        publication = source.get("publication", url)
        date = source.get("date", "")
        text = source.get("text", "")

        print(f"\n[{i}/{len(sources_with_text)}] {publication}")

        if not text:
            print("  SKIP: No text")
            continue

        # Truncate to control input token cost
        if len(text) > MAX_TEXT_CHARS:
            text = text[:MAX_TEXT_CHARS]

        # Extract passages via API
        passages = extract_critic_discourse_from_text(client, artist_name, text,
                                                     publication, url, date, model)

        if passages:
            all_passages.extend(passages)
            print(f"  OK: {len(passages)} passage(s) extracted")
        else:
            print(f"  OK: 0 passages extracted")

        # Rate limiting
        time.sleep(1)

    # Compute corpus metadata
    passage_count = len(all_passages)
    source_count = len({p.get("publication", "") for p in all_passages if p.get("publication")})
    years = sorted({p.get("date", "")[:4] for p in all_passages if p.get("date") and len(p.get("date", "")) >= 4})
    date_range = [years[0], years[-1]] if years else []

    corpus_valid = (
        passage_count >= MIN_QUOTES
        and source_count >= MIN_SOURCES
        and len(set(years)) >= MIN_YEARS
    )

    output = {
        "quotes": all_passages,
        "corpus_meta": {
            "quote_count": passage_count,
            "source_count": source_count,
            "date_range": date_range,
            "corpus_valid": corpus_valid,
        },
    }

    out_path = Path(f"data/artists/{artist_id}/critic_quotes.json")
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    valid_str = "VALID" if corpus_valid else "INVALID"
    print(f"\n{passage_count} passages from {source_count} source(s) ({valid_str})")
    if not corpus_valid:
        print(f"  Needs: ≥{MIN_QUOTES} passages, ≥{MIN_SOURCES} sources, ≥{MIN_YEARS} years")
        print(f"  Has: {passage_count} passages, {source_count} sources, {len(set(years))} years")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM-powered critic discourse extraction")
    parser.add_argument("artist_id", help="Artist ID (e.g. bjork, aphex_twin)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                       help=f"Anthropic model to use (default: {DEFAULT_MODEL})")

    args = parser.parse_args()
    extract_critic(args.artist_id, args.model)
