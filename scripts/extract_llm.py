"""LLM-powered quote extraction using Anthropic API.

Extracts verbatim quotes where the target artist speaks about making music.
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

# Thresholds from extract.py (for corpus validation)
MIN_QUOTES = 5
MIN_SOURCES = 3
MIN_YEARS = 2

# Model to use — Haiku 4.5 is sufficient for structured extraction and ~12-15x cheaper
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# Cost controls — keeps per-artist cost near $0.04
MAX_SOURCES = 4        # Cap sources sent to LLM (prefilter picks best ones first)
MAX_TEXT_CHARS = 6000  # Truncate each article — quotes concentrate in first ~6K chars


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


def extract_quotes_from_text(client: Anthropic, artist_name: str, text: str,
                             publication: str, url: str, date: str,
                             model: str = DEFAULT_MODEL) -> list:
    """Use Claude to extract quotes from a single article."""

    prompt = f"""You are extracting verbatim quotes from a music interview. The target artist is {artist_name}.

Extract every quote where {artist_name} speaks about making music — their creative process, philosophy, influences, artistic decisions, how they approach their craft.

RULES:
- Keep EXACT wording — typos, grammar errors, transcription artifacts. Do not correct anything.
- Strip non-speech insertions: [laughs], [pause], [gestures], etc.
- Preserve ellipses.
- If a quote is interrupted by journalist narration, rejoin the parts.
- Do NOT merge separate quotes from different parts of the article.
- EXCLUDE: interviewer questions, journalist narrative/paraphrase, other speakers, biographical facts without creative content, promotional fluff.
- Only include quotes where you can confirm the target artist is the speaker (via quotation marks with attribution, speaker labels, or Q&A context).

Return a JSON array of objects: [{{"text": "exact quote...", "publication": "{publication}", "date": "{date}"}}]
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
            # Remove opening fence (```json or ```)
            response_text = re.sub(r'^```\w*\n?', '', response_text)
            # Remove closing fence
            response_text = re.sub(r'\n?```$', '', response_text)
            response_text = response_text.strip()

        # Try to parse JSON
        try:
            quotes = json.loads(response_text)
            if not isinstance(quotes, list):
                print(f"  WARNING: Response is not a list, got {type(quotes)}")
                return []
            return quotes
        except json.JSONDecodeError as e:
            print(f"  ERROR: Failed to parse JSON response: {e}")
            print(f"  Response: {response_text[:200]}")
            return []

    except Exception as e:
        print(f"  ERROR: API call failed: {e}")
        return []


def extract(artist_id: str, model: str = DEFAULT_MODEL):
    """Extract quotes for an artist from all sources with text."""

    print(f"Extracting quotes for: {artist_id}")
    sources_path = Path(f"data/artists/{artist_id}/sources.json")

    if not sources_path.exists():
        print(f"Error: Not found: {sources_path}")
        sys.exit(1)

    sources = json.loads(sources_path.read_text())
    sources_with_text = [s for s in sources if s.get("text")]

    if not sources_with_text:
        print("No sources with text found. Run scrape.py first.")
        sys.exit(1)

    # Cap sources to control cost — sort by text length desc (longer = more quotes)
    if len(sources_with_text) > MAX_SOURCES:
        sources_with_text = sorted(sources_with_text, key=lambda s: len(s.get("text", "")), reverse=True)[:MAX_SOURCES]

    print(f"Found {len(sources_with_text)} source(s) with text (capped at {MAX_SOURCES})")

    # Get artist display name
    artist_name = artist_id_to_display_name(artist_id)

    # Initialize Anthropic client
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    client = Anthropic(api_key=api_key)

    all_quotes = []

    for i, source in enumerate(sources_with_text, 1):
        url = source.get("url", "")
        publication = source.get("publication", url)
        date = source.get("date", "")
        text = source.get("text", "")

        print(f"\n[{i}/{len(sources_with_text)}] {publication}")

        if not text:
            print("  SKIP: No text")
            continue

        # Truncate to control input token cost — quotes concentrate near the top
        if len(text) > MAX_TEXT_CHARS:
            text = text[:MAX_TEXT_CHARS]

        # Extract quotes via API
        quotes = extract_quotes_from_text(client, artist_name, text, publication, url, date, model)

        if quotes:
            all_quotes.extend(quotes)
            print(f"  OK: {len(quotes)} quote(s) extracted")
        else:
            print(f"  OK: 0 quotes extracted")

        # Rate limiting (be respectful to the API)
        time.sleep(1)

    # Compute corpus metadata
    quote_count = len(all_quotes)
    source_count = len({q.get("publication", "") for q in all_quotes if q.get("publication")})
    years = sorted({q.get("date", "")[:4] for q in all_quotes if q.get("date") and len(q.get("date", "")) >= 4})
    date_range = [years[0], years[-1]] if years else []

    corpus_valid = (
        quote_count >= MIN_QUOTES
        and source_count >= MIN_SOURCES
        and len(set(years)) >= MIN_YEARS
    )

    output = {
        "quotes": all_quotes,
        "corpus_meta": {
            "quote_count": quote_count,
            "source_count": source_count,
            "date_range": date_range,
            "corpus_valid": corpus_valid,
        },
    }

    out_path = Path(f"data/artists/{artist_id}/quotes.json")
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    valid_str = "VALID" if corpus_valid else "INVALID"
    print(f"\n{quote_count} quotes from {source_count} source(s) ({valid_str})")
    if not corpus_valid:
        print(f"  Needs: ≥{MIN_QUOTES} quotes, ≥{MIN_SOURCES} sources, ≥{MIN_YEARS} years")
        print(f"  Has: {quote_count} quotes, {source_count} sources, {len(set(years))} years")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM-powered quote extraction")
    parser.add_argument("artist_id", help="Artist ID (e.g. bjork, aphex_twin)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                       help=f"Anthropic model to use (default: {DEFAULT_MODEL})")

    args = parser.parse_args()
    extract(args.artist_id, args.model)
