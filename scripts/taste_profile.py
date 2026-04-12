#!/usr/bin/env python3
"""
Taste Profile: Creative DNA profiling from artist selection.

Given a list of artists, computes the centroid of their embeddings and
finds similar artists. Optionally uses Anthropic API to explain themes.

Usage:
    python scripts/taste_profile.py "radiohead,animal_collective,tame_impala" [--top 5] [--explain]

Library usage:
    from scripts.taste_profile import compute_profile
    result = compute_profile(["radiohead","animal_collective","tame_impala"], top_k=5)
"""

import json
import sys
import argparse
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

# Get project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTISTS_DIR = DATA_DIR / "artists"

def load_json(path):
    """Load JSON file safely"""
    if not path.exists():
        return None
    return json.loads(path.read_text())

def compute_profile(artist_ids, top_k=5, explain=False, model=None):
    """
    Compute taste profile for a list of artists.

    Args:
        artist_ids: List of artist IDs
        top_k: Number of recommendations (default 5)
        explain: Whether to use LLM to explain themes (default False)
        model: Sentence transformer model (optional)

    Returns:
        dict with input artists, recommendations, and optional explanation
    """

    # Load model if not provided
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load embeddings
    embeddings_path = DATA_DIR / "embeddings.npy"
    ids_path = DATA_DIR / "embedding_ids.json"

    if not embeddings_path.exists() or not ids_path.exists():
        raise ValueError("Embeddings not found")

    embeddings = np.load(str(embeddings_path))
    all_artist_ids = load_json(ids_path)

    # Validate input artists
    for aid in artist_ids:
        if aid not in all_artist_ids:
            raise ValueError(f"Artist not found: {aid}")

    # Get indices of input artists
    indices = [all_artist_ids.index(aid) for aid in artist_ids]

    # Compute centroid
    centroid = np.mean(embeddings[indices], axis=0)

    # Compute similarities to centroid for all other artists
    similarities = np.dot(embeddings, centroid)

    # Find top-K excluding input artists
    recommendations = []
    input_set = set(artist_ids)

    # Get indices sorted by similarity
    sorted_indices = np.argsort(similarities)[::-1]

    for idx in sorted_indices:
        artist_id = all_artist_ids[idx]
        if artist_id not in input_set and len(recommendations) < top_k:
            recommendations.append({
                "artist": artist_id,
                "similarity_to_centroid": round(float(similarities[idx]), 4)
            })

    # Optional: explain themes using Anthropic API
    explanation = None
    if explain:
        try:
            import os
            import anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")

            client = anthropic.Anthropic(api_key=api_key)

            # Load quotes from input artists to inform explanation
            quotes_list = []
            for aid in artist_ids[:3]:  # Use first 3 for brevity
                quotes_path = ARTISTS_DIR / aid / "quotes.json"
                if quotes_path.exists():
                    data = load_json(quotes_path)
                    quotes = data.get("quotes", [])[:2]  # First 2 quotes per artist
                    for q in quotes:
                        quotes_list.append(f"[{aid}] {q.get('text', '')[:200]}")

            quotes_context = "\n".join(quotes_list)

            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=300,
                messages=[
                    {
                        "role": "user",
                        "content": f"These artists share a creative vision: {', '.join(artist_ids)}. Based on their quotes, what philosophical or creative themes unite them?\n\nQuotes:\n{quotes_context}\n\nRespond in 2-3 sentences."
                    }
                ]
            )

            explanation = message.content[0].text if message.content else None

        except Exception as e:
            # Silently fail if API call fails
            explanation = f"Could not generate explanation: {str(e)}"

    result = {
        "input_artists": artist_ids,
        "recommendations": recommendations
    }

    if explain:
        result["explanation"] = explanation

    return result

def main():
    parser = argparse.ArgumentParser(
        description="Generate taste profile for selected artists"
    )
    parser.add_argument(
        "artists",
        help="Comma-separated list of artist IDs (e.g. 'radiohead,animal_collective')"
    )
    parser.add_argument("--top", type=int, default=5, help="Number of recommendations (default 5)")
    parser.add_argument("--explain", action="store_true", help="Use LLM to explain themes")

    args = parser.parse_args()

    try:
        artist_list = [a.strip() for a in args.artists.split(",")]
        result = compute_profile(artist_list, top_k=args.top, explain=args.explain)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
