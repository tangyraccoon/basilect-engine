#!/usr/bin/env python3
"""
Explain Connection: Find quote pairs that explain artist similarity.

For two artists, finds the closest quote pairs that explain their connection.

Usage:
    python scripts/explain.py artist_a artist_b [--top 5]

Library usage:
    from scripts.explain import explain_connection
    pairs = explain_connection("animal_collective", "badbadnotgood", top_k=5)
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

def explain_connection(artist_a_id, artist_b_id, top_k=5, model=None):
    """
    Find the top-K quote pairs that explain similarity between two artists.

    Args:
        artist_a_id: ID of first artist
        artist_b_id: ID of second artist
        top_k: Number of quote pairs to return (default 5)
        model: Sentence transformer model (optional, creates if None)

    Returns:
        dict with artists, overall_score, and quote_pairs
    """

    # Load model if not provided
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load quotes for both artists
    quotes_a_path = ARTISTS_DIR / artist_a_id / "quotes.json"
    quotes_b_path = ARTISTS_DIR / artist_b_id / "quotes.json"

    if not quotes_a_path.exists() or not quotes_b_path.exists():
        raise ValueError(f"One or both artists not found: {artist_a_id}, {artist_b_id}")

    quotes_a_data = load_json(quotes_a_path)
    quotes_b_data = load_json(quotes_b_path)

    quotes_a = quotes_a_data.get("quotes", []) if quotes_a_data else []
    quotes_b = quotes_b_data.get("quotes", []) if quotes_b_data else []

    if not quotes_a or not quotes_b:
        raise ValueError(f"No quotes found for one or both artists")

    # Extract just text for embedding
    texts_a = [q.get("text", "") for q in quotes_a]
    texts_b = [q.get("text", "") for q in quotes_b]

    # Embed all quotes
    embeddings_a = model.encode(texts_a, convert_to_numpy=True)
    embeddings_b = model.encode(texts_b, convert_to_numpy=True)

    # Compute cross-artist similarity (cosine)
    # Shape: (len(quotes_a), len(quotes_b))
    similarities = np.dot(embeddings_a, embeddings_b.T)

    # Find top-K pairs
    # Flatten and get indices of top-K
    flat_indices = np.argsort(similarities.flatten())[::-1][:top_k]

    quote_pairs = []
    overall_scores = []

    for idx in flat_indices:
        i = idx // len(quotes_b)  # Index in quotes_a
        j = idx % len(quotes_b)   # Index in quotes_b

        sim_score = float(similarities[i, j])
        overall_scores.append(sim_score)

        quote_pairs.append({
            "quote_a": quotes_a[i],
            "quote_b": quotes_b[j],
            "similarity": round(sim_score, 4)
        })

    # Compute overall score as mean of top pairs
    overall_score = float(np.mean(overall_scores)) if overall_scores else 0.0

    return {
        "a": artist_a_id,
        "b": artist_b_id,
        "overall_score": round(overall_score, 4),
        "quote_pairs": quote_pairs
    }

def main():
    parser = argparse.ArgumentParser(
        description="Find quote pairs that explain artist similarity"
    )
    parser.add_argument("artist_a", help="ID of first artist")
    parser.add_argument("artist_b", help="ID of second artist")
    parser.add_argument("--top", type=int, default=5, help="Number of pairs to return (default 5)")

    args = parser.parse_args()

    try:
        result = explain_connection(args.artist_a, args.artist_b, top_k=args.top)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
