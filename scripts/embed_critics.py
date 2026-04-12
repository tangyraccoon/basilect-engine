"""Embed critic discourse using sentence-transformers.

Each artist's critic passages are embedded individually, then aggregated
into a single artist-level vector via component-wise median.
Mirrors embed.py but works with critic_quotes.json files.
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ARTISTS_DIR = DATA_DIR / "artists"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_critic_quotes():
    """Load critic quotes from all artist directories."""
    ids = []
    quotes_per_artist = []
    for path in sorted(ARTISTS_DIR.glob("*/critic_quotes.json")):
        artist_id = path.parent.name
        try:
            node = json.loads(path.read_text(encoding="utf-8"))
            texts = [q["text"] for q in node["quotes"]]
            ids.append(artist_id)
            quotes_per_artist.append(texts)
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")
    return ids, quotes_per_artist


def main():
    ids, quotes_per_artist = load_critic_quotes()
    total = sum(len(q) for q in quotes_per_artist)
    print(f"Loaded {len(ids)} artists, {total} total critic passages")

    if not ids:
        print("No critic quotes found. Run extract_critic.py for some artists first.")
        return

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    all_quotes = []
    boundaries = []
    offset = 0
    for quotes in quotes_per_artist:
        all_quotes.extend(quotes)
        boundaries.append((offset, offset + len(quotes)))
        offset += len(quotes)

    print("Encoding critic passages...")
    all_embeddings = model.encode(all_quotes, show_progress_bar=True)

    dim = all_embeddings.shape[1] if len(all_embeddings) > 0 else 384
    artist_embeddings = []
    for i, (start, end) in enumerate(boundaries):
        vecs = all_embeddings[start:end]
        if len(vecs) == 0:
            median_vec = np.zeros(dim)
        else:
            median_vec = np.median(vecs, axis=0)
        artist_embeddings.append(median_vec)
        print(f"  {ids[i]}: {end - start} passages -> median vector")

    embeddings = np.array(artist_embeddings)

    np.save(DATA_DIR / "critic_embeddings.npy", embeddings)
    (DATA_DIR / "critic_embedding_ids.json").write_text(
        json.dumps(ids, indent=2), encoding="utf-8"
    )

    print(f"\nSaved critic embeddings: {embeddings.shape} to data/critic_embeddings.npy")
    print(f"Saved artist order: {len(ids)} IDs to data/critic_embedding_ids.json")


if __name__ == "__main__":
    main()
