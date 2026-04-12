"""Compute pairwise cosine similarity from critic embeddings.

Mirrors compute.py but works with critic_embeddings.npy.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    embeddings_path = DATA_DIR / "critic_embeddings.npy"
    ids_path = DATA_DIR / "critic_embedding_ids.json"

    if not embeddings_path.exists() or not ids_path.exists():
        print(f"Error: critic embeddings not found. Run embed_critics.py first.")
        return

    embeddings = np.load(embeddings_path)
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(ids)} critic embeddings ({embeddings.shape})")

    sim = cosine_similarity(embeddings)
    np.save(DATA_DIR / "critic_similarity.npy", sim)
    print(f"Saved critic similarity matrix: {sim.shape}")

    n = len(ids)
    upper = np.triu_indices(n, k=1)
    scores = sim[upper]
    print(f"\n{len(scores)} pairs")
    print(f"Similarity — mean: {scores.mean():.3f}, std: {scores.std():.3f}, "
          f"min: {scores.min():.3f}, max: {scores.max():.3f}")


if __name__ == "__main__":
    main()
