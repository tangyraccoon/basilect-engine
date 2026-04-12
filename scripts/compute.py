"""Compute pairwise cosine similarity from artist embeddings."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    embeddings = np.load(DATA_DIR / "embeddings.npy")
    ids = json.loads((DATA_DIR / "embedding_ids.json").read_text(encoding="utf-8"))
    print(f"Loaded {len(ids)} embeddings ({embeddings.shape})")

    sim = cosine_similarity(embeddings)
    np.save(DATA_DIR / "similarity.npy", sim)
    print(f"Saved similarity matrix: {sim.shape}")

    n = len(ids)
    upper = np.triu_indices(n, k=1)
    scores = sim[upper]
    print(f"\n{len(scores)} pairs")
    print(f"Similarity — mean: {scores.mean():.3f}, std: {scores.std():.3f}, "
          f"min: {scores.min():.3f}, max: {scores.max():.3f}")


if __name__ == "__main__":
    main()
