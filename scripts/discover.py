"""Surface ranked artist pairs by quote similarity."""

import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    ids = json.loads((DATA_DIR / "embedding_ids.json").read_text(encoding="utf-8"))
    sim = np.load(DATA_DIR / "similarity.npy")

    n = len(ids)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({"a": ids[i], "b": ids[j], "score": float(sim[i, j])})

    pairs.sort(key=lambda p: p["score"], reverse=True)

    print(f"\n{'=' * 50}")
    print(f"  ARTIST SIMILARITY RANKING ({len(pairs)} pairs)")
    print(f"{'=' * 50}")
    for rank, p in enumerate(pairs, 1):
        print(f"  {rank:2d}. {p['a']} × {p['b']}  —  {p['score']:.3f}")

    out_path = DATA_DIR / "discoveries.json"
    out_path.write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
