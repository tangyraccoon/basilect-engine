"""Compare quote vs critic similarity signals for artists.

For artists that have BOTH interview quote embeddings and critic discourse embeddings,
compute how similarly they are rated by each signal. Ranks by divergence (delta).
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import pearsonr, spearmanr

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_embeddings(embedding_type: str = "quote") -> tuple:
    """Load embeddings and IDs. Returns (ids, embeddings) or ([], None) if not found."""
    if embedding_type == "quote":
        ids_path = DATA_DIR / "embedding_ids.json"
        emb_path = DATA_DIR / "embeddings.npy"
    elif embedding_type == "critic":
        ids_path = DATA_DIR / "critic_embedding_ids.json"
        emb_path = DATA_DIR / "critic_embeddings.npy"
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")

    if not ids_path.exists() or not emb_path.exists():
        return [], None

    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    embeddings = np.load(emb_path)
    return ids, embeddings


def load_similarity(similarity_type: str = "quote") -> Optional[np.ndarray]:
    """Load similarity matrix."""
    if similarity_type == "quote":
        sim_path = DATA_DIR / "similarity.npy"
    elif similarity_type == "critic":
        sim_path = DATA_DIR / "critic_similarity.npy"
    else:
        raise ValueError(f"Unknown similarity type: {similarity_type}")

    if not sim_path.exists():
        return None

    return np.load(sim_path)


def main():
    print("Loading similarity signals...")

    # Load quote signal
    quote_ids, _ = load_embeddings("quote")
    quote_sim = load_similarity("quote")

    # Load critic signal
    critic_ids, _ = load_embeddings("critic")
    critic_sim = load_similarity("critic")

    if quote_sim is None:
        print("Error: Quote similarity not found. Run embed.py and compute.py first.")
        return

    if critic_sim is None:
        print("Error: Critic similarity not found. Run embed_critics.py and compute_critics.py first.")
        return

    print(f"Quote signal: {len(quote_ids)} artists")
    print(f"Critic signal: {len(critic_ids)} artists")

    # Find artists present in both
    quote_set = set(quote_ids)
    critic_set = set(critic_ids)
    common_artists = quote_set & critic_set

    if not common_artists:
        print("Error: No artists present in both signals.")
        return

    print(f"Common artists: {len(common_artists)}")

    # Build mapping from artist ID to index for each signal
    quote_id_to_idx = {aid: i for i, aid in enumerate(quote_ids)}
    critic_id_to_idx = {aid: i for i, aid in enumerate(critic_ids)}

    # Collect all pairs that exist in both signals
    pairs = []

    for i, aid_a in enumerate(quote_ids):
        for j in range(i + 1, len(quote_ids)):
            aid_b = quote_ids[j]

            # Check if both artists are in both signals
            if aid_a not in common_artists or aid_b not in common_artists:
                continue

            # Get quote similarity
            quote_score = float(quote_sim[i, j])

            # Get critic similarity
            critic_i = critic_id_to_idx[aid_a]
            critic_j = critic_id_to_idx[aid_b]
            critic_score = float(critic_sim[critic_i, critic_j])

            # Compute delta
            delta = quote_score - critic_score

            pairs.append({
                "a": aid_a,
                "b": aid_b,
                "quote_sim": quote_score,
                "critic_sim": critic_score,
                "delta": delta,
            })

    if not pairs:
        print("Error: No pairs found in both signals.")
        return

    print(f"Pairs in both signals: {len(pairs)}")

    # Compute correlation
    quote_scores = np.array([p["quote_sim"] for p in pairs])
    critic_scores = np.array([p["critic_sim"] for p in pairs])

    try:
        pearson_r, pearson_p = pearsonr(quote_scores, critic_scores)
    except Exception as e:
        print(f"Warning: Could not compute Pearson correlation: {e}")
        pearson_r = None
        pearson_p = None

    try:
        spearman_r, spearman_p = spearmanr(quote_scores, critic_scores)
    except Exception as e:
        print(f"Warning: Could not compute Spearman correlation: {e}")
        spearman_r = None
        spearman_p = None

    deltas = np.array([p["delta"] for p in pairs])
    mean_delta = float(np.mean(deltas))
    std_delta = float(np.std(deltas))

    # Sort by absolute delta (most interesting divergences)
    pairs.sort(key=lambda p: abs(p["delta"]), reverse=True)

    # Build output
    output = {
        "pairs": pairs,
        "stats": {
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "n_pairs": len(pairs),
            "mean_delta": mean_delta,
            "std_delta": std_delta,
        }
    }

    # Save
    out_path = DATA_DIR / "signal_comparison.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"  SIGNAL COMPARISON: QUOTES vs CRITICS")
    print(f"{'=' * 70}")
    print(f"Pearson correlation: r={pearson_r:.3f}" if pearson_r is not None else "Pearson: N/A")
    print(f"Spearman correlation: rho={spearman_r:.3f}" if spearman_r is not None else "Spearman: N/A")
    print(f"Mean delta (quote - critic): {mean_delta:.3f}")
    print(f"Std delta: {std_delta:.3f}")

    print(f"\nTop 10 divergences (positive = artist alignment stronger than critics see):")
    for rank, p in enumerate(pairs[:10], 1):
        print(f"  {rank:2d}. {p['a']} × {p['b']}")
        print(f"      Quote: {p['quote_sim']:.3f}, Critic: {p['critic_sim']:.3f}, Delta: {p['delta']:+.3f}")

    print(f"\nTop 10 convergences (negative = critics see alignment artists don't express):")
    for rank, p in enumerate(pairs[-10:], 1):
        print(f"  {rank:2d}. {p['a']} × {p['b']}")
        print(f"      Quote: {p['quote_sim']:.3f}, Critic: {p['critic_sim']:.3f}, Delta: {p['delta']:+.3f}")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
