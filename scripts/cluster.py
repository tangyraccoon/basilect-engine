#!/usr/bin/env python3
"""
Cluster: Philosophy clustering of artists.

Uses agglomerative clustering on embeddings to group artists by creative
philosophy. Automatically selects cluster count based on silhouette score.
Optionally uses Anthropic API to name clusters.

Usage:
    python scripts/cluster.py [--n-clusters auto] [--name-clusters]

Output saved to data/clusters.json
"""

import json
import sys
import argparse
from pathlib import Path
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
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

def cluster_artists(n_clusters=None, name_clusters=False, model=None):
    """
    Cluster artists using agglomerative clustering.

    Args:
        n_clusters: Number of clusters (auto-select if None)
        name_clusters: Use LLM to name clusters (default False)
        model: Sentence transformer model (optional)

    Returns:
        dict with cluster info and saves to data/clusters.json
    """

    # Load embeddings
    embeddings_path = DATA_DIR / "embeddings.npy"
    ids_path = DATA_DIR / "embedding_ids.json"

    if not embeddings_path.exists() or not ids_path.exists():
        raise ValueError("Embeddings not found")

    embeddings = np.load(str(embeddings_path))
    artist_ids = load_json(ids_path)

    # Auto-select n_clusters if not provided
    if n_clusters is None:
        best_score = -1
        best_n = 3
        for n in range(3, 8):
            clustering = AgglomerativeClustering(n_clusters=n, linkage='ward')
            labels = clustering.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_score = score
                best_n = n
        n_clusters = best_n
    else:
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = clustering.fit_predict(embeddings)

    # Recalculate with final n_clusters
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = clustering.fit_predict(embeddings)
    sil_score = silhouette_score(embeddings, labels)

    # Build cluster data
    clusters_data = []

    for cluster_id in range(n_clusters):
        # Get artists in this cluster
        mask = labels == cluster_id
        cluster_indices = np.where(mask)[0]
        cluster_artists = [artist_ids[i] for i in cluster_indices]

        # Find centroid and representative quotes
        cluster_embeddings = embeddings[mask]
        centroid = np.mean(cluster_embeddings, axis=0)

        # Find 2 quotes closest to centroid
        representative_quotes = []
        quote_distances = []

        for idx in cluster_indices:
            artist_id = artist_ids[idx]
            quotes_path = ARTISTS_DIR / artist_id / "quotes.json"

            if quotes_path.exists():
                data = load_json(quotes_path)
                quotes = data.get("quotes", [])

                # Load model for quote embedding if needed
                if model is None and representative_quotes:
                    continue

                if model is None:
                    model = SentenceTransformer("all-MiniLM-L6-v2")

                # Embed quotes
                quote_texts = [q.get("text", "") for q in quotes]
                if quote_texts:
                    quote_embeddings = model.encode(quote_texts, convert_to_numpy=True)

                    # Find closest to centroid
                    distances = np.linalg.norm(quote_embeddings - centroid, axis=1)

                    for dist, quote in zip(distances, quotes):
                        quote_distances.append((dist, quote))

        # Sort by distance and take top 2
        quote_distances.sort(key=lambda x: x[0])
        representative_quotes = [q[1]["text"][:150] + "..." for q in quote_distances[:2]]

        # Optional: name cluster using LLM
        cluster_name = None
        if name_clusters and representative_quotes:
            try:
                import os
                import anthropic

                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if api_key:
                    client = anthropic.Anthropic(api_key=api_key)

                    quotes_str = "\n".join(representative_quotes)

                    message = client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=50,
                        messages=[
                            {
                                "role": "user",
                                "content": f"Name this creative philosophy cluster in 3 words based on these representative quotes:\n\n{quotes_str}"
                            }
                        ]
                    )

                    cluster_name = message.content[0].text if message.content else None

            except Exception:
                pass

        cluster_data = {
            "id": cluster_id,
            "artists": sorted(cluster_artists),
            "representative_quotes": representative_quotes
        }

        if cluster_name:
            cluster_data["name"] = cluster_name

        clusters_data.append(cluster_data)

    result = {
        "n_clusters": n_clusters,
        "silhouette_score": round(float(sil_score), 4),
        "clusters": clusters_data
    }

    # Save to file
    output_path = DATA_DIR / "clusters.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    return result

def main():
    parser = argparse.ArgumentParser(
        description="Cluster artists by creative philosophy"
    )
    parser.add_argument(
        "--n-clusters",
        type=str,
        default="auto",
        help="Number of clusters (or 'auto' for automatic selection)"
    )
    parser.add_argument(
        "--name-clusters",
        action="store_true",
        help="Use LLM to name each cluster"
    )

    args = parser.parse_args()

    try:
        n = None if args.n_clusters == "auto" else int(args.n_clusters)
        result = cluster_artists(n_clusters=n, name_clusters=args.name_clusters)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
