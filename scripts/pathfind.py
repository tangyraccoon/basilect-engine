#!/usr/bin/env python3
"""
Pathfind: Find shortest similarity path between two artists.

Uses Dijkstra's algorithm with weight = (1 - similarity) to find the
shortest chain of high-similarity hops. Also finds an alternative path.

Usage:
    python scripts/pathfind.py artist_a artist_b [--min-score 0.70]

Library usage:
    from scripts.pathfind import find_path
    path = find_path("bill_evans", "yung_lean")
"""

import json
import sys
import argparse
import heapq
from pathlib import Path
import numpy as np

# Get project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def load_json(path):
    """Load JSON file safely"""
    if not path.exists():
        return None
    return json.loads(path.read_text())

def find_path(artist_a_id, artist_b_id, min_score=0.70):
    """
    Find shortest path between two artists using similarity matrix.

    Args:
        artist_a_id: ID of start artist
        artist_b_id: ID of end artist
        min_score: Minimum similarity score for edge to exist (default 0.70)

    Returns:
        dict with primary path, alternative path, and scores
    """

    # Load embedding IDs and similarity matrix
    ids_path = DATA_DIR / "embedding_ids.json"
    sim_path = DATA_DIR / "similarity.npy"

    if not ids_path.exists() or not sim_path.exists():
        raise ValueError("Similarity matrix not found")

    embedding_ids = load_json(ids_path)
    similarity = np.load(str(sim_path))

    if artist_a_id not in embedding_ids or artist_b_id not in embedding_ids:
        raise ValueError(f"One or both artists not in embedding set")

    # Get indices
    start_idx = embedding_ids.index(artist_a_id)
    end_idx = embedding_ids.index(artist_b_id)

    # Direct score
    direct_score = float(similarity[start_idx, end_idx])

    # Build adjacency with weights
    # Weight = 1 - similarity (so higher similarity = lower weight = better)
    def dijkstra(start, end, exclude_edges=None):
        """Dijkstra with optional edge exclusion"""
        exclude_edges = exclude_edges or set()

        # Priority queue: (cost, current_idx, path)
        pq = [(0, start, [start])]
        visited = set()
        distances = {i: float('inf') for i in range(len(embedding_ids))}
        distances[start] = 0
        parent = {}

        while pq:
            cost, current, path = heapq.heappop(pq)

            if current in visited:
                continue

            visited.add(current)

            if current == end:
                return path, cost

            # Check all neighbors with sufficient similarity
            for neighbor in range(len(embedding_ids)):
                if neighbor in visited:
                    continue

                edge_key = tuple(sorted([current, neighbor]))
                if edge_key in exclude_edges:
                    continue

                sim_score = similarity[current, neighbor]
                if sim_score < min_score:
                    continue

                # Weight is inverse of similarity
                weight = 1 - float(sim_score)
                new_cost = cost + weight

                if new_cost < distances[neighbor]:
                    distances[neighbor] = new_cost
                    parent[neighbor] = current
                    heapq.heappush(pq, (new_cost, neighbor, path + [neighbor]))

        return None, float('inf')

    # Find primary path
    primary_path, primary_cost = dijkstra(start_idx, end_idx)

    if primary_path is None:
        # No path found - return direct connection
        return {
            "from": artist_a_id,
            "to": artist_b_id,
            "direct_score": round(direct_score, 4),
            "path": [
                {"artist": artist_a_id, "score_to_next": None},
                {"artist": artist_b_id, "score_to_next": None}
            ],
            "path_min_score": round(direct_score, 4),
            "alternative_path": None
        }

    # Find alternative path by excluding edges from primary path
    exclude_edges = set()
    for i in range(len(primary_path) - 1):
        u, v = primary_path[i], primary_path[i + 1]
        exclude_edges.add(tuple(sorted([u, v])))

    alt_path, alt_cost = dijkstra(start_idx, end_idx, exclude_edges)

    # Build output for primary path
    path_data = []
    path_min_score = 1.0

    for i, idx in enumerate(primary_path):
        artist_id = embedding_ids[idx]
        if i < len(primary_path) - 1:
            next_idx = primary_path[i + 1]
            score_to_next = float(similarity[idx, next_idx])
            path_min_score = min(path_min_score, score_to_next)
        else:
            score_to_next = None

        path_data.append({
            "artist": artist_id,
            "score_to_next": round(score_to_next, 4) if score_to_next else None
        })

    # Build output for alternative path
    alt_path_data = None
    if alt_path and alt_path != primary_path:
        alt_path_data = []
        alt_min_score = 1.0
        for i, idx in enumerate(alt_path):
            artist_id = embedding_ids[idx]
            if i < len(alt_path) - 1:
                next_idx = alt_path[i + 1]
                score_to_next = float(similarity[idx, next_idx])
                alt_min_score = min(alt_min_score, score_to_next)
            else:
                score_to_next = None

            alt_path_data.append({
                "artist": artist_id,
                "score_to_next": round(score_to_next, 4) if score_to_next else None
            })

    return {
        "from": artist_a_id,
        "to": artist_b_id,
        "direct_score": round(direct_score, 4),
        "path": path_data,
        "path_min_score": round(path_min_score, 4),
        "alternative_path": alt_path_data
    }

def main():
    parser = argparse.ArgumentParser(
        description="Find shortest similarity path between two artists"
    )
    parser.add_argument("artist_a", help="ID of start artist")
    parser.add_argument("artist_b", help="ID of end artist")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.70,
        help="Minimum similarity score for path (default 0.70)"
    )

    args = parser.parse_args()

    try:
        result = find_path(args.artist_a, args.artist_b, min_score=args.min_score)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
