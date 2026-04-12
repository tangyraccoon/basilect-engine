#!/usr/bin/env python3
"""
Basilect Engine Web App — Flask Backend
Enhanced with interactive feature engines.
"""

import json
import subprocess
import os
import sys
import threading
import uuid
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import re

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Get project root (parent of app/) and ensure scripts/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "data"
ARTISTS_DIR = DATA_DIR / "artists"

# Simple in-memory job tracking
jobs = {}

# Lazy model loader for sentence transformers
_model = None

def get_model():
    """Lazy load sentence transformer model"""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def normalize_artist_id(name):
    """Normalize artist name to artist_id format"""
    normalized = name.lower().strip()
    normalized = re.sub(r'[^\w\s_-]', '', normalized)  # strip punctuation
    normalized = re.sub(r'[\s-]+', '_', normalized)     # spaces/hyphens to underscores
    return normalized

def run_job(job_id, cmd, cwd=None):
    """Run subprocess and track status"""
    if cwd is None:
        cwd = str(PROJECT_ROOT)

    try:
        jobs[job_id] = {
            "status": "running",
            "output": "",
            "error": None,
            "created_at": datetime.now().isoformat()
        }

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=600  # 10 minute timeout
        )

        jobs[job_id]["status"] = "completed" if result.returncode == 0 else "failed"
        jobs[job_id]["output"] = result.stdout
        if result.stderr:
            jobs[job_id]["error"] = result.stderr
        jobs[job_id]["completed_at"] = datetime.now().isoformat()

    except subprocess.TimeoutExpired:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = "Command timed out (10+ minutes)"
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["completed_at"] = datetime.now().isoformat()

def load_json(path):
    """Load JSON, return empty dict/list if not found"""
    if not path.exists():
        return None
    return json.loads(path.read_text())

def get_artist_corpus_meta(artist_id):
    """Load corpus_meta from quotes.json"""
    quotes_path = ARTISTS_DIR / artist_id / "quotes.json"
    if not quotes_path.exists():
        return None
    data = load_json(quotes_path)
    return data.get("corpus_meta") if data else None

def get_artist_critic_corpus_meta(artist_id):
    """Load corpus_meta from critic_quotes.json"""
    path = ARTISTS_DIR / artist_id / "critic_quotes.json"
    if not path.exists():
        return None
    data = load_json(path)
    return data.get("corpus_meta") if data else None

def to_display_name(artist_id):
    """Convert snake_case artist_id to Title Case display name"""
    return " ".join(w.capitalize() for w in artist_id.split("_"))

# ===== Serve Frontend =====

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files (fallback to index.html for SPA routing)"""
    file_path = Path(app.static_folder) / path
    if file_path.exists() and file_path.is_file():
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

# ===== API: Discoveries & Matrix =====

@app.route('/api/discoveries', methods=['GET'])
def get_discoveries():
    """Return ranked artist pairs. Supports ?limit=N (default 200) and ?offset=N for pagination."""
    path = DATA_DIR / "discoveries.json"
    if not path.exists():
        return jsonify([])
    data = load_json(path) or []
    limit = request.args.get('limit', 200, type=int)
    offset = request.args.get('offset', 0, type=int)
    total = len(data)
    page = data[offset:offset + limit]
    response = jsonify(page)
    response.headers['X-Total-Count'] = total
    return response

@app.route('/api/matrix', methods=['GET'])
def get_matrix():
    """Return similarity matrix as JSON"""
    sim_path = DATA_DIR / "similarity.npy"
    ids_path = DATA_DIR / "embedding_ids.json"

    if not sim_path.exists() or not ids_path.exists():
        return jsonify({"error": "Matrix not yet computed"}), 404

    try:
        # Load similarity matrix and IDs
        similarity = np.load(str(sim_path))
        embedding_ids = load_json(ids_path)

        # Convert to list format for JSON
        matrix = {
            "ids": embedding_ids,
            "scores": similarity.tolist()
        }
        return jsonify(matrix)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/blocked-domains', methods=['GET'])
def get_blocked_domains():
    """Return blocked domains markdown"""
    path = DATA_DIR / "blocked_domains.md"
    if not path.exists():
        return jsonify({"content": ""})
    return jsonify({"content": path.read_text()})

# ===== API: Artists =====

@app.route('/api/artists', methods=['GET'])
def list_artists():
    """Return all artists with corpus_meta"""
    if not ARTISTS_DIR.exists():
        return jsonify([])

    artists = []
    for artist_dir in sorted(ARTISTS_DIR.iterdir()):
        if artist_dir.is_dir():
            artist_id = artist_dir.name
            state_path = artist_dir / "state.json"
            if not state_path.exists():
                continue  # never processed at all
            corpus_meta = get_artist_corpus_meta(artist_id)
            critic_meta = get_artist_critic_corpus_meta(artist_id)
            quote_valid = bool(corpus_meta and corpus_meta.get("corpus_valid"))
            critic_valid = bool(critic_meta and critic_meta.get("corpus_valid"))
            artists.append({
                "id": artist_id,
                "name": to_display_name(artist_id),
                "corpus_meta": corpus_meta,
                "critic_corpus_meta": critic_meta,
                "corpus_valid": quote_valid,
                "critic_valid": critic_valid,
            })
    return jsonify(artists)

@app.route('/api/artists/<artist_id>', methods=['GET'])
def get_artist(artist_id):
    """Return full artist data (sources + quotes)"""
    artist_dir = ARTISTS_DIR / artist_id
    if not artist_dir.exists():
        return jsonify({"error": "Artist not found"}), 404

    sources = load_json(artist_dir / "sources.json") or []
    quotes_data = load_json(artist_dir / "quotes.json") or {}
    critic_data = load_json(artist_dir / "critic_quotes.json") or {}
    review_sources = load_json(artist_dir / "review_sources.json") or []

    return jsonify({
        "id": artist_id,
        "name": to_display_name(artist_id),
        "sources": sources,
        "quotes": quotes_data.get("quotes", []),
        "corpus_meta": quotes_data.get("corpus_meta"),
        "critic_quotes": critic_data.get("quotes", []),
        "critic_corpus_meta": critic_data.get("corpus_meta"),
        "review_sources": review_sources,
    })

@app.route('/api/artists/<artist_id>/sources', methods=['GET'])
def get_artist_sources(artist_id):
    """Return sources.json"""
    sources_path = ARTISTS_DIR / artist_id / "sources.json"
    if not sources_path.exists():
        return jsonify([])
    return jsonify(load_json(sources_path) or [])

@app.route('/api/artists/<artist_id>/quotes', methods=['GET'])
def get_artist_quotes(artist_id):
    """Return quotes.json"""
    quotes_path = ARTISTS_DIR / artist_id / "quotes.json"
    if not quotes_path.exists():
        return jsonify({"quotes": [], "corpus_meta": None})
    data = load_json(quotes_path) or {}
    return jsonify({
        "quotes": data.get("quotes", []),
        "corpus_meta": data.get("corpus_meta")
    })

@app.route('/api/artists/<artist_id>/tags', methods=['GET'])
def get_artist_tags(artist_id):
    """Return genre/scene tags for an artist"""
    tags_path = ARTISTS_DIR / artist_id / "tags.json"
    if not tags_path.exists():
        return jsonify({"genres": [], "scenes": []})
    try:
        data = load_json(tags_path) or {}
        return jsonify({
            "genres": data.get("genres", []),
            "scenes": data.get("scenes", [])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/artists/<artist_id>/tags', methods=['POST'])
def set_artist_tags(artist_id):
    """Set genre/scene tags for an artist"""
    artist_dir = ARTISTS_DIR / artist_id
    if not artist_dir.exists():
        return jsonify({"error": "Artist not found"}), 404

    data = request.get_json() or {}
    genres = data.get("genres", [])
    scenes = data.get("scenes", [])

    if not isinstance(genres, list) or not isinstance(scenes, list):
        return jsonify({"error": "genres and scenes must be lists"}), 400

    try:
        tags_path = artist_dir / "tags.json"
        tags_data = {
            "genres": genres,
            "scenes": scenes
        }
        tags_path.write_text(json.dumps(tags_data, indent=2))
        return jsonify(tags_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/artists', methods=['POST'])
def create_artist():
    """Create new artist directory + empty sources.json"""
    data = request.get_json() or {}
    name = data.get("name")

    if not name:
        return jsonify({"error": "Missing 'name' field"}), 400

    artist_id = normalize_artist_id(name)
    artist_dir = ARTISTS_DIR / artist_id

    if artist_dir.exists():
        return jsonify({"error": "Artist already exists"}), 409

    try:
        artist_dir.mkdir(parents=True, exist_ok=True)
        sources_path = artist_dir / "sources.json"
        sources_path.write_text(json.dumps([], indent=2))

        return jsonify({
            "id": artist_id,
            "name": name,
            "sources": [],
            "quotes": [],
            "corpus_meta": None
        }), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/artists/<artist_id>/sources', methods=['POST'])
def add_sources(artist_id):
    """Append sources to sources.json (skip duplicates by URL)"""
    sources_path = ARTISTS_DIR / artist_id / "sources.json"
    if not sources_path.exists():
        return jsonify({"error": "Artist not found"}), 404

    data = request.get_json() or {}
    new_sources = data.get("sources", [])

    if not isinstance(new_sources, list):
        return jsonify({"error": "sources must be a list"}), 400

    try:
        existing = load_json(sources_path) or []
        existing_urls = {s.get("url") for s in existing}

        for src in new_sources:
            if src.get("url") not in existing_urls:
                existing.append(src)
                existing_urls.add(src.get("url"))

        sources_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
        return jsonify({"count": len(existing), "sources": existing})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/artists/<artist_id>/sources', methods=['DELETE'])
def delete_source(artist_id):
    """Remove a source by URL"""
    sources_path = ARTISTS_DIR / artist_id / "sources.json"
    if not sources_path.exists():
        return jsonify({"error": "Artist not found"}), 404

    data = request.get_json() or {}
    url = data.get("url")

    if not url:
        return jsonify({"error": "Missing 'url' field"}), 400

    try:
        sources = load_json(sources_path) or []
        sources = [s for s in sources if s.get("url") != url]
        sources_path.write_text(json.dumps(sources, indent=2, ensure_ascii=False))
        return jsonify({"count": len(sources), "sources": sources})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/artists/<artist_id>', methods=['DELETE'])
def delete_artist(artist_id):
    """Delete entire artist directory"""
    artist_dir = ARTISTS_DIR / artist_id
    if not artist_dir.exists():
        return jsonify({"error": "Artist not found"}), 404

    try:
        import shutil
        shutil.rmtree(artist_dir)
        return jsonify({"message": "Artist deleted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===== API: Feature Engines =====

@app.route('/api/explain/<artist_a>/<artist_b>', methods=['GET'])
def explain_connection(artist_a, artist_b):
    """Explain similarity between two artists via quote pairs"""
    try:
        from scripts.explain import explain_connection as explain_fn
        model = get_model()
        result = explain_fn(artist_a, artist_b, top_k=5, model=model)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/path/<artist_a>/<artist_b>', methods=['GET'])
def find_path(artist_a, artist_b):
    """Find shortest path between two artists"""
    try:
        from scripts.pathfind import find_path as pathfind_fn
        result = pathfind_fn(artist_a, artist_b, min_score=0.70)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/taste-profile', methods=['POST'])
def taste_profile():
    """Generate taste profile for selected artists"""
    try:
        data = request.get_json() or {}
        artists = data.get("artists", [])
        explain = data.get("explain", False)

        if not artists or not isinstance(artists, list):
            return jsonify({"error": "Missing or invalid 'artists' field"}), 400

        from scripts.taste_profile import compute_profile as profile_fn
        model = get_model()
        result = profile_fn(artists, top_k=5, explain=explain, model=model)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/match', methods=['POST'])
def custom_artist_match():
    """Custom artist matching - compute centroid of provided artists, rank all others by similarity"""
    try:
        data = request.get_json() or {}
        artists = data.get("artists", [])
        signal = data.get("signal", "quote")  # "quote" or "critic"

        if not artists or not isinstance(artists, list):
            return jsonify({"error": "Missing or invalid 'artists' field"}), 400

        if signal not in ["quote", "critic"]:
            return jsonify({"error": "signal must be 'quote' or 'critic'"}), 400

        # Determine embedding files based on signal
        if signal == "quote":
            embeddings_file = DATA_DIR / "embeddings.npy"
            ids_file = DATA_DIR / "embedding_ids.json"
        else:
            embeddings_file = DATA_DIR / "embeddings_critic.npy"
            ids_file = DATA_DIR / "embedding_ids_critic.json"

        if not embeddings_file.exists() or not ids_file.exists():
            return jsonify({"error": f"Embeddings not yet computed for signal '{signal}'"}), 404

        try:
            # Load embeddings and IDs
            embeddings = np.load(str(embeddings_file))
            embedding_ids = load_json(ids_file)

            if not embedding_ids:
                return jsonify({"error": "No embeddings found"}), 404

            # Find indices of requested artists
            artist_indices = []
            for artist_id in artists:
                if artist_id in embedding_ids:
                    artist_indices.append(embedding_ids.index(artist_id))
                else:
                    return jsonify({"error": f"Artist '{artist_id}' not found in embeddings"}), 404

            # Compute centroid
            selected_embeddings = embeddings[artist_indices]
            centroid = np.mean(selected_embeddings, axis=0)

            # Compute cosine similarity to centroid for all artists
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity([centroid], embeddings)[0]

            # Create results excluding the input artists
            results = []
            for idx, (artist_id, sim) in enumerate(zip(embedding_ids, similarities)):
                if artist_id not in artists:  # Exclude input artists
                    results.append({
                        "artist_id": artist_id,
                        "similarity": float(sim)
                    })

            # Sort by similarity descending and return top 10
            results.sort(key=lambda x: x["similarity"], reverse=True)
            top_10 = results[:10]

            return jsonify({
                "query_artists": artists,
                "signal": signal,
                "centroid_similarity": top_10,
                "count": len(top_10)
            })

        except Exception as e:
            return jsonify({"error": f"Embedding processing error: {str(e)}"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/clusters', methods=['GET'])
def get_clusters():
    """Return clusters.json if available"""
    path = DATA_DIR / "clusters.json"
    if not path.exists():
        return jsonify({"error": "Clusters not yet computed"}), 404
    try:
        data = load_json(path)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/surprise', methods=['GET'])
def get_surprise():
    """Return highest similarity pair (the most 'basilect' pair)"""
    try:
        discoveries = load_json(DATA_DIR / "discoveries.json")
        if discoveries and len(discoveries) > 0:
            # Already sorted by score descending
            return jsonify(discoveries[0])
        else:
            return jsonify({"error": "No discoveries found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/surprise/random', methods=['GET'])
def get_surprise_random():
    """Return a random discovery pair from the top 20% of scores"""
    try:
        import random
        discoveries = load_json(DATA_DIR / "discoveries.json")
        if not discoveries or len(discoveries) == 0:
            return jsonify({"error": "No discoveries found"}), 404

        # Get top 20% of discoveries
        top_20_percent = max(1, len(discoveries) // 5)
        top_discoveries = discoveries[:top_20_percent]

        # Return a random one from the top 20%
        selected = random.choice(top_discoveries)
        return jsonify(selected)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/signal-comparison', methods=['GET'])
def get_signal_comparison():
    """Return signal_comparison.json (quote vs critic)"""
    path = DATA_DIR / "signal_comparison.json"
    if not path.exists():
        return jsonify({"error": "Signal comparison not yet computed"}), 404
    try:
        data = load_json(path)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/tags', methods=['GET'])
def get_all_tags():
    """Return all unique tags across all artists (for filter dropdowns)"""
    try:
        all_genres = set()
        all_scenes = set()

        if ARTISTS_DIR.exists():
            for artist_dir in ARTISTS_DIR.iterdir():
                if artist_dir.is_dir():
                    tags_path = artist_dir / "tags.json"
                    if tags_path.exists():
                        tags_data = load_json(tags_path) or {}
                        genres = tags_data.get("genres", [])
                        scenes = tags_data.get("scenes", [])
                        if isinstance(genres, list):
                            all_genres.update(genres)
                        if isinstance(scenes, list):
                            all_scenes.update(scenes)

        return jsonify({
            "genres": sorted(list(all_genres)),
            "scenes": sorted(list(all_scenes))
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/discoveries/filtered', methods=['GET'])
def get_filtered_discoveries():
    """Return discoveries filtered by optional query params"""
    try:
        signal = request.args.get('signal', 'quote')  # quote, critic, or divergence
        min_score = request.args.get('min_score', type=float, default=0.0)
        genre = request.args.get('genre')
        scene = request.args.get('scene')

        if signal == 'divergence':
            # Load from signal_comparison.json
            path = DATA_DIR / "signal_comparison.json"
            if not path.exists():
                return jsonify([])
            data = load_json(path)
            if not data:
                return jsonify([])

            # Convert to discovery format and sort by absolute delta
            discoveries = []
            for pair in data:
                if isinstance(pair, dict):
                    quote_score = pair.get("quote_score", 0)
                    critic_score = pair.get("critic_score", 0)
                    delta = abs(quote_score - critic_score)
                    if delta >= min_score:
                        discoveries.append({
                            "artist_a": pair.get("artist_a"),
                            "artist_b": pair.get("artist_b"),
                            "score": delta,
                            "quote_score": quote_score,
                            "critic_score": critic_score,
                            "delta": delta,
                            "signal": "divergence"
                        })

            discoveries.sort(key=lambda x: x["delta"], reverse=True)
        else:
            # Load from discoveries.json or critic_discoveries.json
            if signal == 'critic':
                path = DATA_DIR / "critic_discoveries.json"
            else:
                path = DATA_DIR / "discoveries.json"

            if not path.exists():
                return jsonify([])

            discoveries = load_json(path)
            if not discoveries:
                return jsonify([])

            # Filter by min_score
            discoveries = [d for d in discoveries if d.get("score", 0) >= min_score]

        # Filter by genre and/or scene if provided
        if genre or scene:
            filtered = []
            for pair in discoveries:
                artist_a = pair.get("artist_a")
                artist_b = pair.get("artist_b")

                # Load tags for both artists
                tags_a = load_json(ARTISTS_DIR / artist_a / "tags.json") if artist_a else {}
                tags_b = load_json(ARTISTS_DIR / artist_b / "tags.json") if artist_b else {}

                genres_a = set(tags_a.get("genres", []))
                genres_b = set(tags_b.get("genres", []))
                scenes_a = set(tags_a.get("scenes", []))
                scenes_b = set(tags_b.get("scenes", []))

                # Check if pair matches genre/scene filter
                include = True
                if genre:
                    include = include and (genre in genres_a or genre in genres_b)
                if scene:
                    include = include and (scene in scenes_a or scene in scenes_b)

                if include:
                    filtered.append(pair)

            discoveries = filtered

        return jsonify(discoveries)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/critics/discoveries', methods=['GET'])
def get_critic_discoveries():
    """Return critic_discoveries.json"""
    path = DATA_DIR / "critic_discoveries.json"
    if not path.exists():
        return jsonify([])
    try:
        data = load_json(path)
        return jsonify(data or [])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/artists/<artist_id>/critic-quotes', methods=['GET'])
def get_critic_quotes(artist_id):
    """Return critic_quotes.json for an artist"""
    quotes_path = ARTISTS_DIR / artist_id / "critic_quotes.json"
    if not quotes_path.exists():
        return jsonify({"quotes": [], "corpus_meta": None})
    try:
        data = load_json(quotes_path) or {}
        return jsonify({
            "quotes": data.get("quotes", []),
            "corpus_meta": data.get("corpus_meta")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===== API: Job Tracking =====

@app.route('/api/jobs/<job_id>', methods=['GET'])
def get_job(job_id):
    """Return job status and output"""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(jobs[job_id])

# ===== API: Pipeline Steps =====

@app.route('/api/artists/<artist_id>/scrape', methods=['POST'])
def scrape_artist(artist_id):
    """Run scrape.py for artist"""
    artist_dir = ARTISTS_DIR / artist_id
    if not artist_dir.exists():
        return jsonify({"error": "Artist not found"}), 404

    job_id = str(uuid.uuid4())
    cmd = ["python", "scripts/scrape.py", artist_id]

    thread = threading.Thread(target=run_job, args=(job_id, cmd, str(PROJECT_ROOT)))
    thread.daemon = True
    thread.start()

    return jsonify({"job_id": job_id, "status": "queued"}), 202

@app.route('/api/artists/<artist_id>/extract', methods=['POST'])
def extract_artist(artist_id):
    """Run extract.py for artist"""
    artist_dir = ARTISTS_DIR / artist_id
    if not artist_dir.exists():
        return jsonify({"error": "Artist not found"}), 404

    job_id = str(uuid.uuid4())
    cmd = ["python", "scripts/extract.py", artist_id]

    thread = threading.Thread(target=run_job, args=(job_id, cmd, str(PROJECT_ROOT)))
    thread.daemon = True
    thread.start()

    return jsonify({"job_id": job_id, "status": "queued"}), 202

@app.route('/api/artists/<artist_id>/scrape-reviews', methods=['POST'])
def scrape_reviews(artist_id):
    """Run scrape_reviews.py for artist"""
    artist_dir = ARTISTS_DIR / artist_id
    if not artist_dir.exists():
        return jsonify({"error": "Artist not found"}), 404

    job_id = str(uuid.uuid4())
    cmd = ["python", "scripts/scrape_reviews.py", artist_id]

    thread = threading.Thread(target=run_job, args=(job_id, cmd, str(PROJECT_ROOT)))
    thread.daemon = True
    thread.start()

    return jsonify({"job_id": job_id, "status": "queued"}), 202

@app.route('/api/artists/<artist_id>/extract-critic', methods=['POST'])
def extract_critic(artist_id):
    """Run extract_critic.py for artist"""
    artist_dir = ARTISTS_DIR / artist_id
    if not artist_dir.exists():
        return jsonify({"error": "Artist not found"}), 404

    job_id = str(uuid.uuid4())
    cmd = ["python", "scripts/extract_critic.py", artist_id]

    thread = threading.Thread(target=run_job, args=(job_id, cmd, str(PROJECT_ROOT)))
    thread.daemon = True
    thread.start()

    return jsonify({"job_id": job_id, "status": "queued"}), 202

@app.route('/api/artists/<artist_id>/search-reviews', methods=['POST'])
def search_reviews(artist_id):
    """Run search_reviews.py for artist"""
    artist_dir = ARTISTS_DIR / artist_id
    if not artist_dir.exists():
        return jsonify({"error": "Artist not found"}), 404

    job_id = str(uuid.uuid4())
    cmd = ["python", "scripts/search_reviews.py", artist_id]

    thread = threading.Thread(target=run_job, args=(job_id, cmd, str(PROJECT_ROOT)))
    thread.daemon = True
    thread.start()

    return jsonify({"job_id": job_id, "status": "queued"}), 202

@app.route('/api/pipeline', methods=['POST'])
def run_pipeline():
    """Run embed.py → compute.py → discover.py (quote pipeline)"""
    job_id = str(uuid.uuid4())

    def pipeline():
        """Execute pipeline steps sequentially"""
        try:
            jobs[job_id] = {
                "status": "running",
                "output": "",
                "error": None,
                "created_at": datetime.now().isoformat(),
                "steps": []
            }

            steps = [
                ("embed.py", ["python", "scripts/embed.py"]),
                ("compute.py", ["python", "scripts/compute.py"]),
                ("discover.py", ["python", "scripts/discover.py"])
            ]

            for step_name, cmd in steps:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(PROJECT_ROOT),
                    timeout=600
                )

                step_info = {
                    "name": step_name,
                    "status": "completed" if result.returncode == 0 else "failed",
                    "output": result.stdout,
                    "error": result.stderr if result.stderr else None
                }
                jobs[job_id]["steps"].append(step_info)
                jobs[job_id]["output"] += f"\n=== {step_name} ===\n{result.stdout}"

                if result.returncode != 0:
                    jobs[job_id]["status"] = "failed"
                    jobs[job_id]["error"] = f"Failed at {step_name}: {result.stderr}"
                    jobs[job_id]["completed_at"] = datetime.now().isoformat()
                    return

            jobs[job_id]["status"] = "completed"
            jobs[job_id]["completed_at"] = datetime.now().isoformat()

        except subprocess.TimeoutExpired:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = "Pipeline timed out (10+ minutes)"
            jobs[job_id]["completed_at"] = datetime.now().isoformat()
        except Exception as e:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
            jobs[job_id]["completed_at"] = datetime.now().isoformat()

    thread = threading.Thread(target=pipeline)
    thread.daemon = True
    thread.start()

    return jsonify({"job_id": job_id, "status": "queued"}), 202

@app.route('/api/pipeline/critics', methods=['POST'])
def run_critic_pipeline():
    """Run embed_critics.py → compute.py (critic) → discover.py (critic)"""
    job_id = str(uuid.uuid4())

    def pipeline():
        try:
            jobs[job_id] = {
                "status": "running",
                "output": "",
                "error": None,
                "created_at": datetime.now().isoformat(),
                "steps": []
            }

            steps = [
                ("embed_critics.py", ["python", "scripts/embed_critics.py"]),
                ("compute.py --critic", ["python", "scripts/compute.py", "--critic"]),
                ("discover.py --critic", ["python", "scripts/discover.py", "--critic"])
            ]

            for step_name, cmd in steps:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(PROJECT_ROOT),
                    timeout=600
                )

                step_info = {
                    "name": step_name,
                    "status": "completed" if result.returncode == 0 else "failed",
                    "output": result.stdout,
                    "error": result.stderr if result.stderr else None
                }
                jobs[job_id]["steps"].append(step_info)
                jobs[job_id]["output"] += f"\n=== {step_name} ===\n{result.stdout}"

                if result.returncode != 0:
                    jobs[job_id]["status"] = "failed"
                    jobs[job_id]["error"] = f"Failed at {step_name}: {result.stderr}"
                    jobs[job_id]["completed_at"] = datetime.now().isoformat()
                    return

            jobs[job_id]["status"] = "completed"
            jobs[job_id]["completed_at"] = datetime.now().isoformat()

        except subprocess.TimeoutExpired:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = "Pipeline timed out (10+ minutes)"
            jobs[job_id]["completed_at"] = datetime.now().isoformat()
        except Exception as e:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
            jobs[job_id]["completed_at"] = datetime.now().isoformat()

    thread = threading.Thread(target=pipeline)
    thread.daemon = True
    thread.start()

    return jsonify({"job_id": job_id, "status": "queued"}), 202

@app.route('/api/pipeline/compare', methods=['POST'])
def run_compare_signals():
    """Compare quote and critic signals by running compare_signals.py"""
    job_id = str(uuid.uuid4())
    cmd = ["python", "scripts/compare_signals.py"]

    thread = threading.Thread(target=run_job, args=(job_id, cmd, str(PROJECT_ROOT)))
    thread.daemon = True
    thread.start()

    return jsonify({"job_id": job_id, "status": "queued"}), 202

@app.route('/api/pipeline/full', methods=['POST'])
def run_full_pipeline():
    """Run full pipeline: quote + critic + compare"""
    job_id = str(uuid.uuid4())

    def pipeline():
        try:
            jobs[job_id] = {
                "status": "running",
                "output": "",
                "error": None,
                "created_at": datetime.now().isoformat(),
                "steps": []
            }

            all_steps = [
                ("embed.py", ["python", "scripts/embed.py"]),
                ("compute.py", ["python", "scripts/compute.py"]),
                ("discover.py", ["python", "scripts/discover.py"]),
                ("embed_critics.py", ["python", "scripts/embed_critics.py"]),
                ("compute.py --critic", ["python", "scripts/compute.py", "--critic"]),
                ("discover.py --critic", ["python", "scripts/discover.py", "--critic"])
            ]

            for step_name, cmd in all_steps:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(PROJECT_ROOT),
                    timeout=600
                )

                step_info = {
                    "name": step_name,
                    "status": "completed" if result.returncode == 0 else "failed",
                    "output": result.stdout,
                    "error": result.stderr if result.stderr else None
                }
                jobs[job_id]["steps"].append(step_info)
                jobs[job_id]["output"] += f"\n=== {step_name} ===\n{result.stdout}"

                if result.returncode != 0:
                    jobs[job_id]["status"] = "failed"
                    jobs[job_id]["error"] = f"Failed at {step_name}: {result.stderr}"
                    jobs[job_id]["completed_at"] = datetime.now().isoformat()
                    return

            jobs[job_id]["status"] = "completed"
            jobs[job_id]["completed_at"] = datetime.now().isoformat()

        except subprocess.TimeoutExpired:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = "Pipeline timed out (10+ minutes)"
            jobs[job_id]["completed_at"] = datetime.now().isoformat()
        except Exception as e:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
            jobs[job_id]["completed_at"] = datetime.now().isoformat()

    thread = threading.Thread(target=pipeline)
    thread.daemon = True
    thread.start()

    return jsonify({"job_id": job_id, "status": "queued"}), 202

# ===== Health Check =====

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(DATA_DIR)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
