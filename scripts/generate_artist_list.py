"""Generate a 20K artist list from MusicBrainz + LLM gap-fill.

Step 1a: MusicBrainz seed (~15K)
  - Browse artists who have releases and Wikipedia/Wikidata URLs
  - Filter for artists likely to have enough interview/review coverage
  - Diversify across genres, decades, regions

Step 1b: LLM genre gap-fill (~5K)
  - Analyze genre distribution gaps
  - Use Haiku to suggest artists in underrepresented areas
  - Validate against MusicBrainz

Step 1c: Deduplicate and normalize
  - Output: data/artist_list_20k.txt

Usage:
  python scripts/generate_artist_list.py [--target 20000] [--skip-llm] [--skip-mb]
"""

import sys
import re
import json
import time
import os
import argparse
from pathlib import Path
from collections import Counter

import requests

# MusicBrainz API via requests (avoids urllib SSL issues on macOS Python 3.14)
MB_BASE = "https://musicbrainz.org/ws/2"
MB_HEADERS = {"User-Agent": "BasilectEngine/1.0 (https://github.com/basilect-engine)", "Accept": "application/json"}

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SEED_FILE = DATA_DIR / "seed_artists.txt"
OUTPUT_FILE = DATA_DIR / "artist_list_20k.txt"
CACHE_FILE = DATA_DIR / "mb_artist_cache.json"

# MusicBrainz rate limit: 1 request/sec
MB_DELAY = 1.1


def normalize_artist_id(name: str) -> str:
    """Normalize artist name to ID."""
    id_str = name.lower().strip()
    id_str = re.sub(r'[^\w\s\-]', '', id_str)
    id_str = re.sub(r'\s+', '_', id_str)
    id_str = re.sub(r'_+', '_', id_str)
    return id_str.strip('_')


def load_seed_artists() -> list:
    """Load existing seed artists."""
    if not SEED_FILE.exists():
        return []
    names = []
    for line in SEED_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            names.append(line)
    return names


def load_cache() -> dict:
    """Load cached MusicBrainz results."""
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {"artists": [], "offset": 0, "total": 0}


def save_cache(cache: dict):
    """Save MusicBrainz cache atomically."""
    tmp = CACHE_FILE.with_suffix('.tmp')
    tmp.write_text(json.dumps(cache, ensure_ascii=False))
    tmp.rename(CACHE_FILE)


def mb_search_artists(tag: str, limit: int = 100, offset: int = 0) -> dict:
    """Search MusicBrainz for artists by tag using requests."""
    params = {"query": f"tag:{tag}", "limit": limit, "offset": offset, "fmt": "json"}
    resp = requests.get(f"{MB_BASE}/artist", params=params, headers=MB_HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_mb_artists(target: int = 15000, resume: bool = True) -> list:
    """Fetch artists from MusicBrainz using search by genre tags."""
    cache = load_cache() if resume else {"artists": [], "offset": 0, "total": 0}
    seen_ids = {a["mbid"] for a in cache["artists"]}
    artists = list(cache["artists"])

    if len(artists) >= target:
        print(f"Cache already has {len(artists)} artists (target: {target})")
        return artists

    print(f"Starting MusicBrainz fetch (have {len(artists)}, target {target})")

    genre_tags = [
        "rock", "pop", "electronic", "hip hop", "jazz", "classical",
        "folk", "metal", "punk", "r&b", "soul", "blues", "country",
        "reggae", "world", "ambient", "experimental", "indie",
        "post-punk", "shoegaze", "dream pop", "noise", "industrial",
        "techno", "house", "drum and bass", "dubstep", "trip hop",
        "post-rock", "math rock", "emo", "hardcore", "grunge",
        "synthpop", "new wave", "krautrock", "progressive rock",
        "psychedelic rock", "garage rock", "art rock", "glam rock",
        "black metal", "death metal", "doom metal", "stoner rock",
        "free jazz", "fusion", "bebop", "avant-garde jazz",
        "neo-soul", "funk", "disco", "afrobeat", "bossa nova",
        "minimal", "glitch", "idm", "vaporwave", "lo-fi",
        "singer-songwriter", "americana", "bluegrass",
        "latin", "flamenco", "tropicalia", "cumbia",
        "k-pop", "j-pop", "city pop", "enka",
        "dancehall", "dub", "ska", "grime",
        "contemporary classical", "minimalism",
        "drone", "dark ambient", "witch house",
        "indie pop", "indie rock", "chamber pop", "baroque pop",
        "hip-hop", "trap", "conscious hip hop", "boom bap",
        "alternative rock", "alternative", "new age",
        "gospel", "choral", "opera",
    ]

    for tag in genre_tags:
        if len(artists) >= target:
            break

        offset = 0
        tag_count = 0
        max_per_tag = 500

        while tag_count < max_per_tag and len(artists) < target:
            try:
                result = mb_search_artists(tag, limit=100, offset=offset)
                artist_list = result.get("artists", [])
                if not artist_list:
                    break

                for a in artist_list:
                    mbid = a.get("id", "")
                    name = a.get("name", "")
                    sort_name = a.get("sort-name", "")
                    country = a.get("country", "")
                    atype = a.get("type", "")
                    score = int(a.get("score", "0"))

                    if score < 50:
                        continue
                    if atype and atype.lower() in ("orchestra", "choir", "character"):
                        continue
                    if mbid in seen_ids:
                        continue
                    if not name or len(name) < 2:
                        continue

                    seen_ids.add(mbid)
                    artists.append({
                        "name": name,
                        "mbid": mbid,
                        "sort_name": sort_name,
                        "country": country,
                        "type": atype,
                        "score": score,
                        "source_tag": tag,
                    })
                    tag_count += 1

                offset += 100
                time.sleep(MB_DELAY)

                if len(artists) % 500 == 0:
                    cache["artists"] = artists
                    cache["offset"] = offset
                    cache["total"] = len(artists)
                    save_cache(cache)
                    print(f"  {len(artists)} artists cached ({tag}: +{tag_count})")

            except Exception as e:
                print(f"  ERROR fetching tag '{tag}' offset {offset}: {e}")
                time.sleep(5)
                break

        if tag_count > 0:
            print(f"  [{len(artists):,}/{target:,}] tag '{tag}': +{tag_count}")

    cache["artists"] = artists
    cache["total"] = len(artists)
    save_cache(cache)

    print(f"\nMusicBrainz fetch complete: {len(artists):,} artists")
    return artists


def analyze_genre_gaps(artists: list) -> dict:
    """Analyze genre distribution to find underrepresented areas."""
    tag_counts = Counter()
    country_counts = Counter()

    for a in artists:
        tag = a.get("source_tag", "unknown")
        tag_counts[tag] += 1
        country = a.get("country", "")
        if country:
            country_counts[country] += 1

    return {
        "tag_distribution": dict(tag_counts.most_common()),
        "country_distribution": dict(country_counts.most_common(30)),
        "total": len(artists),
        "underrepresented_regions": [
            region for region, count in country_counts.items()
            if count < len(artists) * 0.01  # Less than 1% representation
        ],
    }


def llm_gap_fill(artists: list, target_additional: int = 5000) -> list:
    """Use Haiku to suggest artists in underrepresented genres/regions."""
    try:
        from anthropic import Anthropic
        from dotenv import load_dotenv
        load_dotenv(BASE_DIR / ".env", override=True)
    except ImportError:
        print("WARNING: anthropic/dotenv not available, skipping LLM gap-fill")
        return []

    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("WARNING: ANTHROPIC_API_KEY not set, skipping LLM gap-fill")
        return []

    client = Anthropic(api_key=api_key)
    gaps = analyze_genre_gaps(artists)
    existing_names = {a["name"].lower() for a in artists}

    # Identify underrepresented areas
    underrepresented = []
    tag_dist = gaps["tag_distribution"]
    total = gaps["total"]

    # Genres with < 2% of total
    for tag, count in tag_dist.items():
        if count < total * 0.02:
            underrepresented.append(f"genre: {tag}")

    # Regions with very few artists
    regions_to_fill = [
        "Africa (West, East, South, North)",
        "Southeast Asia",
        "Middle East",
        "South America (beyond Brazil/Argentina)",
        "Eastern Europe",
        "Central Asia",
        "Caribbean",
        "Pacific Islands",
        "Indigenous music traditions globally",
    ]

    # Also gap-fill specific creative niches
    niches = [
        "sound art and installation artists",
        "film/TV composers known for distinctive approaches",
        "producers who rarely perform but shaped genres",
        "DIY/underground artists with strong interview presence",
        "cross-disciplinary artists (visual art + music)",
        "pioneering women in electronic music",
        "artists known for unusual instruments or techniques",
    ]

    new_artists = []
    batch_size = 100  # Ask for 100 artists per prompt

    prompts = []

    # Build prompts for each gap area
    for region in regions_to_fill:
        prompts.append(
            f"List {batch_size} notable musicians/bands from {region} who are likely to have "
            f"English-language interviews or reviews online. Include a mix of traditional and "
            f"contemporary artists. Format: one name per line, nothing else."
        )

    for niche in niches:
        prompts.append(
            f"List {batch_size} notable {niche} who are likely to have English-language "
            f"interviews or reviews online. Include artists from diverse backgrounds and decades. "
            f"Format: one name per line, nothing else."
        )

    # General diversity prompts
    for decade_start in range(1950, 2030, 10):
        decade_end = decade_start + 9
        prompts.append(
            f"List {batch_size} notable musicians/bands from {decade_start}-{decade_end} who "
            f"are likely to have English-language interviews or reviews online. Include diverse "
            f"genres and regions. Avoid: {', '.join(list(existing_names)[:20])}... "
            f"Format: one name per line, nothing else."
        )

    print(f"\nLLM gap-fill: {len(prompts)} prompts, targeting ~{target_additional} additional artists")

    for i, prompt in enumerate(prompts):
        if len(new_artists) >= target_additional:
            break

        try:
            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )

            response = message.content[0].text
            names = [line.strip().lstrip('0123456789.-) ') for line in response.splitlines()]
            names = [n for n in names if n and len(n) > 1 and n.lower() not in existing_names]

            for name in names:
                if name.lower() not in existing_names:
                    existing_names.add(name.lower())
                    new_artists.append({
                        "name": name,
                        "mbid": "",
                        "sort_name": "",
                        "country": "",
                        "type": "",
                        "score": 0,
                        "source_tag": "llm_gap_fill",
                    })

            print(f"  [{i+1}/{len(prompts)}] +{len(names)} names ({len(new_artists):,} total new)")
            time.sleep(0.5)

        except Exception as e:
            print(f"  ERROR on prompt {i+1}: {e}")
            time.sleep(2)

    print(f"\nLLM gap-fill complete: {len(new_artists):,} new artists")
    return new_artists


def deduplicate_and_normalize(artists: list, seed_names: list) -> list:
    """Deduplicate and normalize the final artist list."""
    seen_ids = set()
    seen_names_lower = set()
    final = []

    # Seed artists go first
    for name in seed_names:
        aid = normalize_artist_id(name)
        if aid not in seen_ids and name.lower() not in seen_names_lower:
            seen_ids.add(aid)
            seen_names_lower.add(name.lower())
            final.append(name)

    # Then MusicBrainz + LLM artists
    for a in artists:
        name = a["name"].strip()
        aid = normalize_artist_id(name)

        if aid in seen_ids or name.lower() in seen_names_lower:
            continue

        # Skip names that are too short or too long
        if len(name) < 2 or len(name) > 100:
            continue

        # Skip names that look like labels, compilations, etc.
        skip_patterns = [
            r'^various\b', r'^unknown\b', r'^va\b', r'^compilation',
            r'^soundtrack', r'^ost\b', r'\brecords\b', r'\brecordings\b',
            r'\blabel\b', r'\bmusic group\b',
        ]
        if any(re.search(p, name, re.IGNORECASE) for p in skip_patterns):
            continue

        seen_ids.add(aid)
        seen_names_lower.add(name.lower())
        final.append(name)

    return final


def main():
    parser = argparse.ArgumentParser(description="Generate 20K artist list")
    parser.add_argument("--target", type=int, default=20000,
                       help="Target number of artists (default: 20000)")
    parser.add_argument("--skip-llm", action="store_true",
                       help="Skip LLM gap-fill step")
    parser.add_argument("--skip-mb", action="store_true",
                       help="Skip MusicBrainz fetch (use cache only)")
    parser.add_argument("--mb-target", type=int, default=15000,
                       help="Target for MusicBrainz fetch (default: 15000)")
    parser.add_argument("--resume", action="store_true", default=True,
                       help="Resume from cache (default: True)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Basilect Engine: Artist List Generation (target: {args.target:,})")
    print("=" * 60)

    # Load seed artists
    seed_names = load_seed_artists()
    print(f"\nSeed artists: {len(seed_names)}")

    # Step 1a: MusicBrainz
    if args.skip_mb:
        cache = load_cache()
        mb_artists = cache.get("artists", [])
        print(f"\nUsing cached MusicBrainz data: {len(mb_artists):,} artists")
    else:
        mb_artists = fetch_mb_artists(target=args.mb_target, resume=args.resume)

    # Step 1b: LLM gap-fill
    llm_artists = []
    if not args.skip_llm:
        remaining = args.target - len(mb_artists) - len(seed_names)
        if remaining > 0:
            llm_artists = llm_gap_fill(mb_artists, target_additional=remaining)
        else:
            print(f"\nSkipping LLM gap-fill (already have enough: {len(mb_artists) + len(seed_names):,})")

    # Step 1c: Deduplicate and normalize
    all_artists = mb_artists + llm_artists
    final_list = deduplicate_and_normalize(all_artists, seed_names)

    # Write output
    OUTPUT_FILE.write_text("\n".join(final_list) + "\n")

    print(f"\n{'=' * 60}")
    print(f"Final artist list: {len(final_list):,} artists")
    print(f"Output: {OUTPUT_FILE}")
    print(f"  From seed: {len(seed_names)}")
    print(f"  From MusicBrainz: {len(mb_artists):,}")
    print(f"  From LLM gap-fill: {len(llm_artists):,}")
    print(f"  After dedup: {len(final_list):,}")
    print(f"{'=' * 60}")

    # Also save genre analysis
    if mb_artists:
        gaps = analyze_genre_gaps(mb_artists)
        analysis_path = DATA_DIR / "artist_list_analysis.json"
        analysis_path.write_text(json.dumps(gaps, indent=2, ensure_ascii=False))
        print(f"\nGenre analysis saved to {analysis_path}")


if __name__ == "__main__":
    main()
