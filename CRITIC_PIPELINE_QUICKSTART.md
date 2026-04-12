# Critic Pipeline Quick Start

## One Artist (Full Dual Signal)

```bash
python scripts/ingest_full.py "Artist Name"
```

This runs:
1. Search for interviews + scrape + extract quotes
2. Search for reviews + scrape + extract critic discourse
3. Report both corpus validities

Output: `data/artists/{artist_id}/quotes.json` and `data/artists/{artist_id}/critic_quotes.json`

## Batch (Multiple Artists)

```bash
# Create seed file
echo "Björk" > data/seed_artists.txt
echo "Arca" >> data/seed_artists.txt
echo "Oneohtrix Point Never" >> data/seed_artists.txt

# Process all
python scripts/batch_full.py --file data/seed_artists.txt
```

This runs `ingest_full` for each artist, then the global dual-signal pipeline:
- All quote embeddings & ranking
- All critic embeddings & ranking
- Signal comparison analysis

Output: `data/signal_comparison.json` with divergence analysis

## Just Critic Signal (If You Already Have Quotes)

```bash
# Get reviews for an artist
python scripts/search_reviews.py "Artist Name"
python scripts/scrape_reviews.py artist_name

# Extract critic discourse
python scripts/extract_critic.py artist_name

# Then embed & analyze globally
python scripts/embed_critics.py
python scripts/compute_critics.py
python scripts/discover_critics.py
python scripts/compare_signals.py
```

## Just One Step

```bash
# Search for reviews only
python scripts/search_reviews.py "Artist Name" --provider brave

# Scrape reviews only
python scripts/scrape_reviews.py artist_name

# Extract critic discourse only
python scripts/extract_critic.py artist_name

# Embed all critic passages
python scripts/embed_critics.py

# Compute critic similarity matrix
python scripts/compute_critics.py

# Discover critic-similar pairs
python scripts/discover_critics.py

# Compare signals (requires both quote & critic embeddings)
python scripts/compare_signals.py
```

## Key Files

**Per Artist:**
- `data/artists/{artist_id}/review_sources.json` — Review URLs found
- `data/artists/{artist_id}/critic_quotes.json` — Extracted critic passages

**Global:**
- `data/critic_embeddings.npy` — 384-dimensional embeddings
- `data/critic_similarity.npy` — Artist pair similarities
- `data/critic_discoveries.json` — Ranked pairs by similarity
- `data/signal_comparison.json` — Quote vs critic divergence analysis
- `data/batch_full_log.json` — Run logs

## Check Results

```bash
# See critic-similar pairs
python -c "import json; d = json.load(open('data/critic_discoveries.json')); print([p['a'] + ' × ' + p['b'] + f\" ({p['score']:.3f})\" for p in d[:5]])"

# See signal comparison
python -c "import json; d = json.load(open('data/signal_comparison.json')); print(f\"Pearson r: {d['stats']['pearson_r']:.3f}\"); print('Top divergences:', [(p['a'], p['b'], f\"+{p['delta']:.3f}\") for p in d['pairs'][:3]])"

# Check corpus validity
python -c "import json; c = json.load(open('data/artists/bjork/critic_quotes.json')); m = c['corpus_meta']; print(f\"Valid: {m['corpus_valid']}, Passages: {m['quote_count']}, Sources: {m['source_count']}\")"
```

## Troubleshooting

**"review_sources.json not found"**
→ Run `search_reviews.py` first

**"No sources with text"**
→ Some URLs failed to scrape (paywalls/blocks). Try different search provider or wait for content mirrors.

**"No passages extracted"**
→ Claude didn't find critic discourse matching criteria. Check review quality.

**"Compare failed: embeddings not found"**
→ Need to run both `embed.py` (for quotes) and `embed_critics.py` (for critic)

**API rate limits**
→ Batch uses 2-second delays between artists by default. Increase with `--delay 5.0`

## Environment

Must have:
- `ANTHROPIC_API_KEY` in .env
- Search provider key (BRAVE_API_KEY, SERPAPI_KEY, or GOOGLE_API_KEY)
- trafilatura, sentence-transformers, anthropic, numpy, sklearn, scipy

All should already be installed from existing quote pipeline.
