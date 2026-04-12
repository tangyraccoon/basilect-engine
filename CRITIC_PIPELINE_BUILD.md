# Critic Discourse Mining Pipeline — Build Summary

## Overview
A complete parallel pipeline for mining critic discourse about artists. Mirrors the existing interview quote pipeline but targets music reviews and features from critics instead of artist self-interviews.

## New Scripts Created

### 1. `scripts/search_reviews.py`
Discovers review and feature URLs for an artist.

**Key features:**
- Targets music review sites: pitchfork.com, thequietus.com, stereogum.com, residentadvisor.net, tinymixtapes.com, drownedinsound.com, noisey.vice.com, albumoftheyear.org, sputnikmusic.com, nme.com, thelineofbestfit.com, popmatters.com, consequence.net
- Uses same search provider abstraction as `search.py` (Brave/SerpAPI/Google CSE)
- Filters video/audio domains and blocked domains
- Scores results by review-specific keywords and site authority
- Returns 3-8 review URLs per artist
- Stores in `data/artists/{artist_id}/review_sources.json`

**Usage:**
```bash
python scripts/search_reviews.py "Artist Name" [--provider brave|serpapi|google]
```

### 2. `scripts/scrape_reviews.py`
Extracts text from review URLs using trafilatura.

**Key features:**
- Reads from `data/artists/{artist_id}/review_sources.json`
- Writes text back to same file
- Handles network failures and paywalled content gracefully
- Identical to `scrape.py` but for review sources

**Usage:**
```bash
python scripts/scrape_reviews.py {artist_id}
```

### 3. `scripts/extract_critic.py`
Extracts critic discourse passages via Claude API.

**Key features:**
- Reads from `data/artists/{artist_id}/review_sources.json`
- Extracts passages where critics describe the artist's creative approach, philosophy, artistic identity
- NOT direct quotes from the artist — purely critic voice
- Focuses on: songwriting approach, production methods, artistic evolution, relationship to genre, aesthetic sensibility
- Uses Claude Sonnet 4
- Output: `data/artists/{artist_id}/critic_quotes.json` (same format as quotes.json)
- Includes corpus validation metadata

**Validation thresholds:**
- Minimum 5 passages
- Minimum 2 sources
- Minimum 1 year span

**Usage:**
```bash
python scripts/extract_critic.py {artist_id} [--model claude-sonnet-4-20250514]
```

### 4. `scripts/embed_critics.py`
Embeds critic passages using all-MiniLM-L6-v2.

**Key features:**
- Loads all `data/artists/*/critic_quotes.json` files
- Embeds each passage individually
- Aggregates per artist via component-wise median (same as `embed.py`)
- Output: `data/critic_embeddings.npy` and `data/critic_embedding_ids.json`
- Mirrors `embed.py` exactly but for critic signal

**Usage:**
```bash
python scripts/embed_critics.py
```

### 5. `scripts/compute_critics.py`
Computes pairwise cosine similarity for critic embeddings.

**Key features:**
- Loads `data/critic_embeddings.npy` and `data/critic_embedding_ids.json`
- Computes full cosine similarity matrix
- Output: `data/critic_similarity.npy`
- Prints correlation statistics

**Usage:**
```bash
python scripts/compute_critics.py
```

### 6. `scripts/discover_critics.py`
Ranks artist pairs by critic discourse similarity.

**Key features:**
- Loads critic similarity matrix
- Ranks all pairs by similarity score (descending)
- Output: `data/critic_discoveries.json`
- Prints formatted ranking table

**Usage:**
```bash
python scripts/discover_critics.py
```

### 7. `scripts/compare_signals.py` ⭐ Novel Analysis
Compares quote similarity vs critic similarity for artists that have both signals.

**Key features:**
- Loads both `embeddings.npy` and `critic_embeddings.npy`
- Loads both similarity matrices
- For each artist pair in both signals:
  - Computes `quote_sim` (from interviews)
  - Computes `critic_sim` (from reviews)
  - Computes `delta = quote_sim - critic_sim`
- Ranks pairs by absolute delta (largest divergences most interesting)
- Computes Pearson and Spearman correlation between signals
- Output: `data/signal_comparison.json`

**Interpretation:**
- **Positive delta**: Artist philosophical alignment stronger than critics recognize
- **Negative delta**: Critics see alignment that artists' own words don't support
- **Correlation**: How well the two signals agree (expects ~0.3-0.5)

**Usage:**
```bash
python scripts/compare_signals.py
```

### 8. `scripts/ingest_full.py`
Complete dual-signal pipeline for one artist.

**Sequence:**
1. Search interviews → scrape → extract quotes (existing pipeline)
2. Search reviews → scrape reviews → extract critic passages
3. Report both corpus validities

**Output:**
- Both `quotes.json` and `critic_quotes.json`
- Separate validity reporting for each signal
- Success if at least one signal is valid

**Usage:**
```bash
python scripts/ingest_full.py "Artist Name" [--search-provider brave]
```

### 9. `scripts/batch_full.py`
Batch orchestrator for multiple artists with dual signals.

**Sequence:**
1. Process N artists with `ingest_full.py` (sequential with delays)
2. Run global dual-signal pipeline:
   - `embed.py` (quotes)
   - `compute.py` (quotes)
   - `discover.py` (quotes)
   - `embed_critics.py` (critic)
   - `compute_critics.py` (critic)
   - `discover_critics.py` (critic)
   - `compare_signals.py` (comparison)
3. Log all results to `data/batch_full_log.json`

**Features:**
- Individual artist failures don't crash batch
- Rate limiting between artists
- Force re-ingest option
- Optional global pipeline skip

**Usage:**
```bash
python scripts/batch_full.py --file data/seed_artists.txt [--search-provider brave] [--delay 2.0] [--force] [--no-pipeline]
```

Or with inline artists:
```bash
python scripts/batch_full.py --artists "Artist 1,Artist 2,Artist 3"
```

## Data Layout

### Per-Artist Directory: `data/artists/{artist_id}/`
```
data/artists/{artist_id}/
├── sources.json                    # Interview sources (existing)
├── review_sources.json            # Review sources (NEW)
├── quotes.json                    # Extracted interview quotes (existing)
└── critic_quotes.json             # Extracted critic passages (NEW)
```

### Global Data Files
```
data/
├── embeddings.npy                 # Quote embeddings (existing)
├── embedding_ids.json             # Quote embedding artist IDs (existing)
├── similarity.npy                 # Quote similarity matrix (existing)
├── discoveries.json               # Quote-based rankings (existing)
├── critic_embeddings.npy          # Critic embeddings (NEW)
├── critic_embedding_ids.json      # Critic embedding artist IDs (NEW)
├── critic_similarity.npy          # Critic similarity matrix (NEW)
├── critic_discoveries.json        # Critic-based rankings (NEW)
├── signal_comparison.json         # Quote vs critic comparison (NEW)
├── batch_log.json                 # Batch ingest log (existing)
└── batch_full_log.json            # Dual-signal batch log (NEW)
```

## Technical Notes

### Design Patterns
- All new critic scripts mirror existing quote scripts exactly
- Reuses `SearchProvider` abstraction from `search.py`
- Same artist ID normalization and directory structure
- Identical embedding and similarity computation (all-MiniLM-L6-v2, median aggregation, cosine similarity)
- Error handling: individual source failures don't halt pipeline

### Key Differences from Quote Pipeline
1. **Review sources** are more abundant (8-10 per artist) vs interview sources (3-6)
2. **Validation thresholds** are looser for critic passages (≥5 passages vs ≥5 quotes, ≥2 sources vs ≥3)
3. **LLM extraction** focuses on critic voice, not artist quotes
4. **Signal comparison** is a novel analysis not present in quote pipeline

### Dependencies
- All new scripts use existing dependencies: trafilatura, anthropic, sentence-transformers, numpy, scikit-learn, scipy
- No new package requirements
- Environment variables: `ANTHROPIC_API_KEY`, search provider keys (same as existing)

## Typical Workflow

### Single Artist with Both Signals
```bash
python scripts/ingest_full.py "Björk"
```

### Batch Processing
```bash
# Create seed list
echo "Björk" > data/seed_artists.txt
echo "Arca" >> data/seed_artists.txt
echo "Oneohtrix Point Never" >> data/seed_artists.txt

# Process all
python scripts/batch_full.py --file data/seed_artists.txt --delay 3.0

# This runs all 9 scripts globally + compare_signals at end
```

### Analysis: Which artists have strongest critic alignment?
```bash
# After batch_full completes:
python -c "import json; data = json.load(open('data/signal_comparison.json')); 
print('Strongest alignment:', data['pairs'][0]); 
print('Weakest alignment:', data['pairs'][-1]); 
print(f\"Pearson r: {data['stats']['pearson_r']:.3f}\")"
```

## Success Indicators

- All 9 scripts pass Python syntax validation ✓
- Reuses SearchProvider abstraction from existing codebase ✓
- Mirrors quote pipeline structure exactly ✓
- No modifications to existing files ✓
- Proper error handling and rate limiting ✓
- Comprehensive logging and validation ✓

## Next Steps

1. Test with a single artist: `python scripts/ingest_full.py "Test Artist"`
2. Verify outputs in `data/artists/{artist_id}/`
3. Run batch with 5-10 artists, check `data/signal_comparison.json`
4. Analyze divergence cases for editorial insights
