# Automated Artist Ingestion Pipeline

This document describes the new fully-automated pipeline for ingesting artist interview data at scale.

---

## Overview

The pipeline automates all manual steps so you can ingest 1400+ artists without manual Claude skill invocations.

**Before (manual):**
1. `/search-artist {name}` (Claude skill)
2. `python scripts/scrape.py {artist_id}` (automated)
3. `/extract-artist {artist_id}` (Claude skill)
4. `/run-pipeline` (automated global pipeline)

**After (fully automated):**
- `python scripts/batch.py --file artists.txt` (one command, handles everything)

---

## Architecture

### Four new scripts

1. **`scripts/search.py`** — Find interview URLs
   - Uses pluggable search backends (Brave, SerpAPI, Google Custom Search)
   - Filters video/audio URLs, blocked domains
   - Scores and deduplicates results
   - Creates/appends to `data/artists/{artist_id}/sources.json`

2. **`scripts/extract_llm.py`** — Extract quotes via Claude API
   - Reads scraped text from `sources.json`
   - Calls Claude Sonnet for verbatim quote extraction
   - Returns structured JSON (text, publication, date)
   - Aggregates into `data/artists/{artist_id}/quotes.json`
   - Validates corpus (≥5 quotes, ≥3 sources, ≥2 years)

3. **`scripts/ingest.py`** — Single-artist full pipeline
   - Orchestrates: search → scrape → extract → validate
   - Skip any step with `--skip-*` flags (to resume partially)
   - Usage: `python scripts/ingest.py "Artist Name"`

4. **`scripts/batch.py`** — Batch processor
   - Processes multiple artists sequentially
   - Respects API rate limits with configurable delays
   - Skips artists with valid corpora (unless `--force`)
   - Logs progress to `data/batch_log.json`
   - Auto-runs global pipeline (embed → compute → discover) when done

---

## Search API Abstraction

All scripts use a pluggable `SearchProvider` base class. Three implementations included:

### BraveSearchProvider (default)
- Requires: `BRAVE_API_KEY` env var
- Best quality, low cost
- 2000 queries/month free tier

### SerpAPIProvider
- Requires: `SERPAPI_KEY` env var
- Reliable, slightly higher cost
- 100 free queries/month

### GoogleCSEProvider
- Requires: `GOOGLE_API_KEY` + `GOOGLE_CSE_ID` env vars
- Most powerful but most expensive
- 100 free queries/day

**Swap providers by passing `--search-provider brave|serpapi|google` to any script.**

---

## Environment Setup

Create a `.env` file in the project root with API keys:

```bash
# Required for extract_llm.py
ANTHROPIC_API_KEY=sk-...

# Choose ONE search backend:
BRAVE_API_KEY=...
# OR
SERPAPI_KEY=...
# OR
GOOGLE_API_KEY=...
GOOGLE_CSE_ID=...
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage Examples

### Single artist (end-to-end)
```bash
python scripts/ingest.py "Bjork"
```

Output:
```
[SEARCH] Finding interview URLs...
[SCRAPE] Processing bjork...
[EXTRACT] Processing bjork...
[REPORT]
SUCCESS: bjork has a VALID corpus
```

### Resume a partial ingest
If search worked but scrape failed:
```bash
python scripts/ingest.py "Bjork" --skip-search
```

### Batch from file
Create `artists.txt`:
```
Bjork
Grimes
FKA Twigs
Aphex Twin
```

Run:
```bash
python scripts/batch.py --file artists.txt
```

Progress logged to stdout + `data/batch_log.json`.

### Batch with rate limiting
```bash
python scripts/batch.py --file artists.txt --delay 3 --search-provider serpapi
```

- `--delay 3` = 3 seconds between artists
- `--search-provider serpapi` = use SerpAPI instead of Brave

### Batch with force re-ingest
```bash
python scripts/batch.py --file artists.txt --force
```

Re-ingests even artists with valid corpora.

### Batch without running global pipeline
```bash
python scripts/batch.py --file artists.txt --no-pipeline
```

Skips the final `embed → compute → discover` run. Useful if you'll batch again later.

---

## Included: seed_artists.txt

A curated list of ~128 diverse artists across genres, decades, and geographies:

- Electronic/Experimental: Aphex Twin, Burial, Tim Hecker, Oneohtrix Point Never, Alva Noto
- Hip-hop/Trap: Tyler the Creator, MF DOOM, Earl Sweatshirt, Flying Lotus, Madlib
- Indie/Alt: Radiohead, Bon Iver, Mount Kimbie, James Blake, St. Vincent
- Contemporary Classical: Arca, SOPHIE, Kaija Saariaho, Sofia Gubaidulina
- Jazz/Fusion: Herbie Hancock, Keith Jarrett, Bill Evans
- Pioneers: John Cage, Pauline Oliveros, Yoko Ono, Pierre Schaeffer

Start with:
```bash
python scripts/batch.py --file data/seed_artists.txt
```

---

## Pipeline Output

### Batch log (`data/batch_log.json`)

```json
[
  {
    "timestamp": "2026-04-03T18:30:00.123456",
    "search_provider": "brave",
    "artists_requested": 10,
    "succeeded": ["bjork", "grimes", "fka_twigs"],
    "failed": ["unknown_artist"],
    "skipped": ["artist_with_valid_corpus"]
  }
]
```

Each batch appends a new entry. Useful for tracking ingestion history.

### Artist corpus (`data/artists/{artist_id}/quotes.json`)

```json
{
  "quotes": [
    {
      "text": "I wanted to hear music which sampled all the old soul and jazz that I liked.",
      "publication": "Sound & Recording Magazine",
      "date": "2003"
    }
  ],
  "corpus_meta": {
    "quote_count": 12,
    "source_count": 4,
    "date_range": ["2001", "2020"],
    "corpus_valid": true
  }
}
```

---

## Filtering & Quality Control

### Blocked domains (`data/blocked_domains.md`)

URLs from these domains are skipped (known paywalls, un-scrapeable sites):
- pitchfork.com
- interviewmagazine.com
- clashmusic.com
- factmag.com
- talkhouse.com
- americansongwriter.com

Add more as needed.

### Video/audio filtering

Automatically skips video/audio platforms:
- YouTube, Vimeo, TikTok
- SoundCloud, Spotify, Bandcamp
- Apple Music, Amazon Music

### Result scoring

Scores search results by:
- Snippet length (longer = more substantial)
- Interview keywords (interview, creative process, how I make, etc.)
- Penalty for fluff (top 10, best of, ranking, review)
- Deduplicates by domain (max 2 per publication)

---

## Error Handling & Recovery

### Failed searches
If search fails (API down, quota exceeded):
- Error logged to stdout
- Script continues with other artists
- Artist added to `failed` list in batch log

### Failed scrapes
If trafilatura can't extract text:
- `sources.json` marks entry with `"text": null`
- Article skipped at extract stage
- Artist can be re-scraped if needed

### Failed extractions
If Claude API call fails:
- Error logged, extraction aborted
- Artist marked as failed
- Can retry later with `--force`

### Rate limiting
- Batch processor respects delays between requests
- LLM extraction includes 1-second delays between API calls
- Search providers have built-in timeouts (10s)

---

## Scaling Considerations

### Current design
- Sequential processing (not parallel) for simplicity
- Per-artist API calls to stay within context limits
- ~1-2 minutes per artist (search + scrape + extract)
- ~100 artists = ~1-2 hours

### For 1400+ artists
At 1 minute per artist: ~23 hours (1 night of compute)

**Recommended approach:**
1. Split `seed_artists.txt` into batches of 100
2. Run overnight in parallel processes on different machines
3. Aggregate results in morning
4. Run global pipeline once

---

## Artist ID Normalization

All artist names are normalized to IDs:
- Lowercase
- Spaces → underscores
- Strip punctuation
- Example: `"FKA Twigs!"` → `"fka_twigs"`

This is consistent with the existing project convention.

---

## Corpus Validity Thresholds

An artist's corpus is `corpus_valid: true` if it has:
- ≥5 quotes
- ≥3 distinct sources
- ≥2 distinct years

This matches the original extract.py validation logic.

Artists with invalid corpora are still stored but marked as invalid. You can:
- Search for more sources manually and re-ingest
- Adjust thresholds in `extract_llm.py` + `batch.py` (MIN_QUOTES, etc.)
- Accept invalid corpora and lower thresholds

---

## Troubleshooting

### "BRAVE_API_KEY not set"
Set the env var: `export BRAVE_API_KEY=...`

### "No results found"
Search may have returned no non-blocked results. Try:
- A different artist name
- A different search provider

### "No sources with text found"
All articles failed to scrape (blocked, JS, paywall). Try:
- Checking `blocked_domains.md` for culprits
- Adding domain to blocked list if persistently failing
- Manually adding sources to `sources.json`

### "INVALID corpus"
Artist has <5 quotes or <3 sources. Options:
- Search for more sources and re-ingest: `python scripts/ingest.py "Artist" --force`
- Manually add sources to `sources.json` and re-scrape + extract
- Lower validation thresholds

### API quota exceeded
- Wait for quota reset
- Switch search provider: `--search-provider serpapi`
- Reduce delay between artists to batch faster

---

## Extending the pipeline

### Custom search provider
Subclass `SearchProvider` and implement `search()`:

```python
class MySearchProvider(SearchProvider):
    def search(self, query: str, num_results: int = 10) -> list:
        # Return [{"title": "...", "url": "...", "snippet": "..."}, ...]
        pass

# Add to get_provider():
providers["mysearch"] = MySearchProvider
```

### Custom quote extraction
Replace Claude with your own LLM in `extract_llm.py`:

```python
def extract_quotes_from_text(...):
    # Call your LLM instead of Claude
    quotes = your_llm.extract(...)
    return quotes
```

### Custom filtering
Add domain/URL patterns to `is_blocked_domain()` or `is_video_audio_url()`.

---

## Next Steps (Post-MVP)

- Parallel ingest (multiprocessing or async)
- Incremental embedding (don't re-embed all artists each time)
- Web UI for batch management + monitoring
- Influence detection (extract "influenced by X" from same corpus)
- Genre/sonic validation (MAEST post-hoc scoring)
- Artist conflict detection (similar names, same person?)
