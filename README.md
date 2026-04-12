# Basilect Engine

Surfaces non-obvious connections between music artists based on how they talk about making music. Two artists are "basilect-connected" if their creative philosophy is similar but their genre or scene is not.

Signal source: verbatim artist quotes from interviews — embedded and compared. No critic voice, no summaries.

---

## How it works

1. **Search** — find interview URLs for an artist
2. **Scrape** — fetch article text via trafilatura
3. **Extract** — pull verbatim artist quotes from scraped text
4. **Embed** — embed quotes, aggregate per artist via median vector
5. **Compute** — build pairwise cosine similarity matrix
6. **Discover** — rank artist pairs by similarity score

Steps 1–3 run per artist. Steps 4–6 run globally across all artists.

---

## Setup

```bash
pip install -r requirements.txt
```

Requires an Anthropic API key in `.env`:
```
ANTHROPIC_API_KEY=sk-...
```

---

## Adding an artist

```bash
# 1. Find interview URLs (Claude Code skill)
/search-artist {Artist Name}

# 2. Scrape articles
python scripts/scrape.py {artist_id}

# 3. Extract quotes (Claude Code skill)
/extract-artist {artist_id}

# 4. Run the global pipeline
/run-pipeline
```

`artist_id` is lowercase with underscores: `"Artist Name!" → "artist_name"`

A valid corpus requires ≥5 quotes, ≥3 distinct sources, ≥2 distinct years.

---

## Data layout

```
data/
  artists/{artist_id}/
    sources.json     # scraped article text + metadata
    quotes.json      # extracted verbatim quotes + corpus_meta
  similarity.npy     # pairwise cosine similarity matrix
  discoveries.json   # ranked artist pairs
```

---

## Scripts

| Script | Purpose |
|---|---|
| `scripts/scrape.py` | Fetch article text via trafilatura |
| `scripts/embed.py` | Embed quotes, aggregate per artist |
| `scripts/compute.py` | Pairwise cosine similarity matrix |
| `scripts/discover.py` | Surface ranked artist pairs |

---

## Current artists

animal_collective, arthur_verocai, astrid_sonne, badbadnotgood, bill_evans, bladee, boy_harsher, clairo, fishmans, nujabes, radiohead, skrillex, tame_impala, yung_lean
