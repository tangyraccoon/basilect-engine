# Basilect Engine — Web App

A full-stack web application for exploring music artist similarities through interview analysis. The app integrates with the Basilect Engine pipeline scripts to scrape, extract, embed, and analyze artist interview content.

## Setup

### Prerequisites
- Python 3.8+
- All project dependencies (see main project README for pipeline deps)

### Installation

1. Install web server dependencies:
```bash
pip install -r app/requirements.txt
```

2. Install pipeline dependencies (in project root):
```bash
pip install -r requirements.txt
```

## Run

From the project root directory:

```bash
python app/server.py
```

The app will start at `http://localhost:5000`

Open your browser and navigate to `http://localhost:5000`

## Features

### Discover Page
Browse ranked artist similarity pairs. Filter by similarity tier (0.70+, 0.80+, 0.85+) and search by artist name.

### Artists Page
View all artists in a grid with corpus statistics (quote count, source count, years covered, validity status).
- Click any artist to view full details
- Add new artists
- Manage interview sources (URLs)
- Trigger scraping and quote extraction
- View extracted quotes

### Artist Detail
For each artist:
- Add interview source URLs (publication, date, title)
- Run scraper to fetch article text via trafilatura
- Run quote extractor to isolate key statements
- View extracted quotes with source attribution
- Monitor job progress in real-time
- Delete sources or entire artist

### Pipeline Page
Run the global pipeline:
1. **Embed** — vectorize all quotes using all-MiniLM-L6-v2
2. **Compute** — calculate cosine similarity matrix
3. **Discover** — rank artist pairs by similarity

Useful after scraping and extracting new artists.

### Matrix Page
Heatmap view of the similarity matrix. Darker colors = higher similarity scores.

## Architecture

### Backend (Flask)
- **server.py** — REST API endpoints, subprocess orchestration, job tracking
- Routes for CRUD operations on artists and sources
- Job queue with real-time polling support

### Frontend (Vanilla JS + CSS)
- **static/index.html** — Single-page app (SPA) with embedded React-style components
- Dark theme with coral/violet/gold gradients
- Modal forms for adding artists and sources
- Real-time job status polling

### Data Layout
All data is stored relative to the project root:

```
data/
  artists/
    {artist_id}/
      sources.json      # [{"url": "...", "publication": "...", "date": "2019", "title": "...", "text": "..."}]
      quotes.json       # {"quotes": [...], "corpus_meta": {...}}
  embeddings.npy        # Embedded quote vectors (after embed.py)
  embedding_ids.json    # Quote IDs (after embed.py)
  similarity.npy        # Cosine similarity matrix (after compute.py)
  discoveries.json      # Ranked artist pairs (after discover.py)
  blocked_domains.md    # Scraping blocklist
```

## API Reference

### Discoveries & Matrix
- `GET /api/discoveries` — ranked artist pairs
- `GET /api/matrix` — similarity matrix (ids + 2D scores array)
- `GET /api/blocked-domains` — scraping blocklist

### Artists
- `GET /api/artists` — list all artists with corpus_meta
- `GET /api/artists/<id>` — full artist data (sources + quotes)
- `GET /api/artists/<id>/sources` — sources.json
- `GET /api/artists/<id>/quotes` — quotes.json
- `POST /api/artists` — create new artist (body: `{"name": "..."}`)
- `POST /api/artists/<id>/sources` — add sources (body: `{"sources": [...]}`)
- `DELETE /api/artists/<id>/sources` — remove source by URL (body: `{"url": "..."}`)
- `DELETE /api/artists/<id>` — delete artist

### Processing
- `POST /api/artists/<id>/scrape` — run scrape.py (returns job_id)
- `POST /api/artists/<id>/extract` — run extract.py (returns job_id)
- `POST /api/pipeline` — run embed → compute → discover (returns job_id)

### Job Tracking
- `GET /api/jobs/<job_id>` — poll job status, output, error

## Workflow

1. **Add Artist** — Create a new artist profile
2. **Add Sources** — Paste interview URLs with publication/date/title
3. **Scrape** — Fetch article text from URLs via trafilatura
4. **Extract Quotes** — Use sentence-transformers to find similar statements
5. **Run Pipeline** — Embed all quotes, compute similarity, discover pairs
6. **Explore** — Browse ranked similarities in the Discover page

## Troubleshooting

### "Matrix not available" on Matrix page
Run the pipeline first. This requires at least 2 artists with valid corpora (≥5 quotes, ≥3 sources, ≥2 years).

### Scraping fails
Some URLs are blocked or paywalled. Check the output for specific failures. Add reliable sources (interviews from open publications).

### Job times out
Long-running operations (especially pipeline on large corpora) have a 10-minute timeout. Check server logs for details.

### Artist ID normalization
Artist names are normalized to lowercase with spaces → underscores and punctuation stripped. "BadBadNotGood" → `badbadnotgood`.

## Development

The frontend is a single HTML file with embedded CSS and vanilla JS. No build step required. Modify `app/static/index.html` directly.

The backend uses Flask with CORS enabled for development. Modify `app/server.py` to add endpoints or change behavior.

All subprocess calls run from the PROJECT ROOT (parent of `app/`) so script paths resolve correctly.
