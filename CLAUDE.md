# Basilect Engine
A music artist similarity engine that surfaces non-obvious connections between artists based on how they talk about making music. Two artists are "basilect-connected" if their creative philosophy is similar but their genre/scene is not.
The signal is artist self-discourse: verbatim quotes from interviews, embedded and compared. No critic voice, no Claude-authored summaries.

---

## Pipeline

1. **Search** — Claude finds interview URLs per artist (see below)
2. **Scrape** — `scripts/scrape.py` fetches article text via trafilatura
3. **Extract** — Claude pulls verbatim artist quotes from scraped text (see below)
4. **Embed** — `scripts/embed.py` embeds quotes and aggregates per artist
5. **Compute** — `scripts/compute.py` builds pairwise cosine similarity matrix
6. **Discover** — `scripts/discover.py` surfaces ranked artist pairs

Steps 1–3 run **per artist**. Steps 4–6 run **globally** across all artists.

**Prototype rule: do not delegate steps 1–3 to subagents.** Run them directly in the main conversation — one artist at a time. Subagents have been unreliable (WebFetching live URLs instead of using stored text, writing invalid JSON, silently correcting verbatim quotes).

---

## Artist node schema

`data/artists/{artist_id}/sources.json`
```json
[{ "url": "...", "publication": "The Wire", "date": "2019", "title": "...", "text": "..." }]
```

`data/artists/{artist_id}/quotes.json`
```json
{
  "quotes": [{ "text": "...", "publication": "...", "url": "...", "date": "..." }],
  "corpus_meta": { "quote_count": 12, "source_count": 4, "date_range": ["2014", "2021"], "corpus_valid": true }
}
```

**corpus_valid thresholds:** ≥5 quotes, ≥3 distinct sources, ≥2 distinct years

---

## Adding a new artist

1. `/search-artist {artist}` — find interview URLs
2. `python scripts/scrape.py {artist_id}` — fetch article text
3. `/extract-artist {artist_id}` — pull quotes from scraped text
4. Check `corpus_valid`
5. `/run-pipeline` when ready — runs embed → compute → discover globally

---

## Searching for interview URLs

command: `/search-artist {artist}`

Full instructions: `.claude/skills/search-artist/SKILL.md`

`artist_id`: lowercase, spaces to underscores, strip punctuation; e.g. "Artist Name!"->"artist_name"

---

## Extracting quotes from scraped text

command: `/extract-artist {artist_id}`

Full instructions: `.claude/skills/extract-artist/SKILL.md`

---

## Scripts

| Script | Purpose |
|---|---|
| `scripts/scrape.py` | Fetch article text via trafilatura |
| `scripts/extract.py` | LEGACY: sentence-transformer probe (fallback only; precision issues) |
| `scripts/embed.py` | Embed quotes, aggregate per artist via median |
| `scripts/compute.py` | Pairwise cosine similarity matrix |
| `scripts/discover.py` | Surface ranked artist pairs |

---

## Known limitations
- **embed.py is global-only.** Re-embeds all artists every run. Fine for current scale (<50 artists).
- **extract.py is legacy.** Sentence-transformer probe has precision issues (catches journalist voice, wrong speakers). Kept as fallback, not primary path.
- **Un-crawlable domains:** See `data/blocked_domains.md`.

---

## Deferred
- Sonic validation via MAEST (post-hoc, later phase)
- Specific influence citation extraction from same corpus
- Manual tag/genre proximity layer (removed for now, may return)
- Visualization
- Scaling beyond prototype
