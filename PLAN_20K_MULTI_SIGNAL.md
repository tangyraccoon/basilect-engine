# Basilect Engine: 20K Artist Multi-Signal Architecture Plan

**Goal:** Populate the Basilect app with 20,000 artists using multiple signals (not just interview quotes), with cost-optimized extraction, crash-resilient batch processing, and a QA gate before committing to a full run. Push to GitHub so it can be mounted.

---

## What Went Wrong Last Time

Two failures compounded into a total loss of credits with zero usable output:

### 1. Extraction model was too expensive

`extract_llm.py` and `extract_critic.py` both hardcode `claude-sonnet-4-20250514`. Sonnet 4 is a powerful model but wildly over-specced for structured extraction from already-scraped text. At scale:

- ~6 interview sources per artist × 20,000 artists = **120,000 API calls** (interview signal alone)
- ~6 review sources per artist × 20,000 = another **120,000 calls** (critic signal)
- Sonnet 4 input pricing on long article text makes this a multi-hundred-dollar run

The extraction task — pulling verbatim quotes from clean text with a structured JSON schema — is well within Haiku's capabilities.

### 2. Batch scraper had no effective resume

`batch.py` has **no checkpointing at all**. On crash/restart, it re-iterates the full artist list from the top. `batch_full.py` added a `batch_checkpoint.json` mechanism with `--resume`, but:

- The checkpoint file is **cleared on successful completion** (`checkpoint_clear()`), so partial runs that error out before the final line lose their checkpoint
- The skip logic checks `is_corpus_valid()` which requires a *complete* corpus — an artist that searched+scraped but didn't extract gets re-processed from scratch
- There's no per-step state — if extraction fails mid-artist, the entire artist is retried including search and scrape

**Net result:** Each restart re-ran search + scrape + expensive Sonnet extraction on the same first N artists, burning credits on duplicate work.

---

## Architecture Overview

The new pipeline has 7 phases. Phases 1–3 are prep work (no API credits spent). Phase 4 is the QA gate. Only after QA passes do phases 5–7 execute the full run.

```
Phase 1: Artist List Generation (20K names)
Phase 2: Code Changes (cost + resilience fixes)
Phase 3: Infrastructure (state machine, dedup, config)
Phase 4: QA Gate (small batch validation)
Phase 5: Full Ingestion Run
Phase 6: Global Pipeline + App Build
Phase 7: GitHub Push + Mount
```

---

## Phase 1: Generate the 20K Artist List

The current `seed_artists.txt` has 128 entries with many duplicates (Ryoji Ikeda ×4, Arca ×4, Merzbow ×3, etc.). After dedup it's closer to ~95 unique artists.

### Approach: MusicBrainz bulk export + LLM expansion

**Step 1a — MusicBrainz seed (~15K)**
- MusicBrainz has a free API and data dumps with millions of artists
- Query by: artist type = "Person" or "Group", has at least 1 release, has an English Wikipedia link (proxy for "enough public discourse exists to scrape")
- This filters to artists who are notable enough to have interviews/reviews online
- Export: `artist_name, musicbrainz_id, country, begin_year, tags[]`
- Script: `scripts/generate_artist_list.py`

**Step 1b — LLM genre gap-fill (~5K)**
- After MusicBrainz pull, analyze genre distribution
- Use Haiku to generate artists in underrepresented genres/regions/decades
- Validate each name against MusicBrainz or Discogs to confirm they're real
- This ensures diversity beyond the Western indie/electronic bias in the current seed list

**Step 1c — Deduplicate and normalize**
- Normalize all names through the existing `normalize_artist_id()` function
- Detect duplicates: exact match, case-insensitive match, and fuzzy match (Levenshtein ≤ 2)
- Detect conflicts: "Cage John" vs "John Cage" — canonicalize to most common form
- Output: `data/artist_list_20k.txt` (one per line, deduplicated, normalized)

**Search API note:** We're using the **Brave Search API** for all URL discovery. Brave free tier = 2,000 queries/month. At ~2 search queries per artist (interviews + reviews) × 20,000 artists = ~40,000 queries. You'll need Brave's paid tier ($5/1,000 queries beyond free) or spread the search phase across multiple months. Budget ~$200 for Brave search at scale, or pre-cache search results during Phase 4 QA and reuse them.

**Deliverable:** A clean list of 20,000 unique artist names ready for ingestion.

---

## Phase 2: Code Changes

### 2a — Switch extraction to Haiku 4.5

**Files:** `scripts/extract_llm.py`, `scripts/extract_critic.py`

Change:
```python
DEFAULT_MODEL = "claude-sonnet-4-20250514"
```
To:
```python
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
```

**Why Haiku is sufficient:** The extraction prompt is highly structured — it's essentially "read this text, find quotes by {artist}, return JSON." There's no complex reasoning, creative generation, or ambiguous judgment. Haiku handles structured extraction well, and the prompt already has clear rules and output format.

**Cost impact:** Haiku is roughly 12–15× cheaper per token than Sonnet 4. For 240K API calls, this is the difference between a $300+ run and a ~$20–30 run.

**Also add `--model` as a CLI flag** (already partially supported but default matters most).

### 2b — Add Anthropic Batch API support

**New file:** `scripts/extract_batch_api.py`

The Anthropic Message Batches API processes requests asynchronously at **50% off** the standard per-token price. For a 20K-artist run, this compounds with the Haiku switch for massive savings.

Flow:
1. Collect all (artist, source_text, metadata) tuples that need extraction
2. Submit as a batch of up to 10,000 requests to `POST /v1/messages/batches`
3. Poll for completion (typically minutes to hours)
4. Parse results back into per-artist `quotes.json` / `critic_quotes.json`

**Fallback:** If batch API has issues, the existing per-request flow still works (just slower and 2× the cost).

### 2c — Per-artist state machine

**New file:** `scripts/state.py`

Replace the binary "has valid corpus or not" check with a granular per-artist state file:

```json
// data/artists/{artist_id}/state.json
{
  "artist_id": "bjork",
  "artist_name": "Bjork",
  "status": "extract_complete",
  "steps": {
    "interview_search": {"status": "done", "timestamp": "2026-04-11T10:00:00", "result": "6 urls"},
    "interview_scrape": {"status": "done", "timestamp": "2026-04-11T10:01:00", "result": "5/6 scraped"},
    "interview_extract": {"status": "done", "timestamp": "2026-04-11T10:02:00", "result": "12 quotes"},
    "review_search": {"status": "done", "timestamp": "2026-04-11T10:03:00", "result": "6 urls"},
    "review_scrape": {"status": "done", "timestamp": "2026-04-11T10:04:00", "result": "4/6 scraped"},
    "review_extract": {"status": "pending", "timestamp": null, "result": null}
  },
  "error": null,
  "last_updated": "2026-04-11T10:04:00"
}
```

**Key behaviors:**
- On restart, read `state.json` and **resume from the first incomplete step**
- Never re-run a step that already succeeded
- If search found 6 URLs and scrape got 5, don't re-search — just scrape the 1 remaining
- State is written atomically (write to `.tmp`, rename) to survive mid-write crashes
- A global `data/batch_state.json` tracks which artists have been visited at all, so the batch processor can skip to the first unvisited artist instantly

**This is the single most important change.** It eliminates the "restart from top" problem entirely.

### 2d — Smarter batch orchestrator

**File:** `scripts/batch_v2.py` (new, replaces `batch.py` and `batch_full.py`)

Changes from current `batch_full.py`:
- Uses `state.json` per artist instead of a single checkpoint file
- On startup, scans all artist directories, reads state, builds a work queue of only incomplete artists
- Sorts work queue so artists with *some* work done come first (finish partial work before starting new)
- Supports `--max-credits` flag: estimates cost per remaining step and stops before exceeding budget
- Supports `--dry-run`: prints what it *would* do without making any API calls
- Supports `--phase interview|critic|both`: run only one signal pipeline at a time
- Writes a running `data/batch_progress.json` with live stats (started, completed, failed, skipped, estimated remaining time)
- Logs per-artist timing so you can estimate total run duration accurately

### 2e — Update monitoring dashboard (`monitor.py`)

The existing `monitor.py` is a rich terminal dashboard that tracks `batch_full.py` runs in real time — artist progress, live logs, cost projection, process health, and ETA. It needs to be updated for the new architecture:

**Pricing constants:** Currently hardcoded to Sonnet 4 pricing:
```python
MODEL_NAME         = "claude-sonnet-4-20250514"
PRICE_INPUT_PER_M  = 3.00
PRICE_OUTPUT_PER_M = 15.00
AVG_INPUT_TOKENS   = 2500
AVG_OUTPUT_TOKENS  = 900
BUDGET             = 100.00
```
Update to Haiku 4.5 pricing. Make the model/pricing configurable (read from a shared config or detect from `batch_v2.py`'s `--model` flag). Reduce `BUDGET` to reflect the cheaper run — or make it a CLI arg.

**Process detection:** Currently does `pgrep -f "batch_full.py"`. Update to detect `batch_v2.py`.

**State machine integration:** Instead of (or in addition to) parsing raw log output, read from per-artist `state.json` files for ground-truth progress. This is more reliable than log parsing and works even if the output file isn't available.

**New signals tracking:** Add counters for influence extractions and tag extractions in addition to quotes and passages.

**Parallel batch support:** If running multiple `batch_v2.py` processes in parallel (Option B in Phase 5), the dashboard should detect all running instances and aggregate their progress.

**Artist list source:** Currently reads from `data/seed_artists.txt` (128 artists). Update to read from `data/artist_list_20k.txt` or accept a `--file` arg. For 20K artists the "All Artists" panel will need pagination or summary mode rather than listing every name.

### 2f — Pre-extraction text filtering

**New file:** `scripts/prefilter.py`

Before sending article text to the LLM, apply cheap heuristic filters:

- **Length check:** Skip articles < 500 chars (too short to contain meaningful quotes)
- **Language check:** Skip non-English articles (use a fast langdetect check)
- **Artist name check:** Skip articles that don't mention the artist name at all (bad search result)
- **Interview signal check:** Look for interview markers (Q:, A:, quotation marks with artist name nearby) — if none found in interview sources, flag as likely not an interview

This reduces unnecessary LLM calls by ~15-25% based on typical scrape quality, saving both cost and time.

---

## Phase 3: Additional Signals Architecture

The current system has two signals: interview quotes (Path A) and critic discourse (Path B). To get to "more signals than just quotes from artists themselves," here's what to add:

### Signal 1: Interview Quotes (existing — Path A)
Artist's own words about their creative process. Already built.

### Signal 2: Critic Discourse (existing — Path B)  
How critics describe the artist's creative approach. Already built.

### Signal 3: Influence Graph
**New file:** `scripts/extract_influences.py`

From the *same scraped interview text* (no additional scraping needed), extract influence citations: "I was inspired by X," "I grew up listening to Y," "Z changed how I think about music."

- Uses Haiku with a targeted prompt
- Output: `data/artists/{artist_id}/influences.json`
  ```json
  {"influences": [{"name": "Brian Eno", "context": "quote about the influence", "direction": "influenced_by"}]}
  ```
- Build a directed graph: artist → influenced_by → artist
- This creates a *structural* signal (graph topology) vs. the *semantic* signals (embeddings)
- Two artists who share influence ancestors but different genres = strong basilect connection

### Signal 4: Thematic Tags
**New file:** `scripts/extract_tags.py`

From scraped text (interviews + reviews), extract thematic tags about creative approach:

- Process: texture, improvisation, sampling, field_recording, layering, minimalism, maximalism, etc.
- Philosophy: anti-commercial, DIY, collaborative, isolation, spiritual, political, etc.
- Approach: studio_as_instrument, live_performance, digital, analog, hybrid, etc.

Output: `data/artists/{artist_id}/tags.json` — a set of weighted tags derived from textual evidence.

This is **cheap to extract** — Haiku can tag an article in a single short call, and the output is tiny. Tags enable faceted filtering in the app ("show me artists who share the 'texture + improvisation' philosophy but are in different genres").

### Signal Combination Strategy

**New file:** `scripts/combine_signals.py`

Rather than one similarity matrix, compute multiple and combine:

```
final_similarity = (
    w1 * quote_cosine_similarity +      # How they talk about music
    w2 * critic_cosine_similarity +       # How critics describe them  
    w3 * influence_jaccard_similarity +   # Shared influences
    w4 * tag_cosine_similarity            # Shared creative themes
)
```

Default weights: w1=0.4, w2=0.2, w3=0.2, w4=0.2 (interview quotes remain primary signal).

Weights are configurable via `data/signal_weights.json` so you can tune after seeing results.

---

## Phase 4: QA Gate

**This is the critical step.** Do NOT proceed to Phase 5 until all QA checks pass.

### QA Step 1: Haiku vs Sonnet extraction quality (10 artists)

Pick 10 artists that already have valid Sonnet-extracted corpora (from the existing 22). Re-extract with Haiku. Compare:

- **Quote count:** Does Haiku find roughly the same number? (±20% is acceptable)
- **Quote fidelity:** Are the quotes actually verbatim? (Spot-check 5 quotes per artist manually)
- **JSON validity:** Does Haiku produce valid JSON 100% of the time?
- **False positives:** Does Haiku incorrectly attribute interviewer/journalist speech to the artist?
- **False negatives:** Does Haiku miss obvious quotes that Sonnet caught?

**Script:** `scripts/qa_model_compare.py` — runs both models on same input, diffs output.

**Pass criteria:** Haiku quality is ≥85% of Sonnet quality on quote count, with no systematic false-positive pattern. If Haiku fails: try Haiku with a slightly more detailed prompt, or fall back to Sonnet 4 with Batch API (50% off).

### QA Step 2: End-to-end mini batch (50 artists)

Run the full new pipeline on 50 artists from the seed list that don't have corpora yet:

- Search → Scrape → Prefilter → Extract (Haiku) → all 4 signals
- Verify state machine works: kill the process mid-run, restart, confirm it resumes correctly
- Check: corpus validity rate (target: ≥60% of artists get valid corpora)
- Check: cost tracking matches estimates
- Check: `batch_progress.json` updates correctly
- Check: influence extraction produces plausible results
- Check: tag extraction produces coherent, non-random tags

**Script:** `scripts/qa_mini_batch.py` — wrapper that runs batch_v2 on 50 artists and produces a QA report.

### QA Step 3: Resume resilience test

1. Start batch on 20 artists
2. After 5 complete, kill the process (Ctrl+C or `kill`)
3. Restart with same command
4. Verify: it picks up at artist 6 (or wherever the interruption was), does not re-search/re-scrape/re-extract artists 1–5
5. Verify: final output is identical to an uninterrupted run

### QA Step 4: Cost projection

After the 50-artist mini batch:
- Calculate actual cost per artist (from Anthropic usage dashboard or token counts in logs)
- Extrapolate to 20,000 artists
- Verify the projected total is within your credit budget
- If not: adjust signal selection (drop least valuable signal), reduce sources per artist, or tighten prefiltering

**Deliverable:** A QA report (`data/qa_report.json`) with pass/fail for each check and cost projection for the full run.

---

## Phase 5: Full Ingestion Run

Only execute after Phase 4 QA passes.

### Execution strategy

**Option A: Single long run (simplest)**
```bash
python scripts/batch_v2.py --file data/artist_list_20k.txt --phase both --delay 1.5 --search-provider brave --model claude-haiku-4-5-20251001
```
- Estimated time: ~1–2 min/artist × 20,000 = ~14–28 days for sequential
- Pro: simple, reliable with state machine
- Con: very slow

**Option B: Parallel batches (recommended)**
- Split `artist_list_20k.txt` into 10 files of 2,000 each
- Run 10 parallel processes (on same machine, different terminals, or different machines)
- Each reads from its own file but writes to the shared `data/artists/` directory
- State machine prevents conflicts (each artist has its own directory)
- Estimated time: ~1.5–3 days

```bash
# Terminal 1
python scripts/batch_v2.py --file data/splits/batch_01.txt --phase both --delay 1.5 --search-provider brave

# Terminal 2
python scripts/batch_v2.py --file data/splits/batch_02.txt --phase both --delay 1.5 --search-provider brave

# ... etc

# Monitoring terminal — launch the dashboard to watch all parallel processes:
python monitor.py
```

**Script:** `scripts/split_artist_list.py` — splits the master list into N even files.

**Option C: Batch API (cheapest)**
- Run search + scrape for all 20K artists first (no LLM cost)
- Collect all scraped text into a single Batch API submission
- Submit to Anthropic Batch API for async processing
- Parse results when complete
- Estimated time: search+scrape ~3–5 days, then batch extraction ~hours

This is the **cheapest** option (Haiku + 50% Batch API discount) but requires the batch API integration from Phase 2b.

### Monitoring

Launch the monitoring dashboard in a separate terminal before starting the batch:
```bash
python monitor.py
```

The dashboard shows:
- Overall progress bar + artist-by-artist status (valid / partial / active / pending)
- Current artist + phase (search → scrape → extract, for both interview and critic signals)
- Live log output with color-coded messages
- Cost tracking: spent so far, projected total, budget bar, budget verdict
- Timing: minutes per artist, ETA, time remaining
- Process health: running / stale / dead detection

Also watch:
- `data/batch_progress.json` for live stats
- Anthropic usage dashboard at console.anthropic.com for credit consumption
- Per-artist `state.json` files for error patterns

If error rate exceeds 20%, pause and investigate before continuing.

---

## Phase 6: Global Pipeline + App Build

After ingestion completes:

### 6a — Run global pipelines
```bash
python scripts/embed.py              # Quote embeddings
python scripts/embed_critics.py      # Critic embeddings
python scripts/embed_tags.py         # Tag embeddings (new)
python scripts/compute.py            # Quote similarity matrix
python scripts/compute_critics.py    # Critic similarity matrix
python scripts/compute_tags.py       # Tag similarity matrix (new)
python scripts/build_influence_graph.py  # Influence graph (new)
python scripts/combine_signals.py    # Combined similarity matrix (new)
python scripts/discover.py           # Ranked pairs (combined signal)
python scripts/cluster.py            # Cluster artists
```

### 6b — Update the app

The Flask app (`app/server.py`) currently reads from single-signal data files. Update to:
- Load combined similarity matrix as default
- Allow signal selection in the UI (quotes only, critic only, combined, etc.)
- Display influence connections on artist detail pages
- Display tags on artist detail pages
- Handle the larger dataset (20K × 20K similarity matrix = ~1.5GB in float32 — may need sparse representation or top-K pruning)

### 6c — Data size considerations

A 20K × 20K float32 similarity matrix is ~1.5GB. Options:
- **Sparse storage:** Only store pairs above a similarity threshold (e.g., > 0.3). Most pairs will be low-similarity.
- **Top-K per artist:** Store only the top 100 most similar artists per artist. This is what the app actually needs for discovery.
- **Output format:** Switch from `.npy` to a compressed format or SQLite for the app layer.

---

## Phase 7: GitHub Push + Mount

### 7a — Clean up repo
- Remove duplicate/legacy scripts (`extract.py` legacy, redundant batch files)
- Update `CLAUDE.md` with new pipeline docs
- Update `README.md` with multi-signal architecture
- Update `requirements.txt` with any new dependencies
- Add `.gitignore` entries for large data files (embeddings, similarity matrices)

### 7b — Data strategy for Git
- Artist JSON data (quotes, tags, influences, state): **commit to repo** (text files, reasonable size)
- Embedding `.npy` files: **do not commit** (binary, large, regenerable). Add to `.gitignore`.
- Similarity matrices: **do not commit**. Regenerable from embeddings.
- Instead, add a `scripts/rebuild_pipeline.py` that regenerates all embeddings + matrices from the JSON data

### 7c — Push
```bash
git add -A
git commit -m "20K artist multi-signal architecture"
git push origin main
```

### 7d — Mount verification
- Clone on target machine
- Run `pip install -r requirements.txt`
- Run `python scripts/rebuild_pipeline.py`
- Start app: `python app/server.py`
- Verify: app loads, shows 20K artists, discovery works

---

## Implementation Order for Claude Code

When you bring this to Claude Code, execute in this order:

```
1.  scripts/state.py                    — Per-artist state machine (the resilience fix)
2.  scripts/prefilter.py                — Cheap text filtering before LLM calls
3.  Modify extract_llm.py              — Default to Haiku, add --model flag
4.  Modify extract_critic.py           — Default to Haiku, add --model flag
5.  scripts/extract_influences.py       — Influence extraction (new signal)
6.  scripts/extract_tags.py             — Tag extraction (new signal)
7.  scripts/batch_v2.py                 — New batch orchestrator with state machine
8.  Update monitor.py                   — Haiku pricing, state.json reads, batch_v2 detection, 20K scale
9.  scripts/qa_model_compare.py         — Haiku vs Sonnet quality test
10. scripts/qa_mini_batch.py            — 50-artist end-to-end QA
11. RUN QA (Phase 4)                    — Stop here until QA passes
12. scripts/generate_artist_list.py     — MusicBrainz 20K list generation
13. scripts/split_artist_list.py        — Split list for parallel runs
14. scripts/extract_batch_api.py        — Batch API integration (optional, for max savings)
15. RUN FULL INGESTION (Phase 5)        — The big run (use --search-provider brave)
16. Launch monitor.py                   — Watch progress live during the run
17. scripts/embed_tags.py               — Tag embeddings
18. scripts/build_influence_graph.py    — Influence graph
19. scripts/combine_signals.py          — Multi-signal combination
20. Update app/server.py                — Multi-signal UI
21. Git cleanup + push (Phase 7)
```

**Search provider:** All search operations use the **Brave Search API** (`--search-provider brave`). This is the default in `search.py` and `search_reviews.py`. Requires `BRAVE_API_KEY` in `.env`. Brave's free tier gives 2,000 queries/month — for 20K artists you'll need a paid tier or to spread searches across months. SerpAPI and Google CSE are available as fallbacks if Brave quota is exhausted.

---

## Cost Estimates

| Scenario | Model | API Type | Est. Cost (20K artists) |
|----------|-------|----------|------------------------|
| Old approach | Sonnet 4 | Standard | $300–500+ |
| Haiku switch only | Haiku 4.5 | Standard | $20–40 |
| Haiku + Batch API | Haiku 4.5 | Batch (50% off) | $10–20 |
| Haiku + Batch + prefilter | Haiku 4.5 | Batch + filter | $8–15 |

The combination of Haiku + Batch API + prefiltering should bring the full 20K run to roughly **$10–20 in API credits** vs. the $300+ the old approach would have cost.

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Haiku extraction quality too low | QA gate catches this before spending credits. Fallback: Sonnet + Batch API. |
| MusicBrainz doesn't yield 20K scrapeable artists | Supplement with Discogs, RateYourMusic, or LLM-generated lists validated against databases. |
| 20K × 20K similarity matrix too large for app | Use sparse/top-K storage. Only materialize what the app needs. |
| Brave Search quota exhausted | Rotate between Brave, SerpAPI, Google CSE. Or batch search phase over multiple days. |
| Mid-run crash | State machine resumes exactly where it left off. This is solved. |
| Bad data pollutes similarity results | Post-hoc QA: sample 100 random artist pairs from results, verify the connection makes sense. Flag outliers. |
