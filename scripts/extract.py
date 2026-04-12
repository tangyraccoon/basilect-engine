import sys
import json
from pathlib import Path

import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

PROBE = "musician describing their creative process, philosophy, and approach to making music"
THRESHOLD = 0.3
MIN_WORDS = 8
MODEL_NAME = "all-MiniLM-L6-v2"

MIN_QUOTES = 5
MIN_SOURCES = 3
MIN_YEARS = 2

nltk.download("punkt_tab", quiet=True)


def extract(artist_id):
    path = Path(f"data/artists/{artist_id}/sources.json")
    if not path.exists():
        print(f"Not found: {path}")
        sys.exit(1)

    sources = json.loads(path.read_text())
    sources_with_text = [s for s in sources if s.get("text")]

    if not sources_with_text:
        print("No sources with text found.")
        sys.exit(1)

    model = SentenceTransformer(MODEL_NAME)
    probe_embedding = model.encode([PROBE])

    quotes = []

    for source in sources_with_text:
        sentences = nltk.sent_tokenize(source["text"])
        candidates = [s for s in sentences if len(s.split()) >= MIN_WORDS]

        if not candidates:
            continue

        embeddings = model.encode(candidates)
        sims = cosine_similarity(embeddings, probe_embedding).flatten()

        for sentence, score in zip(candidates, sims):
            if score >= THRESHOLD:
                quotes.append({
                    "text": sentence,
                    "publication": source.get("publication", ""),
                    "url": source.get("url", ""),
                    "date": source.get("date", ""),
                })

        kept = sum(1 for s in sims if s >= THRESHOLD)
        print(f"{source.get('publication', source.get('url', '?'))} — {kept}/{len(candidates)} sentences kept")

    quote_count = len(quotes)
    source_count = len({q["url"] for q in quotes})
    years = sorted({q["date"][:4] for q in quotes if q.get("date")})
    date_range = [years[0], years[-1]] if years else []
    corpus_valid = (
        quote_count >= MIN_QUOTES
        and source_count >= MIN_SOURCES
        and len(set(years)) >= MIN_YEARS
    )

    output = {
        "quotes": quotes,
        "corpus_meta": {
            "quote_count": quote_count,
            "source_count": source_count,
            "date_range": date_range,
            "corpus_valid": corpus_valid,
        },
    }

    out_path = Path(f"data/artists/{artist_id}/quotes.json")
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    valid_str = "VALID" if corpus_valid else "INVALID"
    print(f"\n{quote_count} quotes from {source_count} sources ({valid_str})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/extract.py <artist_id>")
        sys.exit(1)
    extract(sys.argv[1])
