"""Cheap heuristic pre-filters applied before sending text to the LLM.

Reduces unnecessary API calls by ~15-25% by skipping text that is:
- Too short to contain meaningful quotes
- Not in English
- Doesn't mention the target artist
- Lacks interview markers (for interview sources)
"""

import re
from typing import Optional


MIN_TEXT_LENGTH = 500  # chars — shorter articles rarely have useful quotes


def check_length(text: str) -> Optional[str]:
    """Return rejection reason if text is too short, else None."""
    if len(text) < MIN_TEXT_LENGTH:
        return f"too short ({len(text)} chars < {MIN_TEXT_LENGTH})"
    return None


def check_language(text: str) -> Optional[str]:
    """Return rejection reason if text is not English, else None.

    Uses a fast heuristic: checks for common English stop words.
    Falls back to langdetect if available, but doesn't require it.
    """
    # Fast heuristic: count English stop words in first 1000 chars
    sample = text[:1000].lower()
    stop_words = {"the", "and", "that", "with", "for", "was", "his", "her",
                  "but", "not", "have", "this", "from", "they", "been", "said",
                  "about", "would", "which", "their", "when", "what", "there"}
    words = set(re.findall(r'\b[a-z]+\b', sample))
    hits = len(words & stop_words)

    if hits < 4:
        # Try langdetect as backup
        try:
            from langdetect import detect
            lang = detect(text[:2000])
            if lang != "en":
                return f"non-English (detected: {lang})"
        except ImportError:
            return f"likely non-English ({hits} stop words in sample)"
        except Exception:
            pass  # langdetect can fail on short/ambiguous text

    return None


def check_artist_mention(text: str, artist_name: str) -> Optional[str]:
    """Return rejection reason if artist name isn't mentioned, else None."""
    text_lower = text.lower()
    name_lower = artist_name.lower()

    # Check full name
    if name_lower in text_lower:
        return None

    # Check individual name parts (for multi-word names like "Brian Eno")
    parts = name_lower.split()
    if len(parts) > 1:
        # At least the surname should appear
        for part in parts:
            if len(part) > 2 and part in text_lower:
                return None

    return f"artist name '{artist_name}' not found in text"


def check_interview_markers(text: str) -> Optional[str]:
    """Return rejection reason if no interview markers found, else None.

    Only applies to interview sources — skip this for review/critic text.
    """
    text_sample = text[:5000]

    markers = [
        r'[QA]\s*[:.]',           # Q: / A: format
        r'"[^"]{20,}"',            # Substantial quoted speech
        r'\u201c[^\u201d]{20,}\u201d',  # Smart quotes
        r'\bI\s+(was|am|think|felt|wanted|started|grew|decided|believe|remember)\b',
        r'\b(tells?|says?|explains?|recalls?|describes?)\b.*\b(us|me|interviewer)\b',
        r'\b(interview|conversation|speaks?|talked?)\b',
    ]

    hits = 0
    for pattern in markers:
        if re.search(pattern, text_sample, re.IGNORECASE):
            hits += 1

    if hits < 2:
        return f"few interview markers ({hits} found)"

    return None


def prefilter(text: str, artist_name: str, source_type: str = "interview") -> tuple:
    """Run all pre-filters on text.

    Args:
        text: The scraped article text
        artist_name: Target artist name
        source_type: "interview" or "review" — controls which checks run

    Returns:
        (passed: bool, reason: str or None)
        If passed=False, reason explains why it was filtered out.
    """
    if not text:
        return False, "no text"

    # Always check length
    reason = check_length(text)
    if reason:
        return False, reason

    # Always check language
    reason = check_language(text)
    if reason:
        return False, reason

    # Always check artist mention
    reason = check_artist_mention(text, artist_name)
    if reason:
        return False, reason

    # Interview-only: check for interview markers
    if source_type == "interview":
        reason = check_interview_markers(text)
        if reason:
            return False, reason

    return True, None


def filter_sources(sources: list, artist_name: str, source_type: str = "interview") -> tuple:
    """Filter a list of source dicts, returning (passed, filtered) lists.

    Each source dict should have a 'text' key.
    """
    passed = []
    filtered = []

    for source in sources:
        text = source.get("text")
        if not text:
            filtered.append((source, "no text"))
            continue

        ok, reason = prefilter(text, artist_name, source_type)
        if ok:
            passed.append(source)
        else:
            filtered.append((source, reason))

    return passed, filtered
