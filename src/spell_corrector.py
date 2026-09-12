"""
spell_corrector.py
==================

Optional enhancement: conservative, vocabulary-based spelling correction.

This module corrects misspelled query terms by comparing each unknown token
against the indexed vocabulary using Levenshtein (edit) distance.  It is a
*conservative*, *deterministic*, *vocabulary-anchored* corrector that:

  * Uses the same ``preprocess_text`` pipeline as the rest of the project.
  * Reads the vocabulary exclusively from the existing inverted index
    (``output/inverted_index.json``).  No external API, LLM, embedding model,
    or semantic search service is used.
  * Leaves known terms exactly unchanged.
  * Only attempts correction when a term is absent from the vocabulary.
  * Applies a conservative maximum edit-distance threshold that grows with
    term length to avoid over-correcting short or ambiguous tokens.
  * Breaks ties among equally close candidates by document frequency
    (higher df preferred), then alphabetically (deterministic).
  * Leaves a term unchanged if no candidate within the threshold exists.

Conservative distance thresholds
---------------------------------
  term length 1–3 : max distance 1  (very short terms corrected only if
                                      exactly one edit away from a vocab word)
  term length 4–6 : max distance 2
  term length  7+ : max distance 2  (distance 3 is too liberal for a
                                      100-document clothing vocabulary)

Why "distance 2 for 7+" and not 3
----------------------------------
The clothing vocabulary is small and domain-specific.  At distance 3 a short
token like ``den`` has far too many plausible matches, and long query words
(> 6 chars) are long enough that real misspellings are usually within 2 edits
(one transposition + one insertion, or two substitutions).  Distance 3 is
therefore enabled at length > 6 only as a *configurable* fallback via
``_max_distance``, but the default keeps it at 2 for safety.

Public API
----------
    correct_query(query_string, *, enabled=True) -> dict

    Returns::

        {
            "original_query": str,     # unchanged raw input
            "corrected_query": str,    # query with replacements applied
            "corrections": [           # only entries where a replacement happened
                {
                    "original": str,   # surface token as it appeared in the query
                    "replacement": str,# the vocabulary stem chosen
                    "distance": int,   # edit distance between original stem and replacement
                },
                ...
            ],
            "changed": bool,           # True iff at least one correction was made
        }

When ``enabled=False`` the function returns immediately with ``changed=False``
and ``corrected_query == original_query`` — exactly preserving the original
search behaviour.

The correction prepares a query string.  It must not alter ranking scores,
document weights, IDF, cosine normalization, or any retrieval formula.  Pass
the ``corrected_query`` into the existing retrieval pipeline unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow ``python src/spell_corrector.py`` as well as package-style imports.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from preprocess import preprocess_text  # noqa: E402

REPO_ROOT = _SRC_DIR.parent
DEFAULT_INDEX_PATH = REPO_ROOT / "output" / "inverted_index.json"


# ---------------------------------------------------------------------------
# Levenshtein (edit) distance — implemented directly in Python.
# ---------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    """Compute the standard Levenshtein edit distance between strings *a* and *b*.

    Uses a classic two-row DP (Wagner–Fischer) with O(min(len(a), len(b)))
    memory.  The result is the minimum number of single-character insertions,
    deletions, or substitutions to transform *a* into *b*.

    Properties relevant to this module:
    * Symmetric: ``_levenshtein(a, b) == _levenshtein(b, a)``.
    * Returns 0 iff ``a == b``.
    * Deterministic for any fixed (a, b) — no random state.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Ensure |a| <= |b| for the memory-efficient row swap.
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for j, cb in enumerate(b, start=1):
        curr = [j] + [0] * len(a)
        for i, ca in enumerate(a, start=1):
            if ca == cb:
                curr[i] = prev[i - 1]
            else:
                curr[i] = 1 + min(prev[i - 1], prev[i], curr[i - 1])
        prev = curr
    return prev[len(a)]


# ---------------------------------------------------------------------------
# Conservative threshold
# ---------------------------------------------------------------------------

def _max_distance(term_length: int) -> int:
    """Return the maximum allowed edit distance for a term of *term_length* characters.

    Conservative by design:
    * 1–3 chars  → distance 1  (extremely short; only obvious single-edit fixes)
    * 4–6 chars  → distance 2
    * 7+  chars  → distance 2  (kept at 2; 3 is too liberal for small vocabulary)
    """
    if term_length <= 3:
        return 1
    if term_length <= 6:
        return 2
    return 2


# ---------------------------------------------------------------------------
# Vocabulary loading
# ---------------------------------------------------------------------------

def _load_vocabulary(index_path: str | Path = DEFAULT_INDEX_PATH) -> dict[str, int]:
    """Load the inverted index and return ``{stem: df}`` for every indexed term.

    The vocabulary is read from the same file that ``VectorSpaceModel`` uses,
    so the two modules always agree on which terms are known.
    """
    path = Path(index_path)
    raw: dict = json.loads(path.read_text(encoding="utf-8"))
    return {term: entry["df"] for term, entry in raw.items()}


# ---------------------------------------------------------------------------
# Module-level cached vocabulary (loaded once per process).
# ---------------------------------------------------------------------------

_VOCABULARY: dict[str, int] | None = None


def get_vocabulary(index_path: str | Path = DEFAULT_INDEX_PATH) -> dict[str, int]:
    """Return the lazily loaded vocabulary ``{stem: df}``.

    The first call reads the index from disk; subsequent calls reuse the
    in-memory cache.  Pass *index_path* to override the default (useful in
    tests that build a temporary index).
    """
    global _VOCABULARY
    if _VOCABULARY is None:
        _VOCABULARY = _load_vocabulary(index_path)
    return _VOCABULARY


def reload_vocabulary(index_path: str | Path = DEFAULT_INDEX_PATH) -> dict[str, int]:
    """Force a fresh load of the vocabulary (clears the cache).

    Call this if the inverted index has been rebuilt since the module was
    first imported (e.g. in tests).
    """
    global _VOCABULARY
    _VOCABULARY = _load_vocabulary(index_path)
    return _VOCABULARY


# ---------------------------------------------------------------------------
# Single-term correction
# ---------------------------------------------------------------------------

def _best_candidate(
    stem: str,
    vocabulary: dict[str, int],
) -> tuple[str, int] | None:
    """Find the best vocabulary replacement for *stem*, or return ``None``.

    Candidate ranking (lexicographically in priority order):
      1. Smallest edit distance (closest match first).
      2. Highest document frequency (more common vocabulary term preferred).
      3. Alphabetical order (deterministic tie-break).

    Returns ``(replacement_stem, distance)`` if a safe candidate exists,
    or ``None`` if no vocabulary term is within the conservative threshold.

    Known terms (``stem in vocabulary``) are never passed here; the caller
    skips them before calling this function.
    """
    threshold = _max_distance(len(stem))

    best_term: str | None = None
    best_dist: int = threshold + 1  # sentinel: one beyond the allowed max
    best_df: int = -1

    for vocab_term, df in vocabulary.items():
        dist = _levenshtein(stem, vocab_term)
        if dist > threshold:
            continue
        # Prefer: smaller distance > higher df > alphabetical.
        if dist < best_dist:
            best_dist = dist
            best_term = vocab_term
            best_df = df
        elif dist == best_dist:
            if df > best_df or (df == best_df and vocab_term < best_term):  # type: ignore[operator]
                best_term = vocab_term
                best_df = df

    if best_term is None:
        return None
    return (best_term, best_dist)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def correct_query(
    query_string: str,
    *,
    enabled: bool = True,
    index_path: str | Path = DEFAULT_INDEX_PATH,
) -> dict:
    """Correct misspelled terms in *query_string* using the indexed vocabulary.

    Parameters
    ----------
    query_string:
        The raw query as the user typed it.  Whitespace is preserved in
        ``corrected_query`` so the display shows the user-facing string.
    enabled:
        When ``False`` the function returns immediately with no changes —
        identical to the original search behavior (baseline is preserved).
    index_path:
        Path to the inverted index JSON.  Defaults to the project's
        ``output/inverted_index.json``.  Override in tests.

    Returns
    -------
    A dict with keys:

    ``original_query`` (str)
        The unmodified input.

    ``corrected_query`` (str)
        The query with misspelled tokens replaced by their closest vocabulary
        match.  Equal to *original_query* when no corrections were made (or
        when *enabled* is ``False``).

    ``corrections`` (list[dict])
        One entry per *corrected* token::

            {"original": <surface token>, "replacement": <vocab stem>, "distance": int}

        Empty when no corrections were made.

    ``changed`` (bool)
        ``True`` iff at least one correction was applied.

    Algorithm
    ---------
    1. Split *query_string* on whitespace to preserve surface tokens for
       display.
    2. Preprocess each surface token individually through ``preprocess_text``
       to obtain its stem(s).  A single surface word may produce zero stems
       (stop-word / punctuation) or one stem; multi-stem results are treated
       as a single sequence.
    3. For each surface token whose stem(s) are *all* unknown to the
       vocabulary, attempt correction.
    4. A token whose preprocessed stem is already in the vocabulary is left
       unchanged (known term preservation).
    5. Correction candidates are found via ``_best_candidate``; if none exists
       within the conservative threshold, the original token is kept.
    6. Replacements are applied at the surface-token level in the reconstructed
       query string so the output is human-readable.
    """
    original_query = query_string

    if not enabled or not query_string.strip():
        return {
            "original_query": original_query,
            "corrected_query": original_query,
            "corrections": [],
            "changed": False,
        }

    vocabulary = get_vocabulary(index_path)

    surface_tokens = query_string.split()
    corrected_tokens: list[str] = []
    corrections: list[dict] = []

    for surface in surface_tokens:
        stems = preprocess_text(surface)
        if not stems:
            # Stop-word or punctuation-only token — keep as-is (nothing to correct).
            corrected_tokens.append(surface)
            continue

        # Check whether ALL stems of this surface token are already known.
        all_known = all(stem in vocabulary for stem in stems)
        if all_known:
            corrected_tokens.append(surface)
            continue

        # Collect stems that are unknown and attempt correction for them.
        # In practice ``preprocess_text`` of a single word usually produces
        # one stem; we handle the multi-stem case by correcting each unknown
        # stem independently.
        replacement_parts: list[str] = []
        corrected_this_token = False
        for stem in stems:
            if stem in vocabulary:
                replacement_parts.append(stem)
            else:
                result = _best_candidate(stem, vocabulary)
                if result is not None:
                    replacement_stem, dist = result
                    replacement_parts.append(replacement_stem)
                    corrections.append(
                        {
                            "original": surface,
                            "replacement": replacement_stem,
                            "distance": dist,
                        }
                    )
                    corrected_this_token = True
                else:
                    # No safe candidate — leave this stem (and therefore the
                    # surface token) unchanged.
                    replacement_parts.append(stem)

        if corrected_this_token:
            # Represent the corrected token(s) as a space-joined sequence of
            # stems (they are already in normalized/stemmed form).
            corrected_tokens.append(" ".join(replacement_parts))
        else:
            corrected_tokens.append(surface)

    corrected_query = " ".join(corrected_tokens)
    changed = len(corrections) > 0

    return {
        "original_query": original_query,
        "corrected_query": corrected_query,
        "corrections": corrections,
        "changed": changed,
    }


# ---------------------------------------------------------------------------
# Module-level convenience wrappers
# ---------------------------------------------------------------------------

def is_known_term(surface_token: str, index_path: str | Path = DEFAULT_INDEX_PATH) -> bool:
    """Return ``True`` iff every preprocessed stem of *surface_token* is in the vocabulary."""
    vocabulary = get_vocabulary(index_path)
    stems = preprocess_text(surface_token)
    return bool(stems) and all(stem in vocabulary for stem in stems)


# ---------------------------------------------------------------------------
# Demo / manual verification:  python src/spell_corrector.py
# ---------------------------------------------------------------------------

def _demo() -> None:
    """Print correction results for a few example queries."""
    demo_queries = [
        "cottn shirt",
        "denimm jeans",
        "cotton shirt",
        "winter jackt",
        "high waist leggngs",
        "zxqwerty fabric",
        "the and of",
    ]
    vocab = get_vocabulary()
    print(f"Vocabulary size: {len(vocab)} stems (from inverted index)")
    print()
    for q in demo_queries:
        result = correct_query(q)
        print(f"Query:     {q!r}")
        if result["changed"]:
            print(f"Corrected: {result['corrected_query']!r}")
            for c in result["corrections"]:
                print(
                    f"  {c['original']!r} -> {c['replacement']!r} "
                    f"(distance {c['distance']})"
                )
        else:
            print("  (no corrections)")
        print()


if __name__ == "__main__":
    _demo()
