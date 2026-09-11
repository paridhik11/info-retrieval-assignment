"""
positional_index.py
====================

Part C of the assignment: a **positional index** plus **exact phrase search**
and **ordered/unordered proximity (``WITHIN/k``) search**.

Conceptual structure of the index::

    {
        "term": {
            "df": <number of documents containing the term>,
            "postings": {
                "D001": {
                    "tf": <number of occurrences in that document>,
                    "positions": [p1, p2, ...]
                },
                ...
            }
        }
    }

Same pipeline, one source of truth
----------------------------------
This module does **not** tokenize, normalize, or stem on its own. It reuses
``preprocess.process_documents`` / ``preprocess.preprocess_text`` verbatim, the
exact same functions Part A's inverted index and Part B's VSM call. Positions
are read straight off the processed token list ``Document.tokens`` (the very
list Part A counted tf/df from). So a term's stem, its case-folding, its
punctuation handling, its stopword treatment, and its df/tf are guaranteed to
be identical across the inverted index and this positional index. There is no
second tokenizer with different rules — ``verify_consistency_with_inverted_index``
below asserts this against ``output/inverted_index.json``.

POSITION CONVENTION (the one explicit design decision)
------------------------------------------------------
Positions are **ZERO-BASED indices into the processed token stream**
``Document.tokens`` — i.e. the sequence *after* lowercasing, punctuation
splitting, stopword removal, and Porter stemming. ``positions[i] == p`` means
``Document.tokens[p]`` is that term.

Why zero-based indices into the *processed* stream (not the raw text):
    * The processed token list IS the sequence the indexing pipeline emits, so
      ``document.tokens[p]`` directly recovers the token at position ``p`` with
      no off-by-one translation. Using ``enumerate(document.tokens)`` to build
      the index and plain list indices to match and to display means indexing,
      phrase matching, proximity matching, and printed positions all speak the
      exact same coordinate system. Zero-based is chosen because Python lists
      are zero-based, so the stored position is literally the list index — the
      cheapest possible convention to keep consistent and to hand-verify.
    * We deliberately do NOT use raw character offsets, and we do NOT recompute
      positions from the original unprocessed string. Stopwords are dropped and
      words are stemmed, so raw-text offsets would no longer line up with the
      indexed tokens. Adjacency is defined on the *processed* stream: two terms
      are adjacent iff their processed positions differ by exactly 1, even if an
      English stopword sat between them in the surface text. This is the
      standard positional-index convention and is what makes phrase adjacency
      well-defined after stopword removal.

PHRASE SEARCH — definition
--------------------------
``phrase_search("cotton shirt")`` normalizes the phrase with the *same*
pipeline, then requires the normalized terms to occur in the given order at
*consecutive* positions: some position ``p`` holds ``cotton`` and ``p+1`` holds
``shirt``. Mere co-occurrence of both terms in a document is NOT a match.

PROXIMITY SEARCH — definition and k semantics
---------------------------------------------
``proximity_search(term1, term2, k, ordered=True)``.
``k`` is the **maximum positional difference** between the two matched tokens,
NOT the number of intervening tokens. So ``cotton WITHIN/3 shirt`` (ordered)
means ``0 < p2 - p1 <= 3`` where ``p1`` is a ``cotton`` position and ``p2`` a
``shirt`` position (adjacency is the ``k = 1`` case; ``k = 3`` allows up to two
tokens in between). Unordered uses ``0 < |p1 - p2| <= k``. Every returned match
carries the actual satisfying position pair(s) as evidence that positional
retrieval — not VSM — produced the result.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Allow ``python src/positional_index.py`` as well as package-style imports.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from preprocess import (  # noqa: E402
    Document,
    index_lookup_term,
    load_corpus,
    parse_documents,
    preprocess_text,
    process_documents,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS_PATH = REPO_ROOT / "data" / "corpus_100.txt"
DEFAULT_POSITIONAL_INDEX_PATH = REPO_ROOT / "output" / "positional_index.json"
DEFAULT_INVERTED_INDEX_PATH = REPO_ROOT / "output" / "inverted_index.json"

_DOC_ID_RE = re.compile(r"^(.*?)(\d+)$")


def _doc_id_sort_key(doc_id: str) -> tuple:
    """Ascending docID order using the numeric suffix when present (D2 < D10).

    Mirrors ``index_builder`` so postings order is identical across indexes.
    """
    match = _DOC_ID_RE.match(doc_id)
    if match:
        return (match.group(1), int(match.group(2)))
    return (doc_id, 0)


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Index construction
# --------------------------------------------------------------------------

def build_positional_index(documents: list[Document]) -> dict:
    """Build ``term -> {df, postings: {docID: {tf, positions}}}``.

    Positions are zero-based indices into each document's *processed* token
    stream ``document.tokens`` (see the module docstring). Because we enumerate
    that exact list, ``tf == len(positions)`` and ``df == len(postings)`` match
    the Part A inverted index by construction.
    """
    # term -> docID -> [positions]
    table: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for document in documents:
        # enumerate gives the zero-based position in the processed token stream.
        for position, term in enumerate(document.tokens):
            table[term][document.doc_id].append(position)

    index: dict = {}
    for term in sorted(table):  # deterministic vocabulary order
        postings: dict = {}
        for doc_id in sorted(table[term], key=_doc_id_sort_key):  # sorted by docID
            positions = sorted(table[term][doc_id])  # positions sorted ascending
            postings[doc_id] = {
                "tf": len(positions),
                "positions": positions,
            }
        index[term] = {
            "df": len(postings),
            "postings": postings,
        }
    return index


def save_positional_index(index: dict, path: str | Path) -> None:
    """Write a deterministic, key-sorted positional index JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def build_from_corpus(
    corpus_path: str | Path = DEFAULT_CORPUS_PATH,
) -> tuple[list[Document], dict]:
    """Parse + preprocess the corpus (Part A pipeline) and build the index."""
    documents = process_documents(parse_documents(load_corpus(corpus_path)))
    index = build_positional_index(documents)
    return documents, index


# --------------------------------------------------------------------------
# Verification helpers (prove the pipeline is genuinely reused)
# --------------------------------------------------------------------------

def verify_positions_match_tokens(documents: list[Document], index: dict) -> None:
    """Assert every stored position recovers its term from ``document.tokens``.

    This is the direct proof that positions are indices into the *processed*
    token stream and not raw character offsets or raw-text positions.
    """
    tokens_by_doc = {d.doc_id: d.tokens for d in documents}
    for term, entry in index.items():
        for doc_id, posting in entry["postings"].items():
            tokens = tokens_by_doc[doc_id]
            if posting["tf"] != len(posting["positions"]):
                raise AssertionError(f"{term}/{doc_id}: tf != len(positions)")
            if posting["positions"] != sorted(posting["positions"]):
                raise AssertionError(f"{term}/{doc_id}: positions not sorted")
            for p in posting["positions"]:
                if tokens[p] != term:
                    raise AssertionError(
                        f"{term}/{doc_id}: position {p} holds "
                        f"{tokens[p]!r}, not {term!r}"
                    )


def verify_consistency_with_inverted_index(
    positional_index: dict, inverted_index: dict
) -> None:
    """Assert the positional index shares Part A's vocabulary, df, and tf.

    Same terms, same df per term, and same tf per (term, doc). If the positional
    index used any different tokenizer/stemmer/stopword policy this would fail —
    which is exactly the guarantee the assignment asks for.
    """
    if set(positional_index) != set(inverted_index):
        only_pos = sorted(set(positional_index) - set(inverted_index))[:5]
        only_inv = sorted(set(inverted_index) - set(positional_index))[:5]
        raise AssertionError(
            "vocabulary mismatch vs inverted index "
            f"(positional-only e.g. {only_pos}, inverted-only e.g. {only_inv})"
        )
    for term, pos_entry in positional_index.items():
        inv_entry = inverted_index[term]
        if pos_entry["df"] != inv_entry["df"]:
            raise AssertionError(f"{term}: df differs from inverted index")
        pos_postings = pos_entry["postings"]
        inv_postings = inv_entry["postings"]
        if set(pos_postings) != set(inv_postings):
            raise AssertionError(f"{term}: posting docIDs differ")
        for doc_id, posting in pos_postings.items():
            if posting["tf"] != inv_postings[doc_id]:
                raise AssertionError(
                    f"{term}/{doc_id}: tf {posting['tf']} != "
                    f"inverted tf {inv_postings[doc_id]}"
                )


# --------------------------------------------------------------------------
# Positional index with phrase + proximity search
# --------------------------------------------------------------------------

class PositionalIndex:
    """Query interface over a positional index (phrase + proximity search)."""

    def __init__(self, index: dict) -> None:
        self.index = index

    @classmethod
    def from_files(
        cls, path: str | Path = DEFAULT_POSITIONAL_INDEX_PATH
    ) -> "PositionalIndex":
        return cls(_load_json(path))

    @classmethod
    def from_corpus(
        cls, corpus_path: str | Path = DEFAULT_CORPUS_PATH
    ) -> "PositionalIndex":
        _documents, index = build_from_corpus(corpus_path)
        return cls(index)

    # -- lookup helpers ----------------------------------------------------

    def _postings(self, term: str) -> dict | None:
        """Postings dict ``{docID: {tf, positions}}`` for a *normalized* term."""
        entry = self.index.get(term)
        return None if entry is None else entry["postings"]

    def _positions(self, term: str, doc_id: str) -> list[int]:
        postings = self._postings(term)
        if postings is None or doc_id not in postings:
            return []
        return postings[doc_id]["positions"]

    # -- phrase search -----------------------------------------------------

    def phrase_search(self, phrase: str) -> list[dict]:
        """Exact ordered, consecutive-position phrase search.

        The phrase is normalized with the *same* pipeline as the corpus. For an
        n-term phrase we require a position ``p`` such that term ``i`` occupies
        ``p + i`` for every ``i`` (0..n-1). Returns::

            [{"docID": "D...", "matches": [[p, p+1, ...], ...]}, ...]

        where each ``matches`` entry is the actual consecutive positions that
        satisfy the phrase (the evidence). Documents are sorted by docID and
        each document's matches are sorted by start position.

        Edge cases:
          * empty / stopword-only / punctuation-only phrase -> ``[]``.
          * any term absent from the vocabulary -> ``[]`` (the phrase cannot
            occur), even if the other terms exist.
          * a one-term phrase -> every occurrence position, as a length-1 match.
          * repeated terms (e.g. "cotton cotton") are handled naturally: the
            same postings list is probed at ``p`` and ``p+1``.
        """
        terms = preprocess_text(phrase)
        if not terms:  # empty / stopword-only / punctuation-only
            return []

        # Resolve postings once per query term. A single unknown term means the
        # exact phrase is impossible, so we return nothing.
        postings_per_term: list[dict] = []
        for term in terms:
            postings = self._postings(term)
            if postings is None:
                return []
            postings_per_term.append(postings)

        # Candidate documents must contain *every* term (intersection of docs).
        candidate_docs = set(postings_per_term[0])
        for postings in postings_per_term[1:]:
            candidate_docs &= set(postings)

        results: list[dict] = []
        for doc_id in sorted(candidate_docs, key=_doc_id_sort_key):
            # Sets give O(1) "is term i at p+i?" checks; the first term drives.
            position_sets = [
                set(postings[doc_id]["positions"]) for postings in postings_per_term
            ]
            first_positions = postings_per_term[0][doc_id]["positions"]
            matches: list[list[int]] = []
            for p in first_positions:  # already sorted -> matches stay sorted
                if all(
                    (p + offset) in position_sets[offset]
                    for offset in range(1, len(terms))
                ):
                    matches.append([p + offset for offset in range(len(terms))])
            if matches:
                results.append({"docID": doc_id, "matches": matches})
        return results

    # -- proximity search --------------------------------------------------

    def proximity_search(
        self, term1: str, term2: str, k: int, ordered: bool = True
    ) -> list[dict]:
        """Positional proximity search within ``k`` *positions*.

        ``k`` is the MAXIMUM POSITIONAL DIFFERENCE, not the count of intervening
        tokens. Both query terms are normalized with the shared pipeline.

          * ordered   (default): keep pairs with ``0 < p2 - p1 <= k``
                                  (term1 before term2, gap at most k).
          * unordered:           keep pairs with ``0 < |p1 - p2| <= k``.

        Returns::

            [{"docID": "D...", "pairs": [[p1, p2], ...]}, ...]

        listing the actual satisfying (p1, p2) position pairs as evidence.
        Returns ``[]`` when either term is empty/stopword-only, unknown to the
        vocabulary, or ``k < 1``.
        """
        if k < 1:
            return []
        t1 = index_lookup_term(term1)
        t2 = index_lookup_term(term2)
        if t1 is None or t2 is None:
            return []

        postings1 = self._postings(t1)
        postings2 = self._postings(t2)
        if postings1 is None or postings2 is None:
            return []

        candidate_docs = sorted(
            set(postings1) & set(postings2), key=_doc_id_sort_key
        )
        results: list[dict] = []
        for doc_id in candidate_docs:
            positions1 = postings1[doc_id]["positions"]
            positions2 = postings2[doc_id]["positions"]
            pairs: list[list[int]] = []
            for p1 in positions1:
                for p2 in positions2:
                    if ordered:
                        if 0 < p2 - p1 <= k:
                            pairs.append([p1, p2])
                    else:
                        if p1 != p2 and abs(p1 - p2) <= k:
                            pairs.append([p1, p2])
            if pairs:
                pairs.sort()
                results.append({"docID": doc_id, "pairs": pairs})
        return results


# --------------------------------------------------------------------------
# Module-level convenience API (one lazily loaded index, mirrors vsm.py)
# --------------------------------------------------------------------------

_INDEX: PositionalIndex | None = None


def get_index() -> PositionalIndex:
    """Return a cached ``PositionalIndex`` built fresh from the corpus.

    Built from the corpus (not the JSON) so the query API always reflects the
    same live preprocessing pipeline as the rest of the system.
    """
    global _INDEX
    if _INDEX is None:
        _INDEX = PositionalIndex.from_corpus()
    return _INDEX


def phrase_search(phrase: str) -> list[dict]:
    """Convenience wrapper around ``PositionalIndex.phrase_search``."""
    return get_index().phrase_search(phrase)


def proximity_search(
    term1: str, term2: str, k: int, ordered: bool = True
) -> list[dict]:
    """Convenience wrapper around ``PositionalIndex.proximity_search``."""
    return get_index().proximity_search(term1, term2, k, ordered=ordered)


# --------------------------------------------------------------------------
# Test / demo:  python src/positional_index.py
# --------------------------------------------------------------------------

def _print_sample_postings(index: dict, raw_terms: list[str]) -> None:
    print("sample positional postings:")
    for raw in raw_terms:
        key = index_lookup_term(raw)
        entry = index.get(key) if key else None
        print(f"  '{raw}' -> stem '{key}'")
        if entry is None:
            print("    (not in vocabulary)")
            continue
        print(f"    df = {entry['df']}")
        for doc_id, posting in list(entry["postings"].items())[:3]:
            print(
                f"    {doc_id}: tf={posting['tf']} "
                f"positions={posting['positions']}"
            )


def _demo_phrase(model: PositionalIndex, phrase: str) -> list[dict]:
    terms = preprocess_text(phrase)
    results = model.phrase_search(phrase)
    print(f"  phrase {phrase!r}")
    print(f"    normalized terms : {terms}")
    if not results:
        print("    matches          : (none)")
        return results
    doc_ids = [r["docID"] for r in results]
    print(f"    matching docIDs  : {doc_ids}")
    for r in results[:5]:
        print(f"      {r['docID']}: positions {r['matches']}")
    return results


def _demo_proximity(
    model: PositionalIndex, t1: str, t2: str, k: int, ordered: bool = True
) -> list[dict]:
    results = model.proximity_search(t1, t2, k, ordered=ordered)
    tag = "ordered" if ordered else "unordered"
    print(f"  {t1} WITHIN/{k} {t2}  ({tag}; k = max positional difference)")
    print(f"    normalized terms : {[index_lookup_term(t1), index_lookup_term(t2)]}")
    if not results:
        print("    matches          : (none)")
        return results
    print(f"    matching docIDs  : {[r['docID'] for r in results]}")
    for r in results[:5]:
        print(f"      {r['docID']}: pairs {r['pairs']}")
    return results


def _verify_consecutive(phrase_results: list[dict], phrase: str) -> None:
    """Assert at least one reported phrase match uses consecutive positions."""
    for r in phrase_results:
        for match in r["matches"]:
            consecutive = all(
                match[i + 1] - match[i] == 1 for i in range(len(match) - 1)
            )
            if consecutive and len(match) > 1:
                print(
                    f"  VERIFIED: phrase {phrase!r} truly matches consecutive "
                    f"positions in {r['docID']}: {match} "
                    f"(each step +1)"
                )
                return
    print(f"  (no multi-term consecutive match found for {phrase!r})")


def _demo_cooccurrence_vs_phrase(model: PositionalIndex, phrase: str) -> None:
    """Show a doc where BOTH terms occur but the exact phrase does NOT match."""
    terms = preprocess_text(phrase)
    if len(terms) < 2:
        return
    postings = [model._postings(t) for t in terms]
    if any(p is None for p in postings):
        print(f"  a term of {phrase!r} is unknown; cannot compare co-occurrence")
        return
    cooccur = set(postings[0])
    for p in postings[1:]:
        cooccur &= set(p)
    phrase_docs = {r["docID"] for r in model.phrase_search(phrase)}
    cooccur_only = sorted(cooccur - phrase_docs, key=_doc_id_sort_key)
    print(f"  co-occurrence vs exact phrase for {phrase!r}:")
    print(f"    docs containing all terms somewhere : {len(cooccur)}")
    print(f"    docs where phrase actually matches  : {len(phrase_docs)}")
    if not cooccur_only:
        print(
            "    every co-occurrence doc is also a true phrase match, so for "
            "this phrase co-occurrence and phrase matching coincide."
        )
        return
    example = cooccur_only[0]
    print(
        f"    EXAMPLE {example}: contains all of {terms} but NOT as the phrase."
    )
    for t in terms:
        print(f"      {t!r} at positions {model._positions(t, example)}")
    print(
        "    -> a plain boolean/VSM 'both terms present' test would wrongly "
        f"return {example}; positional phrase search correctly excludes it."
    )


def main() -> None:
    documents, index = build_from_corpus()

    # 1) Persist the deterministic positional index.
    save_positional_index(index, DEFAULT_POSITIONAL_INDEX_PATH)

    # 2) Prove the same preprocessing pipeline is reused.
    verify_positions_match_tokens(documents, index)
    try:
        inverted = _load_json(DEFAULT_INVERTED_INDEX_PATH)
        verify_consistency_with_inverted_index(index, inverted)
        pipeline_note = (
            "positional index matches output/inverted_index.json "
            "(same vocabulary, df, and tf)"
        )
    except FileNotFoundError:
        pipeline_note = (
            "inverted_index.json not found; run index_builder.py to cross-check"
        )

    # Determinism: rebuilding yields identical JSON.
    _docs2, index2 = build_from_corpus()
    if json.dumps(index, sort_keys=True) != json.dumps(index2, sort_keys=True):
        raise AssertionError("positional index is not deterministic")

    print(f"documents indexed  : {len(documents)}")
    print(f"vocabulary size    : {len(index)}")
    print(f"wrote              : {DEFAULT_POSITIONAL_INDEX_PATH}")
    print(f"position convention: ZERO-BASED index into processed token stream")
    print(f"pipeline check     : {pipeline_note}")
    print("=" * 70)

    model = PositionalIndex(index)

    _print_sample_postings(index, ["cotton", "shirt", "denim"])
    print("=" * 70)

    # 3) Phrase search on the assignment examples.
    print("PHRASE SEARCH (exact, consecutive positions):")
    phrase_examples = [
        "cotton shirt",
        "stretch denim",
        "festive wear",
        "winter wear",
        "regular fit",
    ]
    phrase_results = {p: _demo_phrase(model, p) for p in phrase_examples}
    print("=" * 70)

    # 4) Proximity search on the assignment examples.
    print("PROXIMITY SEARCH (WITHIN/k, k = max positional difference):")
    _demo_proximity(model, "cotton", "shirt", 3)
    _demo_proximity(model, "stretch", "denim", 4)
    _demo_proximity(model, "winter", "wear", 3)
    _demo_proximity(model, "festive", "kurta", 4)
    print("=" * 70)

    # 5) Manual verification requirements.
    print("VERIFICATION:")
    for phrase in ("cotton shirt", "stretch denim", "regular fit"):
        _verify_consecutive(phrase_results[phrase], phrase)
    print("-" * 70)
    _demo_cooccurrence_vs_phrase(model, "cotton shirt")
    print("=" * 70)

    # 6) Edge cases must not crash.
    print("EDGE CASES (must not crash):")
    edge = {
        "empty phrase": model.phrase_search(""),
        "stopword-only phrase": model.phrase_search("the and of"),
        "unknown term in phrase": model.phrase_search("cotton zzznotaword"),
        "one-term phrase": model.phrase_search("denim"),
        "repeated-term phrase": model.phrase_search("cotton cotton"),
        "proximity unknown term": model.proximity_search("cotton", "zzz", 3),
        "proximity k=0": model.proximity_search("cotton", "shirt", 0),
        "proximity unordered": model.proximity_search("wear", "winter", 3, ordered=False),
    }
    for name, res in edge.items():
        preview = res[:1] if res else res
        print(f"  {name:24s}: {len(res)} result(s) e.g. {preview}")


if __name__ == "__main__":
    main()
