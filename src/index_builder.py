"""
index_builder.py
================

Part A: inverted index (df + postings) over the preprocessed clothing corpus.

The index is conceptually:

    {
        "term": {
            "df": <number of distinct documents containing the term>,
            "postings": {
                "D001": <tf in that document>,
                ...
            }
        }
    }

df is document frequency, *not* collection frequency (sum of tfs). Every
posting stores at least docID (the key) and tf (the value). Vocabulary keys
and posting docIDs are sorted so the JSON is deterministic across runs.

This module reuses ``preprocess.preprocess_text`` rather than tokenizing
again, so later VSM / positional / query code can share the same terms.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Allow ``python src/index_builder.py`` as well as package-style imports.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from preprocess import (  # noqa: E402
    Document,
    index_lookup_term,
    load_corpus,
    parse_documents,
    process_documents,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS_PATH = REPO_ROOT / "data" / "corpus_100.txt"
DEFAULT_INDEX_PATH = REPO_ROOT / "output" / "inverted_index.json"
DEFAULT_METADATA_PATH = REPO_ROOT / "output" / "doc_metadata.json"

# Assignment collection size. We check the *count* of parsed records; we do
# not hard-code the 100 DOCID strings.
EXPECTED_N = 100

_DOC_ID_RE = re.compile(r"^(.*?)(\d+)$")


def _doc_id_sort_key(doc_id: str) -> tuple:
    """Ascending docID order using the numeric suffix when present (D2 < D10)."""
    match = _DOC_ID_RE.match(doc_id)
    if match:
        return (match.group(1), int(match.group(2)))
    return (doc_id, 0)


def build_document_store(documents: list[Document]) -> dict[str, dict]:
    """
    Reusable docID -> metadata map for later display / query modules.

    Original title/category/text are kept verbatim. Tokens are stored so the
    processed corpus still has an entry for every document, even if a
    document yielded no index terms.
    """
    store: dict[str, dict] = {}
    for document in sorted(documents, key=lambda d: _doc_id_sort_key(d.doc_id)):
        store[document.doc_id] = {
            "category": document.category,
            "title": document.title,
            "text": document.text,
            "tokens": document.tokens,
        }
    return store


def build_inverted_index(documents: list[Document]) -> dict:
    """
    Build term -> {df, postings: {docID: tf}} from preprocessed documents.

    tf is the number of times the *already normalized/stemmed* token occurs
    in that document's TITLE+TEXT token list. df counts distinct documents,
    not total occurrences.
    """
    tf_table: dict[str, Counter] = defaultdict(Counter)
    for document in documents:
        counts = Counter(document.tokens)
        for term, tf in counts.items():
            tf_table[term][document.doc_id] = tf

    index: dict = {}
    for term in sorted(tf_table):  # sort for deterministic JSON output
        postings = tf_table[term]
        # df = number of distinct documents containing this term (posting count).
        # This is NOT collection frequency (sum of tfs).
        ordered = {
            doc_id: int(postings[doc_id])
            for doc_id in sorted(postings, key=_doc_id_sort_key)
        }
        index[term] = {
            "df": len(ordered),
            "postings": ordered,
        }
    return index


def save_index(index: dict, path: str | Path) -> None:
    """Write a human-readable, key-sorted inverted index JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def save_metadata(store: dict, path: str | Path) -> None:
    """Write the docID -> title/category/text(/tokens) store for later modules."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(store, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def validate_documents(documents: list[Document]) -> None:
    """Fail fast on the assignment invariants: N=100, unique IDs, complete fields."""
    n = len(documents)
    if n != EXPECTED_N:
        raise ValueError(f"expected {EXPECTED_N} documents, parsed {n}")

    ids = [document.doc_id for document in documents]
    if any(not doc_id for doc_id in ids):
        raise ValueError("one or more documents are missing a DOCID")

    duplicates = sorted({doc_id for doc_id in ids if ids.count(doc_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate DOCIDs: {duplicates}")

    missing: list[str] = []
    for document in documents:
        for field_name, value in (
            ("CATEGORY", document.category),
            ("TITLE", document.title),
            ("TEXT", document.text),
        ):
            if not value:
                missing.append(f"{document.doc_id}:{field_name}")
    if missing:
        raise ValueError(f"unexpected empty fields: {missing}")


def verify_index_statistics(documents: list[Document], index: dict) -> None:
    """
    Recompute tf/df from the token lists and require an exact match.

    This catches the classic mix-up of collection frequency (sum of tfs)
    with document frequency (number of postings).
    """
    expected: dict[str, Counter] = defaultdict(Counter)
    for document in documents:
        for term, tf in Counter(document.tokens).items():
            expected[term][document.doc_id] = tf

    if set(index) != set(expected):
        raise AssertionError("index vocabulary does not match tokenized corpus")

    for term, entry in index.items():
        postings = entry["postings"]
        if entry["df"] != len(postings):
            raise AssertionError(
                f"{term}: df={entry['df']} but {len(postings)} postings"
            )
        if entry["df"] != len(expected[term]):
            raise AssertionError(f"{term}: stored df does not match distinct docs")
        if list(postings) != sorted(postings, key=_doc_id_sort_key):
            raise AssertionError(f"{term}: postings are not sorted by docID")
        for doc_id, tf in postings.items():
            if tf < 1:
                raise AssertionError(f"{term} in {doc_id}: tf must be >= 1")
            if tf != expected[term][doc_id]:
                raise AssertionError(
                    f"{term} in {doc_id}: stored tf={tf}, "
                    f"counted tf={expected[term][doc_id]}"
                )


def example_postings(index: dict, raw_term: str) -> dict | None:
    """Return the index entry for a surface term, after the shared pipeline."""
    key = index_lookup_term(raw_term)
    if key is None:
        return None
    return index.get(key)


def _print_example(index: dict, raw_term: str) -> None:
    key = index_lookup_term(raw_term)
    print(f"  '{raw_term}' -> stemmed index key '{key}'")
    if key is None:
        print("    (term vanished under the stopword/punctuation rules)")
        return
    entry = index.get(key)
    if entry is None:
        print("    not in vocabulary")
        return
    print(f"    df = {entry['df']}")
    print(f"    postings = {entry['postings']}")


def build_from_corpus(
    corpus_path: str | Path = DEFAULT_CORPUS_PATH,
) -> tuple[list[Document], dict, dict]:
    """Parse, preprocess, validate, and index the clothing corpus."""
    raw = load_corpus(corpus_path)
    documents = parse_documents(raw)
    validate_documents(documents)
    process_documents(documents)
    index = build_inverted_index(documents)
    verify_index_statistics(documents, index)
    store = build_document_store(documents)
    if len(store) != EXPECTED_N:
        raise ValueError("processed corpus does not contain all parsed documents")
    return documents, index, store


def main() -> None:
    documents, index, store = build_from_corpus()

    save_index(index, DEFAULT_INDEX_PATH)
    save_metadata(store, DEFAULT_METADATA_PATH)

    # Determinism: rebuilding must produce identical JSON text.
    documents_again, index_again, _ = build_from_corpus()
    if json.dumps(index, sort_keys=True) != json.dumps(index_again, sort_keys=True):
        raise AssertionError("preprocessing / indexing is not deterministic")
    if [d.tokens for d in documents] != [d.tokens for d in documents_again]:
        raise AssertionError("token streams are not deterministic")

    print(f"documents parsed N = {len(documents)}")
    print(f"unique DOCIDs     = {len(store)}")
    print(f"vocabulary size   = {len(index)}")
    print(f"indexed terms     = {len(index)}")
    print(f"wrote {DEFAULT_INDEX_PATH}")
    print(f"wrote {DEFAULT_METADATA_PATH}")
    print("example postings:")
    for term in ("cotton", "denim", "kurta"):
        _print_example(index, term)


if __name__ == "__main__":
    main()
