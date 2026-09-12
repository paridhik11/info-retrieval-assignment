"""
bm25.py
=======

OPTIONAL experimental extension (NOT part of the required assignment).

A **second classical IR ranking model — Okapi BM25 — implemented directly** so
its arithmetic is transparent and hand-checkable in a viva. This module exists
**only to compare** two classical ranking approaches on the same 100-document
clothing corpus:

    1. Required baseline : lnc.ltc Vector Space Model  (``src/vsm.py``)
    2. Optional extension : BM25                       (this file)

It does **not** replace, modify, or import over ``query_vsm``. The required
lnc.ltc implementation remains the authoritative baseline; BM25 is layered
beside it and reuses the *same* Part A inverted index, document statistics, and
preprocessing pipeline so the two models are compared on identical inputs.

No BM25 library is used. Nothing here hides the formula.

The exact scoring formula implemented
-------------------------------------
For a document ``D`` and query ``Q``::

    BM25(D, Q) = sum over query terms t of
        IDF(t) * ( tf(t,D) * (k1 + 1) )
                 / ( tf(t,D) + k1 * (1 - b + b * |D| / avgdl) )

where

    tf(t,D) = raw term frequency of t in D, read straight from the Part A
              inverted index postings (``index[t]["postings"][D]``).
    |D|     = length of D = the number of PROCESSED tokens in D (i.e. the
              length of the normalized/stemmed token sequence the whole IR
              system indexes — NOT raw characters, NOT title/category handled
              differently from the body).
    avgdl   = average processed document length over all N = 100 documents.
    k1, b   = BM25 free parameters (defaults k1 = 1.5, b = 0.75), exposed as
              configurable function arguments rather than buried constants.

The exact IDF formula selected
------------------------------
We use the standard "BM25+1" / Lucene-style probabilistic IDF::

    IDF(t) = ln( (N - df_t + 0.5) / (df_t + 0.5) + 1 )

with

    N    = 100                (collection size, fixed by the assignment)
    df_t = document frequency of t, read from the inverted index.

Why this particular IDF: the classic Robertson/Sparck-Jones probabilistic IDF
``ln((N - df + 0.5)/(df + 0.5))`` can go *negative* for terms that appear in
more than half the collection (df > N/2), which can subtract from a document's
score. Adding ``+ 1`` inside the logarithm (the form used by Lucene/Elastic-
search) keeps the argument > 1, so ``IDF(t) > 0`` for every term and no term
ever penalizes a document. This is the standard, widely deployed BM25 IDF.

Log base note (why the ranking is unaffected)
---------------------------------------------
We use the natural logarithm ``ln`` for IDF, which is the standard base for
BM25 (Robertson & Zaragoza; Lucene). ``lnc.ltc`` in ``vsm.py`` uses ``log10``.
The base only multiplies *every* BM25 IDF (and therefore every BM25 score) by
the same positive constant ``1/ln(10)``, so it cannot change the BM25 ranking
order — it only rescales the numbers. The two models' raw scores are on
different scales and are **not** directly comparable anyway (cosine similarity
in [0, 1] vs. an unbounded BM25 sum); only the *rankings* are compared.

How BM25 differs from lnc.ltc (the reason it is worth comparing)
---------------------------------------------------------------
* Term frequency: lnc.ltc uses ``1 + log10(tf)`` (log growth, unbounded).
  BM25 uses ``tf*(k1+1) / (tf + k1*(...))`` which **saturates**: extra
  occurrences of a term give ever-smaller gains, asymptotically approaching
  ``(k1 + 1)`` times the length factor. ``k1`` controls how fast tf saturates.
* Length normalization: lnc.ltc divides the whole document vector by its
  Euclidean (cosine) norm. BM25 instead scales tf by
  ``(1 - b + b*|D|/avgdl)``, i.e. it compares each document's length to the
  *average* length; ``b`` controls how strongly length is penalized
  (``b = 0`` = no length normalization, ``b = 1`` = full).
* idf: lnc.ltc applies ``log10(N/df)`` once, on the query side. BM25 applies a
  probabilistic ``ln((N-df+0.5)/(df+0.5)+1)`` per query term inside the sum.
* Query weighting: lnc.ltc weights the query with ``(1+log10(tf_q))*idf`` and
  cosine-normalizes it. The BM25 form used here has **no query-tf factor**
  (the written formula sums each query term once), so repeated query terms are
  collapsed to a set — a deliberate, documented choice matching the formula.

Meaning of the parameters
-------------------------
* ``k1`` (default 1.5): term-frequency saturation. Larger ``k1`` lets tf keep
  mattering for longer before saturating; ``k1 = 0`` reduces the tf factor to a
  constant (binary "term present"), so only IDF distinguishes documents.
* ``b`` (default 0.75): document-length normalization strength. ``b = 0`` turns
  length normalization off entirely; ``b = 1`` fully normalizes by
  ``|D|/avgdl``. 0.75 is the standard default.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# Allow ``python src/bm25.py`` as well as package-style imports.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from preprocess import preprocess_text  # noqa: E402  (shared Part A pipeline)
# Reuse the VSM's collection size, docID ordering, and default index/metadata
# paths so BM25 and lnc.ltc score the *exact same* corpus and inputs.
from vsm import (  # noqa: E402
    DEFAULT_INDEX_PATH,
    DEFAULT_METADATA_PATH,
    N,
    _doc_id_sort_key,
    _load_json,
)

# Standard, reasonable BM25 parameters. Exposed as configurable function
# arguments below; these are only the defaults.
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


class BM25Model:
    """Okapi BM25 ranked retrieval over the Part A inverted index.

    Document lengths and the average document length are precomputed once at
    construction time (they do not depend on the query, only on the corpus), so
    answering a query only touches the postings of the query terms — mirroring
    the efficiency approach of ``VectorSpaceModel``.
    """

    def __init__(
        self,
        index: dict,
        metadata: dict,
        n: int = N,
    ) -> None:
        self.index = index
        self.metadata = metadata
        # N is fixed at 100 by the assignment; verify the corpus really has
        # that many documents so N and the corpus never silently drift apart.
        self.n = n
        if len(metadata) != n:
            raise ValueError(
                f"expected N={n} documents in metadata, found {len(metadata)}"
            )

        # -- document lengths |D| ------------------------------------------
        # |D| = number of PROCESSED tokens in D. The processed/stemmed token
        # sequence is exactly what the rest of the IR system indexes, so we
        # take it straight from the stored processed token list (TITLE+TEXT,
        # the same content the inverted index was built from). We do NOT use
        # raw character length and we do NOT treat title/category differently
        # from the body — CATEGORY is not indexed by the pipeline, so it is not
        # counted here either, keeping BM25 consistent with lnc.ltc.
        self.doc_len: dict[str, int] = {}
        for doc_id, meta in metadata.items():
            tokens = meta.get("tokens")
            if tokens is not None:
                self.doc_len[doc_id] = len(tokens)
            else:
                # Fallback if a metadata file without tokens is supplied: derive
                # |D| by summing the tf across the inverted index (the total
                # processed-token count for the document). This is exactly equal
                # to len(tokens); verify_lengths_match_index() asserts it.
                self.doc_len[doc_id] = 0

        if any(v == 0 for v in self.doc_len.values()) and not all(
            "tokens" in metadata.get(d, {}) for d in metadata
        ):
            # No processed tokens were available from metadata; reconstruct the
            # lengths from the inverted index instead.
            self.doc_len = self._lengths_from_index()

        # -- average document length avgdl ---------------------------------
        # avgdl = average processed document length over ALL N documents.
        total_len = sum(self.doc_len.get(doc_id, 0) for doc_id in metadata)
        self.avgdl: float = total_len / n if n else 0.0

        # -- idf cache -----------------------------------------------------
        # Precompute IDF(t) once per term (it depends only on df and N).
        self.idf: dict[str, float] = {
            term: self._idf(entry["df"]) for term, entry in index.items()
        }

    # -- construction helpers ------------------------------------------------

    @classmethod
    def from_files(
        cls,
        index_path: str | Path = DEFAULT_INDEX_PATH,
        metadata_path: str | Path = DEFAULT_METADATA_PATH,
        n: int = N,
    ) -> "BM25Model":
        """Load the same Part A inverted index and doc metadata as the VSM."""
        index = _load_json(index_path)
        metadata = _load_json(metadata_path)
        return cls(index, metadata, n=n)

    def _lengths_from_index(self) -> dict[str, int]:
        """Reconstruct |D| for every document by summing tf over the index.

        ``sum_t tf(t, D)`` equals the number of processed tokens in D, so this
        yields the identical lengths as counting the processed token list.
        """
        lengths: dict[str, int] = {doc_id: 0 for doc_id in self.metadata}
        for entry in self.index.values():
            for doc_id, tf in entry["postings"].items():
                lengths[doc_id] = lengths.get(doc_id, 0) + int(tf)
        return lengths

    # -- idf -----------------------------------------------------------------

    def _idf(self, df: int) -> float:
        """Standard Lucene-style BM25 IDF (always positive).

            IDF(t) = ln( (N - df + 0.5) / (df + 0.5) + 1 )
        """
        return math.log((self.n - df + 0.5) / (df + 0.5) + 1.0)

    # -- ranking -------------------------------------------------------------

    def query_bm25(
        self,
        query_string: str,
        top_k: int = 10,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> list[dict]:
        """Rank documents for ``query_string`` under Okapi BM25.

        Parameters
        ----------
        query_string:
            Raw free-text query. Normalized internally with the **exact same**
            preprocessing pipeline as lnc.ltc (``preprocess_text``): lowercase
            -> punctuation split -> NLTK English stopword removal -> Porter
            stemming.
        top_k:
            Maximum number of results to return (default 10).
        k1, b:
            BM25 parameters (defaults 1.5 / 0.75), configurable per call.

        Returns
        -------
        A list of at most ``top_k`` result dicts, structurally compatible with
        ``query_vsm``::

            {"docID": ..., "title": ..., "category": ..., "score": ...}

        sorted by **descending BM25 score**, ties broken by **ascending docID**
        (explicit, so results never depend on dict/set iteration order).

        Deterministic edge-case behavior:
          * empty / punctuation-only / all-stopword query -> ``[]``.
          * a query whose every term is unknown to the corpus -> ``[]``.
          * unknown terms have no df/postings and contribute **zero** (they are
            simply skipped; the query never crashes).
          * repeated query terms and terms that stem to the same token are
            collapsed to a single term — the written BM25 formula sums each
            query term once and has no query-tf factor.
        """
        terms = preprocess_text(query_string)
        if not terms:
            return []

        # Collapse repeated / stem-colliding query terms to a set: the formula
        # sums each distinct query term once (no query-tf weighting). Keep only
        # terms that exist in the vocabulary — unknown terms contribute 0.
        query_terms = [t for t in dict.fromkeys(terms) if t in self.index]
        if not query_terms:
            return []

        # Candidate documents = union of the postings of the known query terms.
        # A document not containing term t has tf(t,D)=0, so that term adds 0;
        # only documents sharing at least one query term can score > 0.
        candidates: set[str] = set()
        for term in query_terms:
            candidates.update(self.index[term]["postings"].keys())

        avgdl = self.avgdl
        scores: list[tuple[str, float]] = []
        for doc_id in candidates:
            doc_length = self.doc_len.get(doc_id, 0)
            # Length normalization denominator component, computed once per doc.
            length_norm = 1.0 - b + b * (doc_length / avgdl if avgdl else 0.0)
            score = 0.0
            for term in query_terms:
                postings = self.index[term]["postings"]
                tf = postings.get(doc_id)
                if not tf:  # term absent from this document -> contributes 0
                    continue
                idf = self.idf[term]
                # BM25 term contribution:
                #   IDF * ( tf*(k1+1) ) / ( tf + k1*(1 - b + b*|D|/avgdl) )
                numerator = tf * (k1 + 1.0)
                denominator = tf + k1 * length_norm
                score += idf * (numerator / denominator)
            if score > 0.0:
                scores.append((doc_id, score))

        # Primary: descending score. Secondary: ascending docID (explicit).
        scores.sort(key=lambda pair: (-pair[1], _doc_id_sort_key(pair[0])))

        results: list[dict] = []
        for doc_id, score in scores[:top_k]:
            meta = self.metadata.get(doc_id, {})
            results.append(
                {
                    "docID": doc_id,
                    "title": meta.get("title", ""),
                    "category": meta.get("category", ""),
                    "score": score,
                }
            )
        return results

    # -- verification helpers ------------------------------------------------

    def verify_lengths_match_index(self) -> None:
        """Assert |D| from the processed token list equals the inverted-index tf sum.

        This proves the BM25 document length is the processed/stemmed token
        count the rest of the system uses — not raw characters and not a
        differently-tokenized stream.
        """
        from_index = self._lengths_from_index()
        for doc_id in self.metadata:
            token_len = self.doc_len.get(doc_id, 0)
            index_len = from_index.get(doc_id, 0)
            if token_len != index_len:
                raise AssertionError(
                    f"{doc_id}: processed-token length {token_len} != "
                    f"inverted-index tf sum {index_len}"
                )


# --------------------------------------------------------------------------
# Module-level convenience API (a single shared model, lazily loaded), mirrors
# the shape of vsm.query_vsm so callers can swap models trivially.
# --------------------------------------------------------------------------

_MODEL: BM25Model | None = None


def get_model() -> BM25Model:
    """Return a lazily constructed, cached BM25 model built from Part A outputs."""
    global _MODEL
    if _MODEL is None:
        _MODEL = BM25Model.from_files()
    return _MODEL


def query_bm25(
    query_string: str,
    top_k: int = 10,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
) -> list[dict]:
    """Convenience wrapper: rank ``query_string`` with BM25 (see BM25Model.query_bm25).

    Returns results in the same structure as ``vsm.query_vsm`` (docID, title,
    category, score) so the two classical models can be compared directly. The
    required ``query_vsm`` is never called or modified here.
    """
    return get_model().query_bm25(query_string, top_k=top_k, k1=k1, b=b)


# --------------------------------------------------------------------------
# Local test / demo:  python src/bm25.py
# --------------------------------------------------------------------------

def _manual_verification() -> None:
    """Hand-check the BM25 arithmetic for one query so it is viva-explainable.

    We recompute BM25 for the two-term query 'cotton denim' from first
    principles for a single document and assert it matches ``query_bm25``.
    """
    model = get_model()
    k1, b = DEFAULT_K1, DEFAULT_B
    print("Manual BM25 verification for query 'cotton denim':")
    print(f"  N = {model.n}, avgdl = {model.avgdl:.4f}, k1 = {k1}, b = {b}")

    query_terms = [t for t in ("cotton", "denim") if t in model.index]
    if len(query_terms) < 2:
        print("  one of the demo terms is absent; skipping manual check")
        return

    for term in query_terms:
        df = model.index[term]["df"]
        idf = model._idf(df)
        print(
            f"  term {term!r}: df = {df}, "
            f"IDF = ln(({model.n}-{df}+0.5)/({df}+0.5)+1) = {idf:.6f}"
        )

    # Independently recompute the score for every candidate document.
    candidates: set[str] = set()
    for term in query_terms:
        candidates.update(model.index[term]["postings"].keys())

    expected: list[tuple[str, float]] = []
    for doc_id in candidates:
        dl = model.doc_len[doc_id]
        length_norm = 1.0 - b + b * (dl / model.avgdl)
        score = 0.0
        for term in query_terms:
            tf = model.index[term]["postings"].get(doc_id, 0)
            if not tf:
                continue
            idf = model._idf(model.index[term]["df"])
            score += idf * (tf * (k1 + 1.0)) / (tf + k1 * length_norm)
        if score > 0.0:
            expected.append((doc_id, score))
    expected.sort(key=lambda p: (-p[1], _doc_id_sort_key(p[0])))

    got = model.query_bm25("cotton denim", top_k=len(expected))
    ok = True
    for (doc_id, exp_score), row in zip(expected, got):
        if row["docID"] != doc_id or abs(row["score"] - exp_score) > 1e-9:
            ok = False
            print(f"  MISMATCH {doc_id}: hand={exp_score:.6f} code={row['score']:.6f}")
    print(f"  hand-computed ranking matches query_bm25: {ok}")

    if got:
        top = got[0]
        top_dl = model.doc_len[top["docID"]]
        print(
            f"  top document {top['docID']} score={top['score']:.4f} "
            f"(|D|={top_dl} tokens)"
        )
        # Show the single-document worked arithmetic for the top doc.
        length_norm = 1.0 - b + b * (top_dl / model.avgdl)
        print(f"    length factor (1 - b + b*|D|/avgdl) = {length_norm:.6f}")
        for term in query_terms:
            tf = model.index[term]["postings"].get(top["docID"], 0)
            if not tf:
                print(f"    {term!r}: tf=0 -> contributes 0")
                continue
            idf = model._idf(model.index[term]["df"])
            contrib = idf * (tf * (k1 + 1.0)) / (tf + k1 * length_norm)
            print(
                f"    {term!r}: tf={tf}, "
                f"contrib = {idf:.4f} * ({tf}*{k1 + 1.0:.1f}) / "
                f"({tf} + {k1}*{length_norm:.4f}) = {contrib:.4f}"
            )


def _demo() -> None:
    """Print BM25 top results for a few real multi-term queries."""
    model = get_model()
    demo_queries = [
        "cotton shirt",
        "denim jeans",
        "winter jacket",
        "high waist leggings",
    ]
    for q in demo_queries:
        terms = preprocess_text(q)
        known = [t for t in dict.fromkeys(terms) if t in model.index]
        unknown = [t for t in dict.fromkeys(terms) if t not in model.index]
        print("=" * 68)
        print(f"query               : {q!r}")
        print(f"normalized terms    : {terms}")
        print(f"known corpus terms  : {known}")
        if unknown:
            print(f"unknown (no df)     : {unknown}")
        results = model.query_bm25(q, top_k=10)
        if not results:
            print("results             : (none)")
            continue
        print(f"top {len(results)} results:")
        for rank, row in enumerate(results, start=1):
            print(
                f"  {rank:2d}. {row['docID']}  "
                f"score={row['score']:.4f}  "
                f"[{row['category']}] {row['title']}"
            )


def _edge_cases() -> None:
    """Exercise the documented edge cases; each must return cleanly."""
    model = get_model()
    cases = {
        "empty query": "",
        "punctuation only": "!!! ??? ...",
        "repeated terms": "cotton cotton cotton",
        "unknown term only": "zzznotacorpusword",
        "known + unknown": "cotton zzznotacorpusword",
        "single known term": "denim",
        "stem-collapsing terms": "shirt shirts shirt",
    }
    print("=" * 68)
    print("edge cases (must not crash):")
    for name, q in cases.items():
        results = model.query_bm25(q, top_k=10)
        print(f"  {name:22s}: -> {len(results)} result(s)")


def main() -> None:
    model = get_model()
    model.verify_lengths_match_index()
    print(
        f"Loaded BM25 model (k1={DEFAULT_K1}, b={DEFAULT_B}), N = {model.n}, "
        f"avgdl = {model.avgdl:.4f}"
    )
    print("document length source: processed token stream (verified == index tf sum)")
    print()
    _manual_verification()
    print()
    _demo()
    print()
    _edge_cases()


if __name__ == "__main__":
    main()
