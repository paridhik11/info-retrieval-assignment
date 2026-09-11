"""
vsm.py
======

Part B of the assignment: ranked retrieval with the Vector Space Model (VSM)
using the **lnc.ltc** weighting scheme, implemented from first principles.

SMART notation reminder
------------------------
``lnc.ltc`` is ``ddd.qqq`` where the first triple describes the *document*
term weights and the second triple describes the *query* term weights, each
as (term-frequency) . (document-frequency) . (normalization):

    documents  l n c  = log tf   . no-df . cosine
    queries    l t c  = log tf   . idf   . cosine

So the exact formulas required by the assignment are:

    Document weight (lnc), for term t in document d with in-document tf:
        w(d,t) = 1 + log10(tf)      if tf > 0
        w(d,t) = 0                  if tf = 0
      -> NO idf on the document side (the "n" in ln**n**... — the middle
         letter for documents is 'n' = none).

    Query weight (ltc), for term t in the query with query tf ``tfq``:
        w(q,t) = (1 + log10(tfq)) * log10(N / df_t)
      -> idf *is* applied on the query side (the 't' = idf).

    N = 100  (the collection size, fixed by the assignment).
    df_t = document frequency of t, read straight from the inverted index.

Then both vectors are **cosine normalized** (the trailing 'c' on each side)
and the score is their dot product.

Why idf is absent from document weights but present on the query side
---------------------------------------------------------------------
idf (log10(N/df)) is a property of a *term across the whole collection*, not
of a term inside one document. In lnc.ltc we apply it exactly once — on the
query side — so a rare word still boosts the ranking, but we avoid squaring
that boost by also applying it to every document. Applying idf on only one
side is a standard, deliberate choice (it keeps document weights cheap to
precompute and independent of query statistics, and it is the classic
"lnc.ltc" recipe from Manning et al., IIR ch. 6).

Why cosine normalization is necessary
-------------------------------------
Without it, long documents (more terms, higher tf) would accumulate larger
raw dot products purely because of length, not relevance. Dividing each
vector by its Euclidean norm projects every document and the query onto the
unit sphere, so similarity depends on term *proportions*, not document size.
After normalizing both vectors we simply take the dot product — we must NOT
divide by the norms again, because they are already unit length.

Why candidate documents come from the postings
-----------------------------------------------
A document can only have a non-zero cosine with the query if it shares at
least one query term. So instead of scoring all 100 documents against a full
vocabulary-wide dense matrix, we take the union of the postings lists of the
query terms — that is exactly the set of documents that can score > 0.

Why the final tie-break is docID
--------------------------------
Floating-point cosine scores can tie (e.g. structurally identical product
descriptions). To keep results reproducible across runs, machines, and
Python dict orderings, we break ties deterministically by *increasing* docID
rather than relying on insertion order.

This module deliberately does NOT use TF-IDF/BM25/embedding libraries: the
lnc.ltc arithmetic is spelled out so it can be explained (and hand-checked)
in a viva.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

# Allow ``python src/vsm.py`` as well as package-style imports.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from preprocess import preprocess_text  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_PATH = REPO_ROOT / "output" / "inverted_index.json"
DEFAULT_METADATA_PATH = REPO_ROOT / "output" / "doc_metadata.json"

# Collection size fixed by the assignment. We assert the loaded metadata
# actually contains this many documents so N and the corpus never silently
# drift apart, but the idf formula always uses N = 100 exactly.
N = 100

_DOC_ID_RE = re.compile(r"^(.*?)(\d+)$")


def _doc_id_sort_key(doc_id: str) -> tuple:
    """
    Ascending docID order using the numeric suffix when present (D2 < D10).

    Mirrors the ordering used by ``index_builder`` so the VSM tie-break is
    consistent with the rest of the pipeline.
    """
    match = _DOC_ID_RE.match(doc_id)
    if match:
        return (match.group(1), int(match.group(2)))
    return (doc_id, 0)


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class VectorSpaceModel:
    """
    lnc.ltc ranked retrieval over a prebuilt inverted index.

    Document-side weights and per-document cosine norms are precomputed once
    at construction time, so answering repeated queries only touches the
    postings of the query terms (no full dense matrix, no per-query rescan of
    the whole collection).
    """

    def __init__(
        self,
        index: dict,
        metadata: dict,
        n: int = N,
    ) -> None:
        self.index = index
        self.metadata = metadata
        # N is fixed at 100 by the assignment; we keep it as an attribute but
        # also verify the corpus really has that many documents.
        self.n = n
        if len(metadata) != n:
            raise ValueError(
                f"expected N={n} documents in metadata, found {len(metadata)}"
            )

        # Precompute document-side (lnc) weights and cosine norms.
        #   doc_weights[docID][term] = 1 + log10(tf)
        #   doc_norms[docID]         = sqrt(sum_t (1 + log10(tf))^2)
        self.doc_weights: dict[str, dict[str, float]] = {
            doc_id: {} for doc_id in metadata
        }
        for term, entry in index.items():
            for doc_id, tf in entry["postings"].items():
                # tf in the postings is always >= 1, so log10(tf) is defined.
                # w(d,t) = 1 + log10(tf); the tf = 0 case (weight 0) simply
                # never appears in a postings list, which is exactly why the
                # inverted index lets us skip absent terms.
                self.doc_weights.setdefault(doc_id, {})[term] = 1.0 + math.log10(tf)

        self.doc_norms: dict[str, float] = {}
        for doc_id, weights in self.doc_weights.items():
            norm = math.sqrt(sum(w * w for w in weights.values()))
            self.doc_norms[doc_id] = norm

    # -- construction helpers ------------------------------------------------

    @classmethod
    def from_files(
        cls,
        index_path: str | Path = DEFAULT_INDEX_PATH,
        metadata_path: str | Path = DEFAULT_METADATA_PATH,
        n: int = N,
    ) -> "VectorSpaceModel":
        """Load the inverted index and doc metadata written by Part A."""
        index = _load_json(index_path)
        metadata = _load_json(metadata_path)
        return cls(index, metadata, n=n)

    # -- query side (ltc) ----------------------------------------------------

    def query_weights(self, query_string: str) -> tuple[list[str], dict[str, float]]:
        """
        Build the *raw* (pre-normalization) ltc query weights.

        Returns ``(normalized_terms, weights)`` where ``normalized_terms`` is
        the full stemmed query token list (useful for display/debugging, and
        it shows repeated / collapsed terms) and ``weights`` maps each query
        term that actually occurs in the corpus to::

            w(q,t) = (1 + log10(tfq)) * log10(N / df_t)

        Deterministic edge-case behavior:
          * empty / punctuation-only / all-stopword query -> ``[]``, ``{}``.
          * repeated terms and terms that collapse to the same stem raise
            that stem's query tf (``tfq``), exactly as intended.
          * a term absent from the corpus has no df, so it contributes no
            weight and is simply skipped (the query does not crash).
        """
        terms = preprocess_text(query_string)

        # Query term frequencies. Repeated words and words that stem to the
        # same token both increase tfq for that single term.
        query_tf: dict[str, int] = {}
        for term in terms:
            query_tf[term] = query_tf.get(term, 0) + 1

        weights: dict[str, float] = {}
        for term, tfq in query_tf.items():
            entry = self.index.get(term)
            if entry is None:
                # Unknown term: no usable df, contributes nothing to the score.
                continue
            df = entry["df"]
            idf = math.log10(self.n / df)  # ltc idf, N fixed at 100
            weights[term] = (1.0 + math.log10(tfq)) * idf
        return terms, weights

    # -- ranking -------------------------------------------------------------

    def query_vsm(self, query_string: str, top_k: int = 10) -> list[dict]:
        """
        Rank documents for ``query_string`` under lnc.ltc cosine similarity.

        Returns at most ``top_k`` results, each a dict::

            {"docID": ..., "title": ..., "category": ..., "score": ...}

        sorted by decreasing cosine score, ties broken by increasing docID.
        An empty list is returned cleanly when nothing can score (empty query,
        or every query term is outside the corpus).
        """
        _terms, raw_query_weights = self.query_weights(query_string)
        if not raw_query_weights:
            # Empty query, punctuation/stopword-only query, or a query whose
            # every term is unknown -> no possible non-zero score.
            return []

        # Cosine-normalize the query vector (the 'c' in ltc). If every idf is
        # zero (e.g. the only query term appears in all N documents, df = N),
        # the norm is 0 and no document can be distinguished -> empty result.
        q_norm = math.sqrt(sum(w * w for w in raw_query_weights.values()))
        if q_norm == 0.0:
            return []
        query_vector = {t: w / q_norm for t, w in raw_query_weights.items()}

        # Candidate documents = union of the postings of the query terms.
        # These are the only documents that can have a non-zero cosine.
        candidates: set[str] = set()
        for term in query_vector:
            candidates.update(self.index[term]["postings"].keys())

        scores: list[tuple[str, float]] = []
        for doc_id in candidates:
            doc_norm = self.doc_norms.get(doc_id, 0.0)
            if doc_norm == 0.0:
                continue
            doc_terms = self.doc_weights.get(doc_id, {})
            dot = 0.0
            for term, q_weight in query_vector.items():
                d_raw = doc_terms.get(term)
                if d_raw is None:
                    continue
                # Normalize the document weight (cosine) then multiply by the
                # already-normalized query weight. Both vectors are unit
                # length, so the dot product IS the cosine similarity — we do
                # not divide by the norms a second time.
                dot += (d_raw / doc_norm) * q_weight
            if dot > 0.0:
                scores.append((doc_id, dot))

        # Primary sort: decreasing score. Secondary sort: increasing docID
        # (explicit, so results never depend on dict/set iteration order).
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


# --------------------------------------------------------------------------
# Module-level convenience API (a single shared model, lazily loaded).
# --------------------------------------------------------------------------

_MODEL: VectorSpaceModel | None = None


def get_model() -> VectorSpaceModel:
    """Return a lazily constructed, cached VSM built from the Part A outputs."""
    global _MODEL
    if _MODEL is None:
        _MODEL = VectorSpaceModel.from_files()
    return _MODEL


def query_vsm(query_string: str, top_k: int = 10) -> list[dict]:
    """
    Convenience wrapper so callers (UI, evaluation) can rank without managing
    the model object. Keeps the pure VSM baseline available and untouched by
    any later novelty reranker.
    """
    return get_model().query_vsm(query_string, top_k=top_k)


# --------------------------------------------------------------------------
# Local test / demo. Run:  python src/vsm.py
# --------------------------------------------------------------------------

def _manual_verification() -> None:
    """
    Hand-check the lnc.ltc arithmetic for one simple, single-term query so the
    implementation can be explained in a viva.

    Query: "denim" (a single content word).

    Query side (ltc), tfq = 1:
        idf   = log10(N / df_denim)
        w(q)  = (1 + log10(1)) * idf = 1 * idf = idf
        After cosine normalization of a 1-D query vector, the query weight is
        idf / |idf| = 1.0.  (A single-term query always normalizes to 1.)

    Document side (lnc) for a document d containing 'denim' with tf:
        w(d, denim) = 1 + log10(tf)
        normalized  = (1 + log10(tf)) / doc_norm(d)

    So the cosine for a single-term query reduces to:
        score(d) = 1.0 * (1 + log10(tf_denim_in_d)) / doc_norm(d)

    We recompute this independently below and assert it matches query_vsm.
    """
    model = get_model()
    term = "denim"  # already a Porter stem of "denim"
    entry = model.index.get(term)
    print("Manual verification for single-term query 'denim':")
    print(f"  N = {model.n}")
    if entry is None:
        print("  'denim' is not in the corpus; skipping manual check")
        return

    df = entry["df"]
    idf = math.log10(model.n / df)
    print(f"  df(denim) = {df}")
    print(f"  idf = log10({model.n}/{df}) = {idf:.6f}")
    print("  query weight after 1-D cosine normalization = 1.000000")

    # Independent recomputation of the score for each posting document.
    expected: list[tuple[str, float]] = []
    for doc_id, tf in entry["postings"].items():
        d_raw = 1.0 + math.log10(tf)
        score = (d_raw / model.doc_norms[doc_id]) * 1.0
        expected.append((doc_id, score))
    expected.sort(key=lambda p: (-p[1], _doc_id_sort_key(p[0])))

    got = model.query_vsm(term, top_k=len(expected))
    ok = True
    for (doc_id, exp_score), row in zip(expected, got):
        if row["docID"] != doc_id or abs(row["score"] - exp_score) > 1e-12:
            ok = False
            print(f"  MISMATCH {doc_id}: hand={exp_score:.6f} code={row['score']:.6f}")
    print(f"  hand-computed ranking matches query_vsm: {ok}")
    top = expected[0]
    print(f"  top document by hand: {top[0]} score={top[1]:.4f}")


def _demo() -> None:
    """Print normalized query terms and top results for a few real queries."""
    model = get_model()
    demo_queries = [
        "cotton kurta",
        "denim jeans",
        "winter jacket for men",
    ]
    for q in demo_queries:
        terms, weights = model.query_weights(q)
        known = [t for t in dict.fromkeys(terms) if t in weights]
        unknown = [t for t in dict.fromkeys(terms) if t not in weights]
        print("=" * 68)
        print(f"query               : {q!r}")
        print(f"normalized terms    : {terms}")
        print(f"known corpus terms  : {known}")
        if unknown:
            print(f"unknown (no df)     : {unknown}")
        results = model.query_vsm(q, top_k=10)
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
        terms, weights = model.query_weights(q)
        results = model.query_vsm(q, top_k=10)
        print(
            f"  {name:22s}: terms={terms} "
            f"known={list(weights)} -> {len(results)} result(s)"
        )


def main() -> None:
    print(f"Loaded VSM baseline (lnc.ltc), N = {get_model().n}")
    print()
    _manual_verification()
    print()
    _demo()
    print()
    _edge_cases()


if __name__ == "__main__":
    main()
