"""
test_ir.py
==========

Behavioural tests for the IR system (corpus, preprocessing, inverted index,
lnc.ltc VSM, positional index, and the proximity-aware reranker).

These tests exercise *actual behaviour* — recomputing df/tf/idf/cosine by hand
and comparing against the modules — rather than merely asserting that functions
exist. They import the live ``src`` modules and run against the real corpus, so
they double as a regression guard for the whole pipeline.

Run::

    python -m pytest tests/ -q
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

import pytest

# Make the src package importable whether pytest is run from the repo root or
# elsewhere. The src modules import each other by bare name, so src must be on
# sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from index_builder import (  # noqa: E402
    EXPECTED_N,
    build_from_corpus,
    build_inverted_index,
)
from positional_index import (  # noqa: E402
    build_positional_index,
    get_index,
    phrase_search,
    proximity_search,
    verify_consistency_with_inverted_index,
    verify_positions_match_tokens,
)
from preprocess import preprocess_text, tokenize  # noqa: E402
from reranker import (  # noqa: E402
    _proximity_bonus,
    compare_baseline_and_reranked,
    rerank_with_proximity,
)
from vsm import N, get_model, query_vsm  # noqa: E402


# --------------------------------------------------------------------------
# Shared fixtures (build once per test session; the corpus is read-only).
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def corpus():
    """Parsed documents, inverted index, and doc store from the real corpus."""
    documents, index, store = build_from_corpus()
    return documents, index, store


@pytest.fixture(scope="session")
def documents(corpus):
    return corpus[0]


@pytest.fixture(scope="session")
def inverted_index(corpus):
    return corpus[1]


@pytest.fixture(scope="session")
def model():
    return get_model()


@pytest.fixture(scope="session")
def pos_index():
    return get_index()


# ==========================================================================
# Corpus
# ==========================================================================

def test_corpus_has_exactly_100_documents(documents):
    assert len(documents) == EXPECTED_N == 100


def test_document_ids_are_unique(documents):
    ids = [d.doc_id for d in documents]
    assert len(ids) == len(set(ids))


def test_required_fields_present(documents):
    for d in documents:
        assert d.doc_id, "missing DOCID"
        assert d.category, f"{d.doc_id} missing CATEGORY"
        assert d.title, f"{d.doc_id} missing TITLE"
        assert d.text, f"{d.doc_id} missing TEXT"


# ==========================================================================
# Preprocessing
# ==========================================================================

def test_lowercasing():
    assert preprocess_text("COTTON Shirt") == preprocess_text("cotton shirt")
    assert all(tok == tok.lower() for tok in preprocess_text("MEN Women BLUE"))


def test_punctuation_removed_and_split():
    # Punctuation becomes a split point, not a glued token.
    tokens = tokenize("t-shirt, easy-care!")
    assert "," not in "".join(tokens)
    assert "t" in tokens and "shirt" in tokens
    assert "tshirt" not in tokens


def test_stopwords_removed():
    # "the", "and", "of", "is", "for" are NLTK English stopwords.
    assert preprocess_text("the and of is for") == []
    # A content word survives alongside stopwords.
    assert "cotton" in preprocess_text("the cotton is soft")


def test_stemming_behaviour():
    # Porter stemming collapses morphological variants to one stem.
    assert preprocess_text("shirts") == preprocess_text("shirt")
    assert preprocess_text("leggings")  # non-empty
    # Known aggressive Porter stems used across the project.
    assert preprocess_text("festive") == ["festiv"]


def test_query_and_index_use_same_pipeline(documents):
    # The stems produced for a query term must match the stems stored for the
    # same surface word in a document's token stream.
    doc = next(d for d in documents if "cotton" in d.tokens)
    assert preprocess_text("cotton") == ["cotton"]
    assert "cotton" in doc.tokens


# ==========================================================================
# Inverted index
# ==========================================================================

def test_df_equals_number_of_postings(inverted_index):
    for term, entry in inverted_index.items():
        assert entry["df"] == len(entry["postings"]), term


def test_tf_matches_recount(documents, inverted_index):
    expected: dict[str, Counter] = {}
    for d in documents:
        for term, tf in Counter(d.tokens).items():
            expected.setdefault(term, Counter())[d.doc_id] = tf
    for term, entry in inverted_index.items():
        for doc_id, tf in entry["postings"].items():
            assert tf == expected[term][doc_id], f"{term}/{doc_id}"


def test_postings_valid_and_no_duplicate_docs(inverted_index):
    for term, entry in inverted_index.items():
        doc_ids = list(entry["postings"].keys())
        assert len(doc_ids) == len(set(doc_ids)), f"duplicate doc in {term}"
        for doc_id, tf in entry["postings"].items():
            assert tf >= 1, f"{term}/{doc_id} tf < 1"


def test_df_is_not_collection_frequency(documents):
    # df counts distinct documents, never the sum of tfs. Rebuild and confirm.
    index = build_inverted_index(documents)
    # 'cotton' occurs multiple times in some docs; df must still be doc count.
    entry = index["cotton"]
    distinct_docs = {d.doc_id for d in documents if "cotton" in d.tokens}
    assert entry["df"] == len(distinct_docs)
    total_occurrences = sum(entry["postings"].values())
    assert total_occurrences >= entry["df"]  # cf >= df, and here strictly >


# ==========================================================================
# VSM (lnc.ltc)
# ==========================================================================

def test_unknown_query_terms_are_safe(model):
    # A term absent from the vocabulary must not crash and must not invent hits.
    assert model.query_vsm("zzznotacorpusword") == []
    # Known + unknown behaves like the known term alone (unknown contributes 0).
    known_only = [r["docID"] for r in model.query_vsm("cotton")]
    mixed = [r["docID"] for r in model.query_vsm("cotton zzznotacorpusword")]
    assert known_only == mixed


def test_empty_and_stopword_queries_return_empty(model):
    assert model.query_vsm("") == []
    assert model.query_vsm("!!! ???") == []
    assert model.query_vsm("the and of") == []


def test_cosine_scores_in_valid_range(model):
    for q in ("cotton shirt", "winter jacket", "denim", "printed saree"):
        for row in model.query_vsm(q, top_k=10):
            assert 0.0 <= row["score"] <= 1.0 + 1e-9, (q, row)


def test_ranking_is_deterministic(model):
    a = model.query_vsm("cotton shirt", top_k=10)
    b = model.query_vsm("cotton shirt", top_k=10)
    assert [r["docID"] for r in a] == [r["docID"] for r in b]
    assert [r["score"] for r in a] == [r["score"] for r in b]


def test_tie_break_is_increasing_docid(model):
    # Results with equal scores must be ordered by ascending docID.
    results = model.query_vsm("cotton shirt", top_k=10)
    for i in range(len(results) - 1):
        s0, s1 = results[i]["score"], results[i + 1]["score"]
        if abs(s0 - s1) < 1e-12:
            assert results[i]["docID"] < results[i + 1]["docID"], (
                results[i], results[i + 1]
            )


def test_top_k_behaviour(model):
    assert len(model.query_vsm("cotton", top_k=3)) == 3
    assert len(model.query_vsm("cotton", top_k=10)) <= 10
    # top_k larger than candidate count returns all candidates, not padding.
    all_results = model.query_vsm("kurta", top_k=1000)
    assert 0 < len(all_results) <= 100


def test_lnc_ltc_formula_single_term(model):
    # Hand-recompute the lnc.ltc score for a single-term query and compare.
    term = "denim"
    entry = model.index[term]
    df = entry["df"]
    idf = math.log10(N / df)  # ltc idf, N = 100
    # A single-term query normalizes to weight 1.0, so score = w(d)/norm.
    expected = []
    for doc_id, tf in entry["postings"].items():
        d_raw = 1.0 + math.log10(tf)  # lnc doc weight, NO idf on documents
        expected.append((doc_id, d_raw / model.doc_norms[doc_id]))
    expected.sort(key=lambda p: (-p[1], p[0]))
    got = model.query_vsm(term, top_k=len(expected))
    assert idf > 0  # denim is not in all documents
    for (exp_id, exp_score), row in zip(expected, got):
        assert row["docID"] == exp_id
        assert row["score"] == pytest.approx(exp_score, abs=1e-12)


def test_documents_have_no_idf(model):
    # The document weight for a term is exactly 1 + log10(tf), independent of df.
    term = "cotton"
    entry = model.index[term]
    for doc_id, tf in list(entry["postings"].items())[:5]:
        assert model.doc_weights[doc_id][term] == pytest.approx(
            1.0 + math.log10(tf), abs=1e-12
        )


# ==========================================================================
# Positional index
# ==========================================================================

def test_positions_recorded_and_match_tokens(documents):
    index = build_positional_index(documents)
    # This raises if any stored position does not recover its own term.
    verify_positions_match_tokens(documents, index)


def test_tf_agrees_with_number_of_positions(documents):
    index = build_positional_index(documents)
    for term, entry in index.items():
        for doc_id, posting in entry["postings"].items():
            assert posting["tf"] == len(posting["positions"])


def test_positional_consistent_with_inverted_index(documents):
    index = build_positional_index(documents)
    inverted = build_inverted_index(documents)
    verify_consistency_with_inverted_index(index, inverted)


def test_phrase_requires_consecutive_positions(pos_index):
    # 'cotton shirt' matches only where shirt sits immediately after cotton.
    results = phrase_search("cotton shirt")
    assert results, "expected some phrase matches"
    for res in results:
        for match in res["matches"]:
            assert len(match) == 2
            assert match[1] - match[0] == 1  # strictly consecutive


def test_phrase_excludes_mere_cooccurrence(pos_index):
    # D011 contains both 'cotton' and 'shirt' but NOT adjacently.
    both_present = {
        d for d in pos_index._postings("cotton") if d in pos_index._postings("shirt")
    }
    phrase_docs = {r["docID"] for r in phrase_search("cotton shirt")}
    cooccur_only = both_present - phrase_docs
    assert cooccur_only, "expected a co-occurrence-only document"
    # For such a doc, no cotton position is immediately followed by shirt.
    example = sorted(cooccur_only)[0]
    cpos = set(pos_index._positions("cotton", example))
    spos = set(pos_index._positions("shirt", example))
    assert not any((p + 1) in spos for p in cpos)


def test_phrase_order_matters(pos_index):
    forward = {r["docID"] for r in phrase_search("cotton shirt")}
    reverse = {r["docID"] for r in phrase_search("shirt cotton")}
    # The two phrases are different constraints; forward matches must exist and
    # differ from the reverse match set (order changes the result).
    assert forward
    assert forward != reverse


def test_proximity_respects_k(pos_index):
    for res in proximity_search("cotton", "shirt", 3, ordered=True):
        for p1, p2 in res["pairs"]:
            assert 0 < p2 - p1 <= 3


def test_proximity_ordered_definition(pos_index):
    # Ordered: only p2 > p1 pairs are kept (term1 before term2).
    res = proximity_search("high", "waist", 1, ordered=True)
    assert res
    for doc in res:
        for p1, p2 in doc["pairs"]:
            assert p2 - p1 == 1  # adjacency is the k = 1 case


def test_proximity_unknown_and_invalid_k(pos_index):
    assert proximity_search("cotton", "zzznotaword", 3) == []
    assert proximity_search("cotton", "shirt", 0) == []


def test_phrase_unknown_term_returns_empty(pos_index):
    assert phrase_search("cotton zzznotaword") == []
    assert phrase_search("") == []


# ==========================================================================
# Reranking (novelty)
# ==========================================================================

def test_baseline_vsm_unchanged_by_reranker(model):
    # Calling the reranker must not mutate the baseline VSM ranking.
    before = [r["docID"] for r in model.query_vsm("cotton denim", top_k=10)]
    rerank_with_proximity("cotton denim")
    after = [r["docID"] for r in model.query_vsm("cotton denim", top_k=10)]
    assert before == after


def test_alpha_zero_recovers_baseline_order(model):
    baseline = [r["docID"] for r in model.query_vsm("cotton denim", top_k=10)]
    reranked = [
        r["docID"] for r in rerank_with_proximity("cotton denim", alpha=0.0)
    ]
    assert baseline == reranked


def test_proximity_bonus_is_deterministic():
    a = rerank_with_proximity("cotton denim")
    b = rerank_with_proximity("cotton denim")
    assert [(r["docID"], r["proximity_bonus"], r["final_score"]) for r in a] == \
           [(r["docID"], r["proximity_bonus"], r["final_score"]) for r in b]


def test_reranked_results_are_deterministic():
    a = compare_baseline_and_reranked("regular winter")
    b = compare_baseline_and_reranked("regular winter")
    assert a["reranked_order"] == b["reranked_order"]
    assert a["changed"] == b["changed"]


def test_no_bonus_without_positional_evidence(pos_index):
    # A document where the two query terms never co-occur must get bonus 0.
    cpos = set(pos_index._positions("cotton", "D053"))
    # Pick a doc/term-pair with no co-occurrence: single-term query -> no pairs.
    bonus, closest = _proximity_bonus("D001", ["cotton"], pos_index)
    assert bonus == 0.0 and closest is None
    # Two distinct terms that do co-occur adjacently -> positive bonus.
    bonus2, closest2 = _proximity_bonus("D053", ["cotton", "denim"], pos_index)
    assert bonus2 > 0.0 and closest2 is not None
    assert cpos  # sanity: cotton actually occurs in D053


def test_reranker_final_score_formula():
    alpha = 0.15
    for row in rerank_with_proximity("cotton denim", alpha=alpha):
        assert row["final_score"] == pytest.approx(
            row["cosine_score"] + alpha * row["proximity_bonus"], abs=1e-12
        )
