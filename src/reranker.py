"""
reranker.py
===========

Novelty component: **proximity-aware re-ranking**.

This module is a *lightweight, explainable* retrieval enhancement that
**extends** the required classical IR system rather than replacing it. It uses
only information that already exists in the project:

    1. the lnc.ltc Vector Space Model from Part B (``vsm.query_vsm``), and
    2. the positional index from Part C (``positional_index``).

It introduces **no** LLMs, embeddings, neural networks, vector databases,
external/semantic search APIs, pretrained models, or any different ranking
algorithm. The baseline lnc.ltc cosine ranking is left completely
untouched (``vsm.query_vsm`` is *called*, never modified) so the two rankings
can be compared side by side.

Motivation (why proximity is a useful secondary signal)
-------------------------------------------------------
The lnc.ltc VSM is a **bag-of-words** model: it measures term importance and
vector similarity but discards word order, so it cannot tell whether two query
terms occur *next to each other* or *at opposite ends* of a document. Two
documents can earn the same cosine score while one actually contains the query
terms tightly together (usually a better match) and the other scatters them.

The positional index already knows *where* each term occurs. So after the
baseline VSM has selected its candidates, we reward candidates in which the
query terms occur close together — a classic proximity heuristic layered on
top of the classical lexical score.

Why alpha is kept small
-----------------------
The proximity bonus is a **controlled secondary signal**. ``alpha`` (default
0.15) scales it so it can only *nudge* the ordering of documents the lnc.ltc
model already considers relevant; it must never overwhelm the required cosine
similarity. Keeping alpha small preserves the classical IR baseline as the
primary ranking force and keeps the whole thing explainable in a viva. alpha is
a parameter (not hard-coded deep in the loop) so its influence can be tuned and
demonstrated.

Algorithm (summary)
-------------------
    STEP 1  Candidate retrieval: take the top ``candidate_k`` (default 20)
            documents from the lnc.ltc VSM — the VSM candidate set only, never
            arbitrary unrelated documents.
    STEP 2  Query normalization: use the exact same preprocessing pipeline;
            drop unknown terms (no positional postings). If fewer than two
            distinct known terms remain, the proximity bonus is zero (we never
            fabricate positional evidence).
    STEP 3  Pairwise proximity signal: for every pair of DISTINCT known query
            terms that both occur in a candidate document, take the minimum
            absolute positional gap and add ``1 / (1 + min_gap)``.
    STEP 4  Combine: ``final_score = cosine_score + alpha * proximity_bonus``.
    STEP 5  Rank by final_score descending, docID ascending for ties; return
            the top ``top_k``.

This is a project-level retrieval enhancement combining lexical weighting from
lnc.ltc with positional proximity information. It is **not** claimed to be a
new research algorithm and is **not** claimed to universally improve retrieval.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

# Allow ``python src/reranker.py`` as well as package-style imports.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from preprocess import preprocess_text  # noqa: E402  (shared Part A pipeline)
from positional_index import get_index  # noqa: E402  (Part C positional index)
from vsm import _doc_id_sort_key, get_model  # noqa: E402  (Part B baseline)

# Default weight of the positional signal. Deliberately small so proximity is a
# secondary tie-shaping signal on top of — never a replacement for — the
# required lnc.ltc cosine similarity.
DEFAULT_ALPHA = 0.15
DEFAULT_TOP_K = 10
DEFAULT_CANDIDATE_K = 20


def _distinct_known_terms(query_string: str, pos_index) -> list[str]:
    """Normalized, distinct query terms that have positional postings.

    Uses the exact same preprocessing pipeline as the rest of the system
    (``preprocess_text``). Unknown terms — those with no entry in the
    positional index — are removed because they have no positions and therefore
    cannot contribute positional evidence. Order is preserved and duplicates
    are collapsed (a pair needs two *distinct* terms).
    """
    terms = preprocess_text(query_string)
    seen: list[str] = []
    for term in terms:
        if term in pos_index.index and term not in seen:
            seen.append(term)
    return seen


def _min_gap(positions_a: list[int], positions_b: list[int]) -> int | None:
    """Minimum absolute positional distance between any A position and any B.

    Both position lists are sorted ascending (guaranteed by the positional
    index), so a two-pointer merge finds the closest pair in O(len_a + len_b)
    instead of the naive O(len_a * len_b). Returns ``None`` if either list is
    empty (the pair does not co-occur in the document).
    """
    if not positions_a or not positions_b:
        return None
    i = j = 0
    best = None
    while i < len(positions_a) and j < len(positions_b):
        gap = abs(positions_a[i] - positions_b[j])
        if best is None or gap < best:
            best = gap
        if best == 0:
            break  # cannot get closer than adjacent-identical position
        # Advance the pointer on the smaller value to try to close the gap.
        if positions_a[i] < positions_b[j]:
            i += 1
        else:
            j += 1
    return best


def _proximity_bonus(
    doc_id: str, distinct_terms: list[str], pos_index
) -> tuple[float, dict | None]:
    """Sum of pairwise proximity bonuses for one document.

    For every pair of DISTINCT known query terms that BOTH occur in the
    document, the contribution is ``1 / (1 + min_gap)`` where ``min_gap`` is
    the smallest absolute positional distance between an occurrence of the two
    terms. Terms occurring very close together give a stronger bonus; terms far
    apart give a smaller one; a pair that does not co-occur contributes 0 (we
    do not fabricate positional evidence).

    Returns ``(proximity_bonus, closest_pair)`` where ``closest_pair`` records
    the single pair with the smallest gap (used purely to *explain* the bonus),
    or ``None`` when no pair co-occurs.
    """
    if len(distinct_terms) < 2:
        # Fewer than two distinct known terms -> no pairs -> zero bonus.
        return 0.0, None

    total = 0.0
    closest_pair: dict | None = None
    for term_a, term_b in itertools.combinations(distinct_terms, 2):
        positions_a = pos_index._positions(term_a, doc_id)
        positions_b = pos_index._positions(term_b, doc_id)
        gap = _min_gap(positions_a, positions_b)
        if gap is None:
            continue  # pair does not co-occur in this document -> contributes 0
        # Adjacent terms (gap=1) get a bonus of 0.5; farther apart gives less.
        pair_bonus = 1.0 / (1.0 + gap)
        total += pair_bonus
        if closest_pair is None or gap < closest_pair["gap"]:
            closest_pair = {"terms": (term_a, term_b), "gap": gap, "bonus": pair_bonus}
    return total, closest_pair


def rerank_with_proximity(
    query_string: str,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    alpha: float = DEFAULT_ALPHA,
) -> list[dict]:
    """Re-rank the lnc.ltc VSM candidates using a positional proximity signal.

    Parameters
    ----------
    query_string:
        The raw free-text query (normalized internally with the shared
        pipeline).
    top_k:
        Number of results to return after re-ranking (default 10).
    candidate_k:
        Number of top lnc.ltc VSM candidates to re-rank (default 20). Only
        these VSM candidates are considered — never arbitrary documents outside
        the VSM candidate set.
    alpha:
        Weight of the proximity signal in ``final_score = cosine + alpha *
        proximity_bonus`` (default 0.15). Kept small so the positional signal
        stays a controlled secondary influence.

    Returns
    -------
    A list of at most ``top_k`` result dicts, each carrying enough information
    to *explain* the ranking (for the viva)::

        {
            "docID": str,
            "title": str,
            "category": str,
            "cosine_score": float,       # baseline lnc.ltc score (unchanged)
            "proximity_bonus": float,    # summed pairwise 1/(1+min_gap)
            "final_score": float,        # cosine_score + alpha * proximity_bonus
            "closest_pair": {            # the pair/gap that best explains the bonus
                "terms": (termA, termB),
                "gap": int,
                "bonus": float,
            } | None,
            "alpha": float,
        }

    sorted by ``final_score`` descending, ``docID`` ascending for exact ties.
    """
    model = get_model()
    pos_index = get_index()

    # STEP 1 — Candidate retrieval from the baseline VSM (candidate set only).
    # Keep the baseline score unchanged and add proximity only during this
    # optional reranking stage; query_vsm is called but never modified.
    candidates = model.query_vsm(query_string, top_k=candidate_k)
    if not candidates:
        return []

    # STEP 2 — Query normalization (same pipeline) + drop unknown terms. If
    # fewer than two distinct known terms remain, every proximity bonus is 0
    # and re-ranking degenerates to the baseline order (reported honestly).
    distinct_terms = _distinct_known_terms(query_string, pos_index)

    # STEP 3 + STEP 4 — compute the proximity bonus per candidate, then combine.
    enriched: list[dict] = []
    for cand in candidates:
        doc_id = cand["docID"]
        cosine_score = cand["score"]
        bonus, closest_pair = _proximity_bonus(doc_id, distinct_terms, pos_index)
        # final_score = cosine_score + alpha * proximity_bonus
        final_score = cosine_score + alpha * bonus
        enriched.append(
            {
                "docID": doc_id,
                "title": cand.get("title", ""),
                "category": cand.get("category", ""),
                "cosine_score": cosine_score,
                "proximity_bonus": bonus,
                "final_score": final_score,
                "closest_pair": closest_pair,
                "alpha": alpha,
            }
        )

    # STEP 5 — Rank by final_score descending, docID ascending for exact ties.
    enriched.sort(key=lambda r: (-r["final_score"], _doc_id_sort_key(r["docID"])))
    return enriched[:top_k]


def compare_baseline_and_reranked(
    query_string: str,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    """Produce a side-by-side comparison of baseline vs. proximity re-ranking.

    Returns a dict with the baseline top-``top_k`` ordering, the re-ranked
    top-``top_k`` ordering (with proximity evidence), and ``changed`` — whether
    the ordering of docIDs actually differs. This is the honest evidence used by
    the UI and the evaluation harness: if a query produces no ranking change,
    ``changed`` is ``False`` and no improvement is fabricated.
    """
    baseline = get_model().query_vsm(query_string, top_k=top_k)
    reranked = rerank_with_proximity(
        query_string, top_k=top_k, candidate_k=candidate_k, alpha=alpha
    )
    baseline_order = [row["docID"] for row in baseline]
    reranked_order = [row["docID"] for row in reranked]
    return {
        "query": query_string,
        "alpha": alpha,
        "candidate_k": candidate_k,
        "top_k": top_k,
        "baseline": baseline,
        "reranked": reranked,
        "baseline_order": baseline_order,
        "reranked_order": reranked_order,
        "changed": baseline_order != reranked_order,
    }


# --------------------------------------------------------------------------
# Local test / demo:  python src/reranker.py
# --------------------------------------------------------------------------

def _print_comparison(comparison: dict) -> None:
    query = comparison["query"]
    print("=" * 72)
    print(f"query          : {query!r}")
    print(f"alpha          : {comparison['alpha']}   candidate_k: "
          f"{comparison['candidate_k']}   top_k: {comparison['top_k']}")
    print(f"ranking changed: {comparison['changed']}")
    if not comparison["reranked"]:
        print("  (no candidates — query has no lnc.ltc match)")
        return

    print(f"  {'rank':>4} | {'baseline':^10} | {'reranked':^10} | "
          f"{'cosine':>8} | {'prox':>6} | {'final':>8} | closest pair (gap)")
    print("  " + "-" * 84)
    baseline_order = comparison["baseline_order"]
    for i, row in enumerate(comparison["reranked"]):
        base_doc = baseline_order[i] if i < len(baseline_order) else "-"
        moved = " *" if base_doc != row["docID"] else "  "
        cp = row["closest_pair"]
        cp_text = (
            f"{cp['terms'][0]}/{cp['terms'][1]} (gap={cp['gap']})"
            if cp else "-"
        )
        print(
            f"  {i + 1:>4} | {base_doc:^10} | {row['docID']:^10} | "
            f"{row['cosine_score']:>8.4f} | {row['proximity_bonus']:>6.3f} | "
            f"{row['final_score']:>8.4f} | {cp_text}{moved}"
        )


def main() -> None:
    # A spread of legitimate multi-term queries drawn from the corpus so we can
    # inspect whether proximity re-ranking actually changes any orderings. We do
    # not fabricate improvements — queries that do not change are reported as
    # unchanged.
    demo_queries = [
        "cotton denim",   # changes: D053 (adjacent) rises over higher-cosine non-cooccurring docs
        "regular winter",  # changes: docs with the two terms closer together rise
        "jacket festive",  # changes: small proximity differences reshuffle near-tied cosines
        "cotton shirt",   # honest no-change case: proximity is uniform across the top group
    ]
    print(f"Proximity-aware re-ranking demo (alpha = {DEFAULT_ALPHA})")
    changed_examples = 0
    for query in demo_queries:
        comparison = compare_baseline_and_reranked(query)
        _print_comparison(comparison)
        if comparison["changed"]:
            changed_examples += 1
    print("=" * 72)
    print(
        f"{changed_examples}/{len(demo_queries)} demo queries had their "
        f"ranking changed by the proximity signal."
    )


if __name__ == "__main__":
    main()
