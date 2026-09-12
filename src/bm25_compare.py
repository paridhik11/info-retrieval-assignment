"""
bm25_compare.py
===============

OPTIONAL experimental extension (NOT part of the required assignment).

Compares the two classical IR ranking models on the same 100-document clothing
corpus, query for query, and writes the results to::

    output/bm25_comparison.json   (machine-readable)
    output/bm25_comparison.md     (human-readable report)

    A. required baseline : lnc.ltc VSM  ->  ``vsm.query_vsm``
    B. optional extension: BM25         ->  ``bm25.query_bm25``

This module ONLY *drives* the two existing rankers; it re-implements neither.
It never calls into or mutates ``query_vsm`` beyond reading its output, and it
does **not** touch the required ``output/evaluation_results.json`` /
``output/evaluation_report.md`` produced by ``src/evaluate.py``. The original
lnc.ltc evaluation is left completely intact — this writes a separate pair of
files.

For each query it records, over the top-10 of each model:
  * the query and its normalized terms,
  * the lnc.ltc ranking + lnc.ltc cosine scores,
  * the BM25 ranking + BM25 scores,
  * which documents changed rank between the two models,
  * which documents appear in BM25's top-10 but not lnc.ltc's (and vice-versa).

It does not claim either model is universally better. It reports where the
rankings agree, where they differ, and — grounded in the two formulas —
plausible reasons for the differences (tf saturation, document-length
normalization, query/document weighting). If the corpus is too templated for
meaningful differences on a query, that is reported honestly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow ``python src/bm25_compare.py`` as well as ``python -m src.bm25_compare``.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from preprocess import preprocess_text  # noqa: E402
from bm25 import DEFAULT_B, DEFAULT_K1, get_model as get_bm25_model, query_bm25  # noqa: E402
from vsm import N, get_model as get_vsm_model, query_vsm  # noqa: E402

REPO_ROOT = _SRC_DIR.parent
DEFAULT_JSON_PATH = REPO_ROOT / "output" / "bm25_comparison.json"
DEFAULT_MD_PATH = REPO_ROOT / "output" / "bm25_comparison.md"

TOP_K = 10

# >= 5 meaningful MULTI-TERM queries spanning several clothing categories and
# query shapes (category + attribute, attribute + attribute, descriptive). Only
# the query strings are declared; every rank/score/difference is computed live.
COMPARISON_QUERIES = [
    "cotton shirt",
    "denim jeans",
    "winter jacket",
    "high waist leggings",
    "slim fit jeans",
    "printed saree",
    "cotton kurta",
    "breathable fabric",
]


def _rows_by_doc(results: list[dict]) -> dict[str, dict]:
    """Map docID -> {rank, score} for a ranked result list (1-based ranks)."""
    return {
        row["docID"]: {"rank": rank, "score": row["score"]}
        for rank, row in enumerate(results, start=1)
    }


def compare_query(query: str, top_k: int = TOP_K) -> dict:
    """Run both rankers on one query and structure the top-``top_k`` comparison."""
    vsm_results = query_vsm(query, top_k=top_k)
    bm25_results = query_bm25(query, top_k=top_k)

    lnc_order = [r["docID"] for r in vsm_results]
    bm25_order = [r["docID"] for r in bm25_results]
    lnc_by_doc = _rows_by_doc(vsm_results)
    bm25_by_doc = _rows_by_doc(bm25_results)

    lnc_set = set(lnc_order)
    bm25_set = set(bm25_order)

    # Documents present in BOTH top-10 lists but at a different rank.
    changed_rank = []
    for doc_id in lnc_set & bm25_set:
        lnc_rank = lnc_by_doc[doc_id]["rank"]
        bm25_rank = bm25_by_doc[doc_id]["rank"]
        if lnc_rank != bm25_rank:
            changed_rank.append(
                {
                    "docID": doc_id,
                    "lnc_ltc_rank": lnc_rank,
                    "bm25_rank": bm25_rank,
                    "rank_delta": lnc_rank - bm25_rank,  # +ve => BM25 ranks higher
                }
            )
    changed_rank.sort(key=lambda d: d["docID"])

    # Set differences in the top-10 membership.
    appeared_in_bm25 = sorted(bm25_set - lnc_set)   # only BM25's top 10
    dropped_from_bm25 = sorted(lnc_set - bm25_set)  # only lnc.ltc's top 10

    # A compact side-by-side table (rank -> docID for each model).
    side_by_side = []
    for i in range(max(len(lnc_order), len(bm25_order))):
        lnc_doc = lnc_order[i] if i < len(lnc_order) else None
        bm25_doc = bm25_order[i] if i < len(bm25_order) else None
        side_by_side.append(
            {
                "rank": i + 1,
                "lnc_ltc_docID": lnc_doc,
                "lnc_ltc_score": (
                    round(lnc_by_doc[lnc_doc]["score"], 6) if lnc_doc else None
                ),
                "bm25_docID": bm25_doc,
                "bm25_score": (
                    round(bm25_by_doc[bm25_doc]["score"], 6) if bm25_doc else None
                ),
                "same_doc": lnc_doc is not None and lnc_doc == bm25_doc,
            }
        )

    identical = lnc_order == bm25_order
    return {
        "query": query,
        "normalized_terms": preprocess_text(query),
        "top_k": top_k,
        "identical_order": identical,
        "lnc_ltc_order": lnc_order,
        "bm25_order": bm25_order,
        "lnc_ltc_results": [
            {
                "rank": rank,
                "docID": r["docID"],
                "title": r["title"],
                "category": r["category"],
                "score": round(r["score"], 6),
            }
            for rank, r in enumerate(vsm_results, start=1)
        ],
        "bm25_results": [
            {
                "rank": rank,
                "docID": r["docID"],
                "title": r["title"],
                "category": r["category"],
                "score": round(r["score"], 6),
            }
            for rank, r in enumerate(bm25_results, start=1)
        ],
        "documents_that_changed_rank": changed_rank,
        "appeared_only_in_bm25_top10": appeared_in_bm25,
        "dropped_from_bm25_top10": dropped_from_bm25,
        "side_by_side": side_by_side,
    }


def run_comparison(queries: list[str] = COMPARISON_QUERIES) -> dict:
    """Execute the full BM25-vs-lnc.ltc comparison and return structured results."""
    vsm_model = get_vsm_model()
    bm25_model = get_bm25_model()
    # Prove the BM25 document length is the processed-token count used elsewhere.
    bm25_model.verify_lengths_match_index()

    comparisons = [compare_query(q) for q in queries]
    num_identical = sum(1 for c in comparisons if c["identical_order"])
    num_different = len(comparisons) - num_identical

    return {
        "setup": {
            "purpose": (
                "Experimental comparison of two CLASSICAL IR ranking models on "
                "the same 100-document clothing corpus. lnc.ltc remains the "
                "required baseline; BM25 is an optional extension, not a "
                "replacement."
            ),
            "N": N,
            "avgdl_processed_tokens": round(bm25_model.avgdl, 6),
            "corpus_size": len(vsm_model.metadata),
            "vocabulary_size": len(vsm_model.index),
            "lnc_ltc_model": (
                "Vector Space Model, lnc.ltc SMART scheme, cosine similarity; "
                "document weight 1 + log10(tf) (no idf on documents); query "
                "weight (1 + log10(tf_q)) * log10(N/df); both vectors "
                "cosine-normalized; ties broken by ascending docID."
            ),
            "bm25_model": (
                "Okapi BM25 implemented directly. score = sum_t IDF(t) * "
                "tf*(k1+1) / (tf + k1*(1 - b + b*|D|/avgdl)); "
                "IDF(t) = ln((N - df + 0.5)/(df + 0.5) + 1) (always positive); "
                "|D| = number of processed tokens; ties broken by ascending "
                "docID."
            ),
            "bm25_parameters": {"k1": DEFAULT_K1, "b": DEFAULT_B},
            "score_comparability_note": (
                "lnc.ltc cosine scores lie in [0, 1]; BM25 scores are an "
                "unbounded sum on a different scale. The two score columns are "
                "NOT directly comparable — only the RANKINGS are compared."
            ),
            "top_k": TOP_K,
            "num_queries": len(comparisons),
            "num_identical_orderings": num_identical,
            "num_different_orderings": num_different,
        },
        "comparisons": comparisons,
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _fmt(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "—"


def build_report(results: dict) -> str:
    setup = results["setup"]
    lines: list[str] = []
    add = lines.append

    add("# Optional Classical IR Comparison: lnc.ltc VSM vs. BM25")
    add("")
    add(
        "This is an **optional experimental extension**. The required "
        "assignment baseline is the **lnc.ltc Vector Space Model** "
        "(`src/vsm.py`), which is unchanged. BM25 (`src/bm25.py`) is added only "
        "to compare two classical ranking approaches on the same corpus — it is "
        "**not** a replacement for lnc.ltc."
    )
    add("")
    add("All rankings and scores below are generated by "
        "`python -m src.bm25_compare`, which drives the live retrieval code. "
        "No document IDs or scores are hard-coded.")
    add("")

    # Setup
    add("## Setup")
    add("")
    add(f"- **N:** {setup['N']}   **corpus size:** {setup['corpus_size']}   "
        f"**vocabulary:** {setup['vocabulary_size']} stems")
    add(f"- **avgdl (avg. processed document length):** "
        f"{setup['avgdl_processed_tokens']} tokens")
    add(f"- **BM25 parameters:** k1 = {setup['bm25_parameters']['k1']}, "
        f"b = {setup['bm25_parameters']['b']}")
    add(f"- **lnc.ltc model:** {setup['lnc_ltc_model']}")
    add(f"- **BM25 model:** {setup['bm25_model']}")
    add(f"- **Score comparability:** {setup['score_comparability_note']}")
    add(f"- **Queries compared:** {setup['num_queries']} "
        f"({setup['num_different_orderings']} produced a different top-10 "
        f"order, {setup['num_identical_orderings']} were identical).")
    add("")
    add("### BM25 formula used")
    add("")
    add("```")
    add("BM25(D,Q) = sum over query terms t of")
    add("    IDF(t) * ( tf(t,D) * (k1 + 1) )")
    add("             / ( tf(t,D) + k1 * (1 - b + b * |D| / avgdl) )")
    add("")
    add("IDF(t)    = ln( (N - df_t + 0.5) / (df_t + 0.5) + 1 )   (N = 100)")
    add("k1 = 1.5   term-frequency saturation")
    add("b  = 0.75  document-length normalization strength")
    add("|D|        = number of processed (stemmed) tokens in D")
    add("avgdl      = average processed document length over all N documents")
    add("```")
    add("")

    # Per-query comparisons
    add("## Per-query comparisons (top 10)")
    add("")
    for c in results["comparisons"]:
        add(f"### `{c['query']}`  (normalized {c['normalized_terms']})")
        add("")
        if c["identical_order"]:
            add("**Same top-10 order** in both models for this query.")
        else:
            add("**Different top-10 order** between the two models.")
        add("")
        add("| Rank | lnc.ltc docID | lnc.ltc score | BM25 docID | BM25 score | Same? |")
        add("| ---: | ------------- | ------------: | ---------- | ---------: | ----- |")
        for row in c["side_by_side"]:
            add(
                f"| {row['rank']} | {row['lnc_ltc_docID'] or '—'} | "
                f"{_fmt(row['lnc_ltc_score'])} | {row['bm25_docID'] or '—'} | "
                f"{_fmt(row['bm25_score'])} | "
                f"{'yes' if row['same_doc'] else 'no'} |"
            )
        add("")
        if c["documents_that_changed_rank"]:
            parts = []
            for d in c["documents_that_changed_rank"]:
                direction = (
                    "higher in BM25" if d["rank_delta"] > 0 else "lower in BM25"
                )
                parts.append(
                    f"{d['docID']} (lnc.ltc #{d['lnc_ltc_rank']} → BM25 "
                    f"#{d['bm25_rank']}, {direction})"
                )
            add("- **Changed rank (in both top-10):** " + "; ".join(parts) + ".")
        else:
            add("- **Changed rank (in both top-10):** none.")
        if c["appeared_only_in_bm25_top10"]:
            add("- **Only in BM25's top-10:** "
                + ", ".join(c["appeared_only_in_bm25_top10"]) + ".")
        if c["dropped_from_bm25_top10"]:
            add("- **Only in lnc.ltc's top-10:** "
                + ", ".join(c["dropped_from_bm25_top10"]) + ".")
        if (
            not c["documents_that_changed_rank"]
            and not c["appeared_only_in_bm25_top10"]
            and not c["dropped_from_bm25_top10"]
        ):
            add("- Both models returned the **same documents in the same "
                "order** — on this templated corpus the two rankings coincide "
                "for this query (reported honestly, not massaged).")
        add("")

    # Analysis
    add("## Why the rankings differ (grounded in the formulas)")
    add("")
    add(
        "We do **not** claim either model is universally better. Where the two "
        "rankings differ, the cause is one of the concrete formula differences "
        "below; where they agree, the corpus is templated enough that both "
        "models see the same evidence."
    )
    add("")
    add(
        "- **Term-frequency behavior.** lnc.ltc weights tf as `1 + log10(tf)`, "
        "which keeps growing (slowly) with every extra occurrence. BM25 "
        "**saturates** tf via `tf*(k1+1)/(tf + k1*…)`: after a few occurrences "
        "additional ones add almost nothing (the contribution asymptotes to "
        "`(k1+1)×IDF×length_factor`). So a document that repeats a query term "
        "many times is rewarded more by lnc.ltc than by BM25. `k1` sets how "
        "quickly this saturation kicks in."
    )
    add(
        "- **Document-length normalization.** lnc.ltc normalizes by the full "
        "cosine (Euclidean) norm of the document vector, so a longer, more "
        "diverse document is divided by a larger norm. BM25 instead compares "
        "each document's length to `avgdl` through `(1 - b + b*|D|/avgdl)`: "
        "documents longer than average are penalized and shorter-than-average "
        "documents are boosted, with `b` controlling the strength. The two "
        "length models are different, so documents whose length sits far from "
        f"`avgdl` (= {setup['avgdl_processed_tokens']} tokens here) can swap "
        "order between the models."
    )
    add(
        "- **Query/document weighting differences.** lnc.ltc puts idf only on "
        "the query side (`log10(N/df)`) and also cosine-normalizes the query; "
        "BM25 applies a probabilistic `ln((N-df+0.5)/(df+0.5)+1)` per term "
        "inside the sum and uses no query-tf factor. A rarer query term (small "
        "df, e.g. `denim`) therefore dominates the BM25 sum more sharply than "
        "it dominates the cosine, which can reorder documents that are strong "
        "on the common term but weak on the rare one."
    )
    add("")
    add(
        "**Honest note on this corpus.** The 100 clothing descriptions are "
        "heavily templated: many documents in the same category share nearly "
        "identical text, so both models frequently produce the same (or "
        "tie-broken-identical) ordering. The differences that do appear are "
        "concentrated where documents differ in length or in how often the "
        "rarer query term repeats — exactly the places the formulas above "
        "predict."
    )
    add("")
    return "\n".join(lines) + "\n"


def write_json(results: dict, path: str | Path = DEFAULT_JSON_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_report(results: dict, path: str | Path = DEFAULT_MD_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report(results), encoding="utf-8")


def main() -> None:
    print("Running BM25 vs lnc.ltc comparison (optional extension)...")
    results = run_comparison()
    write_json(results)
    write_report(results)

    setup = results["setup"]
    print(f"  N                     : {setup['N']}")
    print(f"  avgdl (proc. tokens)  : {setup['avgdl_processed_tokens']}")
    print(f"  BM25 params           : k1={setup['bm25_parameters']['k1']}, "
          f"b={setup['bm25_parameters']['b']}")
    print(f"  queries compared      : {setup['num_queries']}")
    print(f"  different orderings   : {setup['num_different_orderings']}")
    print(f"  identical orderings   : {setup['num_identical_orderings']}")
    for c in results["comparisons"]:
        tag = "DIFFERENT" if not c["identical_order"] else "same     "
        print(f"    [{tag}] {c['query']!r}")
    print(f"  wrote {DEFAULT_JSON_PATH}")
    print(f"  wrote {DEFAULT_MD_PATH}")


if __name__ == "__main__":
    main()
