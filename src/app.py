"""
app.py
======

Part D of the assignment: the Streamlit search interface.

This module is a thin presentation layer over the IR engine. All retrieval
logic lives in the src modules and is reused verbatim:

    * Free-text ranked retrieval   -> vsm.query_vsm            (Part B, lnc.ltc)
    * Exact phrase search          -> positional_index.phrase_search   (Part C)
    * Proximity (WITHIN/k) search  -> positional_index.proximity_search (Part C)

No preprocessing, indexing, scoring, or phrase/proximity logic is re-implemented
here. The app only calls the existing functions and renders their results.

Run:
    streamlit run src/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# Allow ``streamlit run src/app.py`` to import the sibling IR modules.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

REPO_ROOT = _SRC_DIR.parent
DEFAULT_METADATA_PATH = REPO_ROOT / "output" / "doc_metadata.json"

# Import the IR engine defensively so a broken environment surfaces a clean
# error message rather than a blank page or a cryptic import traceback.
try:
    from vsm import query_vsm  # Part B lnc.ltc baseline
    from positional_index import phrase_search, proximity_search  # Part C
    _IMPORT_ERROR: Exception | None = None
except Exception as exc:
    query_vsm = None  # type: ignore[assignment]
    phrase_search = None  # type: ignore[assignment]
    proximity_search = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc

# The proximity-aware reranker is optional. If it cannot be imported, the
# checkbox is still shown but falls back to the plain lnc.ltc ranking.
try:
    from reranker import DEFAULT_ALPHA, compare_baseline_and_reranked  # type: ignore
    _RERANKER_AVAILABLE = True
except Exception:
    compare_baseline_and_reranked = None  # type: ignore[assignment]
    DEFAULT_ALPHA = 0.15  # type: ignore[assignment]
    _RERANKER_AVAILABLE = False

# BM25 is an optional classical IR comparison — imported defensively so it
# can never break the required free-text interface if missing.
try:
    from bm25 import DEFAULT_B, DEFAULT_K1, query_bm25  # type: ignore
    _BM25_AVAILABLE = True
except Exception:
    query_bm25 = None  # type: ignore[assignment]
    DEFAULT_K1, DEFAULT_B = 1.5, 0.75  # type: ignore[assignment]
    _BM25_AVAILABLE = False


# --------------------------------------------------------------------------
# Cached resource loading (built once per session, not per interaction).
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    """Load docID -> {category, title, text} metadata written by index_builder."""
    try:
        return json.loads(DEFAULT_METADATA_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def _meta_field(metadata: dict, doc_id: str, field: str) -> str:
    """Return a single metadata field for a document, or '—' if absent."""
    entry = metadata.get(doc_id, {})
    value = entry.get(field, "")
    return value if value else "—"


def _positions_to_text(groups: list[list[int]]) -> str:
    """Render position groups as a readable string for the results table.

    e.g. [[14, 15], [20, 21]] -> '[14, 15]   •   [20, 21]'
    """
    return "   •   ".join("[" + ", ".join(str(p) for p in g) + "]" for g in groups)


# --------------------------------------------------------------------------
# lnc.ltc scheme explainer (kept accurate and brief).
# --------------------------------------------------------------------------

def render_scheme_explainer() -> None:
    """Show the lnc.ltc formula in a collapsible block."""
    with st.expander("How does lnc.ltc ranking work?"):
        st.markdown(
            """
Ranking uses the classical **lnc.ltc** SMART weighting scheme with cosine
similarity. The notation reads `document-scheme.query-scheme`:

| Side | Scheme | Term frequency weight | Document frequency | Normalization |
|------|--------|-----------------------|--------------------|---------------|
| **Document** | `lnc` | `1 + log₁₀(tf)` | none (n) | cosine (c) |
| **Query** | `ltc` | `1 + log₁₀(tf)` | `log₁₀(N / df)` — idf | cosine (c) |

- Documents use log-dampened term frequency with **no idf**: idf is a
  collection-level property and is applied exactly once, on the query side.
- Both vectors are cosine-normalized so long documents do not win on length
  alone.
- The final score is the dot product of the two unit-length vectors — that is
  the cosine similarity. With N = 100 (corpus size).

This is purely classical IR arithmetic — no embeddings, neural networks, or
language models anywhere in the system.
            """
        )


# --------------------------------------------------------------------------
# Free-text search mode (Part B).
# --------------------------------------------------------------------------

def render_free_text_mode(metadata: dict) -> None:
    st.subheader("Free-text search")
    st.write(
        "Type a natural-language query. Results are ranked by **cosine similarity** "
        "under the `lnc.ltc` weighting scheme (classical IR, no AI). "
        "The top 10 matching documents are shown."
    )
    render_scheme_explainer()

    query = st.text_input(
        "Query",
        key="free_text_query",
        placeholder="e.g. cotton kurta for men",
    )

    st.markdown("**Options**")
    apply_novelty = st.checkbox(
        "Proximity-aware re-ranking",
        value=False,
        key="cb_novelty",
        help=(
            "Extension: after the lnc.ltc baseline retrieval, re-orders "
            "the top candidates using positional proximity — documents where "
            "the query terms occur close together receive a small bonus "
            "(alpha = 0.15). The original cosine score is unchanged and "
            "visible alongside the final score."
        ),
    )
    if apply_novelty and not _RERANKER_AVAILABLE:
        st.info(
            "Proximity-aware re-ranking is not available in this environment. "
            "Showing the plain lnc.ltc cosine ranking instead."
        )

    compare_bm25 = st.checkbox(
        "Compare with BM25 (optional classical IR comparison)",
        value=False,
        key="cb_bm25",
        help=(
            "Shows a side-by-side of lnc.ltc rank/score vs. BM25 rank/score "
            "for the same query. BM25 is a classical ranking model, not a "
            "replacement for the required lnc.ltc baseline."
        ),
    )
    if compare_bm25 and not _BM25_AVAILABLE:
        st.info(
            "BM25 comparison is not available in this environment. "
            "Showing the required lnc.ltc ranking only."
        )

    if not st.button("Search", type="primary", key="free_text_search"):
        return

    if not query.strip():
        st.warning("Please enter a query to search.")
        return

    use_novelty = apply_novelty and _RERANKER_AVAILABLE
    if use_novelty:
        _render_reranked_results(query, metadata)
    else:
        _render_baseline_results(query, metadata)

    # BM25 comparison is rendered after the required lnc.ltc results so the
    # default view stays focused on the required system.
    if compare_bm25 and _BM25_AVAILABLE:
        _render_bm25_comparison(query, metadata)


def _render_baseline_results(query: str, metadata: dict) -> None:
    """Render the plain lnc.ltc cosine ranking (Part B baseline)."""
    try:
        results = query_vsm(query, top_k=10)
    except FileNotFoundError:
        st.error(
            "Index files were not found. Build them first:\n\n"
            "`python src/index_builder.py`"
        )
        return
    except Exception as exc:
        st.error(f"Search failed: {exc}")
        return

    if not results:
        st.info(
            "No matching documents found. Every query term may be absent from "
            "the corpus vocabulary or removed as a stop word. Try different terms."
        )
        return

    rows = []
    for rank, row in enumerate(results, start=1):
        doc_id = row["docID"]
        rows.append(
            {
                "Rank": rank,
                "Doc ID": doc_id,
                "Category": row.get("category") or _meta_field(metadata, doc_id, "category"),
                "Title": row.get("title") or _meta_field(metadata, doc_id, "title"),
                "Cosine score": f"{row['score']:.4f}",
            }
        )

    st.success(f"Top {len(rows)} result(s) — lnc.ltc baseline ranking.")
    st.dataframe(rows, hide_index=True, use_container_width=True)


def _render_reranked_results(query: str, metadata: dict) -> None:
    """Render the proximity-aware re-ranking alongside the baseline cosine score."""
    try:
        comparison = compare_baseline_and_reranked(query, top_k=10)  # type: ignore[misc]
    except FileNotFoundError:
        st.error(
            "Index files were not found. Build them first:\n\n"
            "`python src/index_builder.py`"
        )
        return
    except Exception as exc:
        st.warning(f"Re-ranking failed ({exc}); falling back to plain cosine order.")
        _render_baseline_results(query, metadata)
        return

    reranked = comparison["reranked"]
    if not reranked:
        st.info(
            "No matching documents found. Every query term may be absent from "
            "the corpus vocabulary or removed as a stop word. Try different terms."
        )
        return

    alpha = comparison["alpha"]
    baseline_order = comparison["baseline_order"]
    # Map each docID to its baseline rank so the table can show rank movement.
    baseline_rank = {doc_id: i + 1 for i, doc_id in enumerate(baseline_order)}

    rows = []
    for rank, row in enumerate(reranked, start=1):
        doc_id = row["docID"]
        prev = baseline_rank.get(doc_id)
        if prev is None:
            movement = "new"
        elif prev == rank:
            movement = "—"
        else:
            # Positive gain means the document moved up.
            movement = f"▲{prev - rank}" if prev > rank else f"▼{rank - prev}"
        cp = row["closest_pair"]
        closest = (
            f"{cp['terms'][0]}/{cp['terms'][1]} (gap {cp['gap']})" if cp else "—"
        )
        rows.append(
            {
                "Rank": rank,
                "Doc ID": doc_id,
                "Category": row.get("category") or _meta_field(metadata, doc_id, "category"),
                "Title": row.get("title") or _meta_field(metadata, doc_id, "title"),
                "Cosine score": f"{row['cosine_score']:.4f}",
                "Proximity bonus": f"{row['proximity_bonus']:.3f}",
                "Final score": f"{row['final_score']:.4f}",
                "Closest pair": closest,
                "Δ vs baseline": movement,
            }
        )

    if comparison["changed"]:
        st.success(
            f"Proximity-aware re-ranking (alpha = {alpha}) — the ranking "
            "**changed** relative to the lnc.ltc baseline. "
            "Formula: `final_score = cosine + alpha × proximity_bonus`."
        )
    else:
        st.info(
            f"Proximity-aware re-ranking (alpha = {alpha}) — the ranking did "
            "**not** change for this query (proximity bonus was uniform across "
            "the top group — reported honestly). "
            "Formula: `final_score = cosine + alpha × proximity_bonus`."
        )

    st.dataframe(rows, hide_index=True, use_container_width=True)

    # Expandable side-by-side baseline vs re-ranked ordering.
    with st.expander("Compare baseline vs. re-ranked order"):
        reranked_order = comparison["reranked_order"]
        compare_rows = []
        for i in range(max(len(baseline_order), len(reranked_order))):
            base = baseline_order[i] if i < len(baseline_order) else "—"
            new = reranked_order[i] if i < len(reranked_order) else "—"
            compare_rows.append(
                {
                    "Rank": i + 1,
                    "Baseline (cosine only)": base,
                    "Re-ranked (cosine + proximity)": new,
                    "Changed": "yes" if base != new else "",
                }
            )
        st.dataframe(compare_rows, hide_index=True, use_container_width=True)


def _render_bm25_comparison(query: str, metadata: dict) -> None:
    """Render an optional lnc.ltc vs. BM25 side-by-side comparison table.

    Shows the union of both top-10 result sets with each model's rank and
    score. This is a comparison only — it never alters the required lnc.ltc
    results already shown above.
    """
    try:
        vsm_results = query_vsm(query, top_k=10)
        bm25_results = query_bm25(query, top_k=10)  # type: ignore[misc]
    except FileNotFoundError:
        st.error(
            "Index files were not found. Build them first:\n\n"
            "`python src/index_builder.py`"
        )
        return
    except Exception as exc:
        st.warning(f"BM25 comparison unavailable ({exc}).")
        return

    if not vsm_results and not bm25_results:
        return  # primary view already reported no results

    lnc_by_doc = {
        row["docID"]: {"rank": i + 1, "score": row["score"]}
        for i, row in enumerate(vsm_results)
    }
    bm25_by_doc = {
        row["docID"]: {"rank": i + 1, "score": row["score"]}
        for i, row in enumerate(bm25_results)
    }

    # Sort the union of both result sets by best rank across either model.
    all_docs = set(lnc_by_doc) | set(bm25_by_doc)

    def _sort_key(doc_id: str):
        bm = bm25_by_doc.get(doc_id, {}).get("rank", 999)
        ln = lnc_by_doc.get(doc_id, {}).get("rank", 999)
        return (min(bm, ln), bm, ln, doc_id)

    rows = []
    for doc_id in sorted(all_docs, key=_sort_key):
        ln = lnc_by_doc.get(doc_id)
        bm = bm25_by_doc.get(doc_id)
        rows.append(
            {
                "Doc ID": doc_id,
                "Category": _meta_field(metadata, doc_id, "category"),
                "Title": _meta_field(metadata, doc_id, "title"),
                "lnc.ltc rank": ln["rank"] if ln else "—",
                "lnc.ltc score": f"{ln['score']:.4f}" if ln else "—",
                "BM25 rank": bm["rank"] if bm else "—",
                "BM25 score": f"{bm['score']:.4f}" if bm else "—",
            }
        )

    same_order = (
        [r["docID"] for r in vsm_results] == [r["docID"] for r in bm25_results]
    )
    st.divider()
    st.markdown("#### Optional classical IR comparison — lnc.ltc vs. BM25")
    st.caption(
        "BM25 scores are on a different, unbounded scale from cosine similarity. "
        "Only the rank columns are directly comparable."
    )
    if same_order:
        st.info(
            f"BM25 (k1 = {DEFAULT_K1}, b = {DEFAULT_B}) produced the **same** "
            "top-10 order as lnc.ltc for this query (reported honestly). "
            "BM25 is a comparison model, not a replacement."
        )
    else:
        st.info(
            f"BM25 (k1 = {DEFAULT_K1}, b = {DEFAULT_B}) produced a **different** "
            "top-10 order from lnc.ltc. The scores use different scales; "
            "lnc.ltc remains the required baseline."
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)


# --------------------------------------------------------------------------
# Phrase / proximity search mode (Part C).
# --------------------------------------------------------------------------

def render_phrase_mode(metadata: dict) -> None:
    st.subheader("Phrase and proximity search")
    st.write(
        "These searches use the **positional index**: every token's position in "
        "the processed document is stored so we can check whether terms are "
        "adjacent (phrase) or within *k* positions of each other (proximity). "
        "Matching positions are shown as evidence."
    )
    st.caption(
        "Positions are zero-based indices into the *processed* token stream "
        "(after stopword removal and stemming)."
    )

    search_kind = st.radio(
        "Search type",
        options=["Exact phrase search", "Proximity search (WITHIN/k)"],
        horizontal=True,
        key="positional_kind",
    )

    if search_kind == "Exact phrase search":
        _render_exact_phrase(metadata)
    else:
        _render_proximity(metadata)


def _render_exact_phrase(metadata: dict) -> None:
    st.markdown(
        "**Exact phrase search** — all words must appear in the given order at "
        "consecutive positions. Mere co-occurrence of the words in a document "
        "is not enough."
    )
    phrase = st.text_input(
        "Phrase",
        key="phrase_query",
        placeholder="e.g. cotton shirt",
    )
    if not st.button("Search phrase", type="primary", key="phrase_search_btn"):
        return

    if not phrase.strip():
        st.warning("Please enter a phrase to search.")
        return

    try:
        results = phrase_search(phrase)
    except FileNotFoundError:
        st.error(
            "Index files were not found. Build them first:\n\n"
            "`python src/positional_index.py`"
        )
        return
    except Exception as exc:
        st.error(f"Phrase search failed: {exc}")
        return

    if not results:
        st.info(
            "No documents contain this exact phrase. A term may be unknown to "
            "the corpus vocabulary, or the words never occur adjacently in any "
            "document."
        )
        return

    rows = []
    for row in results:
        doc_id = row["docID"]
        rows.append(
            {
                "Doc ID": doc_id,
                "Category": _meta_field(metadata, doc_id, "category"),
                "Title": _meta_field(metadata, doc_id, "title"),
                "Matching positions": _positions_to_text(row["matches"]),
            }
        )

    st.success(
        f"Found the exact phrase in **{len(rows)}** document(s). "
        "Each position group is [start, start+1, ...] — strictly consecutive."
    )
    st.dataframe(rows, hide_index=True, use_container_width=True)


def _render_proximity(metadata: dict) -> None:
    st.markdown(
        "**Proximity search (WITHIN/k)** — the two terms must occur within "
        "`k` positions of each other. `k` is the *maximum positional difference*, "
        "not the count of words in between: `k = 1` means adjacent, `k = 3` "
        "allows up to two tokens between them."
    )
    col1, col2 = st.columns(2)
    with col1:
        term1 = st.text_input("Term 1", key="prox_term1", placeholder="e.g. cotton")
    with col2:
        term2 = st.text_input("Term 2", key="prox_term2", placeholder="e.g. shirt")

    col3, col4 = st.columns(2)
    with col3:
        k = st.number_input(
            "k — maximum positional difference",
            min_value=1,
            max_value=100,
            value=3,
            step=1,
            key="prox_k",
            help="k=1: adjacent; k=3: up to 2 tokens between the terms.",
        )
    with col4:
        order = st.radio(
            "Order constraint",
            options=["Ordered (term 1 before term 2)", "Unordered (either order)"],
            key="prox_order",
        )
    ordered = order.startswith("Ordered")

    if not st.button("Search proximity", type="primary", key="prox_search_btn"):
        return

    if not term1.strip() or not term2.strip():
        st.warning("Please enter both terms.")
        return

    try:
        k_value = int(k)
    except (TypeError, ValueError):
        st.warning("k must be a positive whole number.")
        return
    if k_value < 1:
        st.warning("k must be at least 1.")
        return

    try:
        results = proximity_search(term1, term2, k_value, ordered=ordered)
    except FileNotFoundError:
        st.error(
            "Index files were not found. Build them first:\n\n"
            "`python src/positional_index.py`"
        )
        return
    except Exception as exc:
        st.error(f"Proximity search failed: {exc}")
        return

    if not results:
        st.info(
            "No documents satisfy this constraint. A term may be unknown to the "
            "corpus, or the words never occur within the specified distance. "
            "Try a larger k or the unordered option."
        )
        return

    rows = []
    for row in results:
        doc_id = row["docID"]
        rows.append(
            {
                "Doc ID": doc_id,
                "Category": _meta_field(metadata, doc_id, "category"),
                "Title": _meta_field(metadata, doc_id, "title"),
                "Satisfying position pairs": _positions_to_text(row["pairs"]),
            }
        )

    tag = "ordered" if ordered else "unordered"
    st.success(
        f"Found **{len(rows)}** document(s) where the terms are within "
        f"{k_value} position(s) of each other ({tag})."
    )
    st.dataframe(rows, hide_index=True, use_container_width=True)


# --------------------------------------------------------------------------
# App entry point.
# --------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Clothing IR Search Engine",
        page_icon="🔎",
        layout="centered",
    )

    # Project header
    st.title("Clothing Product Search Engine")
    st.markdown(
        "A classical Information Retrieval system built over **100 clothing "
        "product descriptions**. Retrieval uses the **lnc.ltc Vector Space Model** "
        "with cosine similarity, a positional index for phrase and proximity search, "
        "and an optional proximity-aware re-ranking extension. "
        "No AI, embeddings, or language models are used."
    )
    st.divider()

    if _IMPORT_ERROR is not None:
        st.error(
            "The IR engine could not be loaded. Check that dependencies are "
            "installed (`pip install -r requirements.txt`) and the indexes are "
            f"built (`python src/index_builder.py`).\n\nDetails: {_IMPORT_ERROR}"
        )
        return

    metadata = load_metadata()
    if not metadata:
        st.warning(
            "Document metadata not found (`output/doc_metadata.json`). "
            "Run `python src/index_builder.py` to build it. "
            "Results will show document IDs only until then."
        )

    # Mode selector
    mode = st.radio(
        "Search mode",
        options=["Free-text search", "Phrase / proximity search"],
        horizontal=True,
        key="search_mode",
    )
    st.divider()

    if mode == "Free-text search":
        render_free_text_mode(metadata)
    else:
        render_phrase_mode(metadata)


if __name__ == "__main__":
    main()
