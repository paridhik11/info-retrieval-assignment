"""
app.py
======

Part D of the assignment: the Streamlit search interface.

This module is a *thin* presentation layer over the IR engine. All retrieval
logic lives in the ``src`` modules and is reused verbatim:

    * Free-text ranked retrieval  -> ``vsm.query_vsm``            (Part B, lnc.ltc)
    * Exact phrase search         -> ``positional_index.phrase_search``   (Part C)
    * Proximity (WITHIN/k) search -> ``positional_index.proximity_search`` (Part C)

No preprocessing, indexing, scoring, or phrase/proximity logic is re-implemented
here — the app only calls the existing functions and renders their results.

Design goals:
    * Two primary modes: free-text search and phrase/proximity search.
    * Show the evidence the assignment requires: cosine scores for ranked
      retrieval and actual matching positions for positional retrieval.
    * Never expose raw Python dictionaries; render clean tables instead.
    * Load the indexes once (cached) rather than rebuilding per interaction.
    * Fail gracefully on empty queries, unknown terms, no results, invalid k,
      and missing index files — the app must not crash.

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

# The IR engine imports can fail if the environment is broken (e.g. NLTK data
# missing at import time); surface that as a clean error later rather than a
# blank page.
try:
    from vsm import query_vsm  # Part B
    from positional_index import phrase_search, proximity_search  # Part C
    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - defensive UI guard
    query_vsm = None  # type: ignore[assignment]
    phrase_search = None  # type: ignore[assignment]
    proximity_search = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc

# The novelty re-ranker (Part D+) is optional. If it is not implemented yet we
# still show the control, but transparently fall back to the plain VSM ranking.
try:
    from reranker import rerank_results as _rerank_results  # type: ignore
    _RERANKER_AVAILABLE = True
except Exception:
    _rerank_results = None  # type: ignore[assignment]
    _RERANKER_AVAILABLE = False


# --------------------------------------------------------------------------
# Cached resource loading (built once per session, not per interaction).
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    """Load docID -> {category, title, text} display metadata (Part A output).

    Cached so we read the JSON once. Returns an empty dict if the file is
    missing; callers degrade to showing just the docID.
    """
    try:
        return json.loads(DEFAULT_METADATA_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def _meta_field(metadata: dict, doc_id: str, field: str) -> str:
    entry = metadata.get(doc_id, {})
    value = entry.get(field, "")
    return value if value else "—"


def _positions_to_text(groups: list[list[int]]) -> str:
    """Render a list of position groups as a readable, non-dict string.

    e.g. ``[[14, 15], [20, 21]]`` -> ``"[14, 15]  •  [20, 21]"``.
    """
    return "   •   ".join("[" + ", ".join(str(p) for p in g) + "]" for g in groups)


# --------------------------------------------------------------------------
# lnc.ltc explanation (kept accurate, not misleading).
# --------------------------------------------------------------------------

def render_scheme_explainer() -> None:
    with st.expander("What does the `lnc.ltc` weighting mean?"):
        st.markdown(
            """
Free-text ranking uses the classic **`lnc.ltc`** SMART weighting scheme with
cosine similarity. The notation is `ddd.qqq` — the first triple is the
**document** side, the second is the **query** side, each read as
*(term-frequency) . (document-frequency) . (normalization)*:

| Side | Scheme | Term frequency | Document frequency | Normalization |
|------|--------|----------------|--------------------|----------------|
| **Document** | `lnc` | `1 + log₁₀(tf)` | none | cosine |
| **Query** | `ltc` | `1 + log₁₀(tf)` | `log₁₀(N / df)` (idf) | cosine |

- **Documents** use **log term-frequency** weighting with **no idf**.
- **Queries** use **log term-frequency × idf**.
- **Both** vectors are **cosine-normalized**, and the score is their dot
  product (the cosine similarity).

idf is applied **once**, on the query side, so a rare word still boosts the
ranking without double-counting. Cosine normalization stops long documents
from winning on length alone. Here `N = 100` (the corpus size).
            """
        )


# --------------------------------------------------------------------------
# Free-text search mode (Part B).
# --------------------------------------------------------------------------

def render_free_text_mode(metadata: dict) -> None:
    st.subheader("Free-text search")
    st.write(
        "Type a natural-language query. Documents are ranked by cosine "
        "similarity under the `lnc.ltc` scheme and the top 10 are shown."
    )
    render_scheme_explainer()

    query = st.text_input(
        "Search query",
        key="free_text_query",
        placeholder="e.g. cotton kurta for men",
    )

    apply_novelty = st.checkbox(
        "Apply novelty re-ranking",
        value=False,
        help=(
            "Re-orders results using positional evidence on top of the "
            "cosine ranking. Connects to the re-ranker when it is available."
        ),
    )
    if apply_novelty and not _RERANKER_AVAILABLE:
        st.info(
            "Novelty re-ranking is not available yet — showing the plain "
            "`lnc.ltc` cosine ranking. This control will activate once the "
            "re-ranker is implemented."
        )

    search = st.button("Search", type="primary", key="free_text_search")
    if not search:
        return

    if not query.strip():
        st.warning("Please enter a query to search.")
        return

    try:
        results = query_vsm(query, top_k=10)
    except FileNotFoundError:
        st.error(
            "Index files were not found. Please build the indexes first:\n\n"
            "`python src/index_builder.py`"
        )
        return
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(f"Search failed: {exc}")
        return

    if apply_novelty and _RERANKER_AVAILABLE and _rerank_results is not None:
        try:
            results = _rerank_results(query, results)
        except Exception as exc:  # pragma: no cover - defensive UI guard
            st.warning(f"Re-ranking unavailable ({exc}); showing cosine order.")

    if not results:
        st.info(
            "No matching documents. Every query term may be unknown to the "
            "corpus or filtered out as a stop word — try different terms."
        )
        return

    # Preserve the exact ranking order returned by query_vsm.
    rows = []
    for rank, row in enumerate(results, start=1):
        doc_id = row["docID"]
        rows.append(
            {
                "Rank": rank,
                "Doc ID": doc_id,
                "Category": row.get("category") or _meta_field(metadata, doc_id, "category"),
                "Product title": row.get("title") or _meta_field(metadata, doc_id, "title"),
                "Cosine score": f"{row['score']:.4f}",
            }
        )

    st.success(f"Showing top {len(rows)} result(s).")
    st.dataframe(rows, hide_index=True, use_container_width=True)


# --------------------------------------------------------------------------
# Phrase / proximity search mode (Part C).
# --------------------------------------------------------------------------

def render_phrase_mode(metadata: dict) -> None:
    st.subheader("Phrase / proximity search")
    st.write(
        "These searches use the **positional index**. Matching term positions "
        "are shown as evidence that positions — not just word presence — drive "
        "the result. Positions are zero-based indices into each document's "
        "processed token stream."
    )

    search_kind = st.radio(
        "Search type",
        options=["Exact phrase search", "Proximity search"],
        horizontal=True,
        key="positional_kind",
    )

    if search_kind == "Exact phrase search":
        _render_exact_phrase(metadata)
    else:
        _render_proximity(metadata)


def _render_exact_phrase(metadata: dict) -> None:
    st.markdown("**Exact phrase search** — terms must appear in order at "
                "consecutive positions.")
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
            "Index files were not found. Please build the indexes first:\n\n"
            "`python src/positional_index.py`"
        )
        return
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(f"Phrase search failed: {exc}")
        return

    if not results:
        st.info(
            "No documents contain this exact phrase. A term may be unknown to "
            "the corpus, or the words never occur adjacently."
        )
        return

    rows = []
    for row in results:
        doc_id = row["docID"]
        rows.append(
            {
                "Doc ID": doc_id,
                "Category": _meta_field(metadata, doc_id, "category"),
                "Product title": _meta_field(metadata, doc_id, "title"),
                "Matching positions": _positions_to_text(row["matches"]),
            }
        )

    st.success(f"Found the phrase in {len(rows)} document(s).")
    st.dataframe(rows, hide_index=True, use_container_width=True)


def _render_proximity(metadata: dict) -> None:
    st.markdown(
        "**Proximity search** — the two terms must occur within `k` positions "
        "of each other. `k` is the *maximum positional difference* "
        "(so `k = 1` means adjacent)."
    )
    col1, col2 = st.columns(2)
    with col1:
        term1 = st.text_input("Term 1", key="prox_term1", placeholder="e.g. cotton")
    with col2:
        term2 = st.text_input("Term 2", key="prox_term2", placeholder="e.g. shirt")

    col3, col4 = st.columns(2)
    with col3:
        k = st.number_input(
            "k (maximum positional difference)",
            min_value=1,
            max_value=100,
            value=3,
            step=1,
            key="prox_k",
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
        st.warning("Please enter both terms to search.")
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
            "Index files were not found. Please build the indexes first:\n\n"
            "`python src/positional_index.py`"
        )
        return
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(f"Proximity search failed: {exc}")
        return

    if not results:
        st.info(
            "No documents satisfy this proximity constraint. A term may be "
            "unknown to the corpus, or the words never occur within `k` "
            "positions — try a larger `k` or the unordered option."
        )
        return

    rows = []
    for row in results:
        doc_id = row["docID"]
        rows.append(
            {
                "Doc ID": doc_id,
                "Category": _meta_field(metadata, doc_id, "category"),
                "Product title": _meta_field(metadata, doc_id, "title"),
                "Satisfying position pairs": _positions_to_text(row["pairs"]),
            }
        )

    tag = "ordered" if ordered else "unordered"
    st.success(
        f"Found {len(rows)} document(s) with the terms within {k_value} "
        f"position(s) ({tag})."
    )
    st.dataframe(rows, hide_index=True, use_container_width=True)


# --------------------------------------------------------------------------
# App entry point.
# --------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Clothing Search Engine", page_icon="🔎")
    st.title("Clothing Search Engine")
    st.caption(
        "A classical Information Retrieval demo over 100 clothing product "
        "descriptions — Vector Space Model ranking plus positional "
        "phrase/proximity search."
    )

    if _IMPORT_ERROR is not None:
        st.error(
            "The IR engine could not be loaded. Make sure dependencies are "
            "installed (`pip install -r requirements.txt`) and the indexes are "
            f"built.\n\nDetails: {_IMPORT_ERROR}"
        )
        return

    metadata = load_metadata()
    if not metadata:
        st.warning(
            "Document metadata was not found (`output/doc_metadata.json`). "
            "Results will show document IDs only. Build it with "
            "`python src/index_builder.py`."
        )

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
