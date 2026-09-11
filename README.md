# CSD358 Information Retrieval — Assignment 1: Clothing Search Engine

A small classical Information Retrieval system built over a corpus of **100
clothing product descriptions** (`data/corpus_100.txt`).

The project implements, from first principles, a standard IR pipeline:

- **Pre-processing** — tokenization, case normalization, punctuation removal,
  Porter stemming, and a documented English stop-word policy (NLTK).
- **Inverted index** — terms with document frequency (df) and postings
  `(docID, term frequency)`.
- **Vector Space Model** — ranked retrieval using the **lnc.ltc** weighting
  scheme with cosine similarity (implemented directly, not via a library).
- **Positional index** — postings extended with term positions to support
  **exact phrase search** and **ordered proximity (`WITHIN/k`) search**.
- **Streamlit interface** — free-text search plus a phrase/proximity mode.
- **Evaluation harness** — free-text, phrase, and proximity test queries.

## Project structure

```
data/                 corpus_100.txt (100 product documents)
src/                  IR implementation modules
  preprocess.py       Part A: corpus reading + text pipeline
  index_builder.py    Part A: inverted index (df + postings)
  vsm.py              Part B: Vector Space Model (lnc.ltc cosine)
  positional_index.py Part C: positional index + phrase/proximity search
  reranker.py         Combines VSM ranking with positional evidence
  app.py              Part D: Streamlit interface
tests/
  run_evaluation.py   Part E: mandatory evaluation queries
output/               generated indexes, test results, and analysis
screenshots/          application / query screenshots
```

## Setup

```bash
pip install -r requirements.txt
python src/index_builder.py
```

The first command installs NLTK (and later Streamlit). The second parses the
100-document corpus, writes `output/inverted_index.json`, and writes
`output/doc_metadata.json` for later modules.

## Ranked retrieval — exact `lnc.ltc` (Part B)

`src/vsm.py` implements the Vector Space Model with the **`lnc.ltc`** SMART
weighting scheme by hand (no TF-IDF/BM25/embedding library hides the math).
`lnc.ltc` reads as `ddd.qqq` = (tf) . (df) . (normalization) for the
**d**ocument and **q**uery sides:

| side | scheme | tf | df | norm |
|------|--------|----|----|------|
| document | `lnc` | `1 + log10(tf)` | **none** | cosine |
| query | `ltc` | `1 + log10(tf_query)` | `log10(N / df)` | cosine |

Exact formulas (with **N = 100** fixed):

- Document weight: `w(d,t) = 1 + log10(tf)` for `tf > 0`, else `0`.
  **No IDF** is applied to documents — IDF is a collection-level term
  property and is applied exactly once (on the query) to avoid double-counting
  and to keep document weights independent of any query.
- Query weight: `w(q,t) = (1 + log10(tf_query)) * log10(N / df_t)`, where
  `df_t` comes straight from the inverted index. IDF on the query side lets
  rare words boost the ranking.
- Both vectors are **cosine-normalized** (divide by Euclidean norm) so that
  long documents do not win on length alone; similarity then depends on term
  *proportions*. After normalizing, `cosine = dot(doc_vec, query_vec)` — we do
  **not** divide by norms again.

Retrieval is efficient: document weights and per-document norms are
precomputed once, and each query only scores the **candidate documents** found
by taking the union of the query terms' postings lists (the only documents
that can score `> 0`). Results are the top 10, sorted by **decreasing cosine**,
ties broken by **increasing docID** (explicit, never dict order).

```bash
python src/vsm.py        # demo queries, top-10 results, and a manual hand-check
```

Deterministic edge-case behavior: empty / punctuation-only / all-stopword
queries and queries whose every term is outside the corpus return an empty
list; unknown terms have no `df` and contribute no score; repeated words and
words that stem to the same token raise that term's query `tf`.

The pure VSM baseline (`query_vsm(query_string, top_k=10)`) is kept
independent so it stays available for comparison against the later novelty
reranker.

## Status

Part A is in place: XML-style corpus parsing, a documented English stopword
policy with Porter stemming, and a deterministic inverted index (df + tf
postings) over all 100 documents. Part B is in place: exact `lnc.ltc` cosine
ranked retrieval in `src/vsm.py`. Parts C–E are still pending.
