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

The first command installs NLTK and Streamlit. The second parses the
100-document corpus, writes `output/inverted_index.json`, and writes
`output/doc_metadata.json` for later modules.

### NLTK data

The pipeline needs the NLTK **English stop-word list**. `preprocess.py`
downloads it automatically on first use (`ensure_nltk_stopwords()`), so no
manual step is normally required. If your environment blocks that automatic
download, fetch it once by hand:

```bash
python -c "import nltk; nltk.download('stopwords')"
```

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

## Positional retrieval — phrase & proximity (Part C)

`src/positional_index.py` extends Part A's postings with **term positions** to
support **exact phrase search** and **ordered/unordered proximity
(`WITHIN/k`) search**. It **reuses the exact same preprocessing pipeline** as
Part A — it imports `process_documents` / `preprocess_text` from
`preprocess.py` and reads positions straight off the processed token list, so
there is no second tokenizer. `verify_consistency_with_inverted_index` asserts
the positional index has the **same vocabulary, df, and tf** as
`output/inverted_index.json`.

### Index structure

```
{
  "term": {
    "df": <number of documents containing the term>,
    "postings": {
      "D001": { "tf": <occurrences in doc>, "positions": [p1, p2, ...] },
      ...
    }
  }
}
```

`df` = documents containing the stem, `tf` = occurrences in the processed
document, `positions` = every position of the stem (sorted). Vocabulary keys
and posting docIDs are sorted, so `output/positional_index.json` is
deterministic.

### Position convention

Positions are **zero-based indices into the processed token stream** (the list
produced *after* lowercasing → punctuation split → stopword removal → Porter
stemming). `positions[i] = p` means `document.tokens[p]` is that term.

*Why this convention:* the processed token list **is** the sequence the
indexing pipeline emits, so `document.tokens[p]` recovers the token at
position `p` with no off-by-one translation, and zero-based means the stored
position is literally the Python list index. Indexing, phrase matching,
proximity matching, and displayed positions therefore all use the **one
identical coordinate system**. We deliberately do **not** use raw character
offsets and do **not** recompute positions from the original string after
stemming/stopword removal — after those steps raw offsets no longer line up
with the indexed tokens. Adjacency is defined on the *processed* stream: two
terms are adjacent iff their positions differ by exactly 1, even if an English
stopword sat between them in the surface text.

### Phrase matching (`phrase_search(phrase)`)

The phrase is normalized with the **same** pipeline. For an n-term phrase the
result is only documents where the normalized terms occur in **exact query
order at consecutive positions** — some position `p` holds term 0, `p+1` holds
term 1, …, `p+n-1` holds term n-1. Mere co-occurrence of both terms in a
document is **not** a match. Each result returns the evidence:

```
[ { "docID": "D...", "matches": [[p, p+1, ...], ...] } ]
```

Edge cases handled: empty / stopword-only phrase → `[]`; any unknown term →
`[]` (phrase impossible); one-term phrase → every occurrence position;
repeated terms (e.g. `cotton cotton`) probed at `p` and `p+1`.

### Proximity matching (`proximity_search(term1, term2, k, ordered=True)`)

`k` is the **maximum positional difference** between the two matched tokens —
**not** the number of intervening tokens. So `cotton WITHIN/3 shirt` (ordered)
keeps pairs with `0 < p2 - p1 <= 3` (adjacency is the `k = 1` case; `k = 3`
allows up to two tokens in between). Unordered keeps `0 < |p1 - p2| <= k`.
Every result carries the actual satisfying position pair(s):

```
[ { "docID": "D...", "pairs": [[p1, p2], ...] } ]
```

### Ordinary VSM vs positional retrieval

- **VSM (`lnc.ltc`, Part B)** is a **bag-of-words** model: a document is a
  multiset of term weights and word *order is discarded*. It answers *"how
  similar in term proportions is this document to the query?"* and ranks by
  cosine. It cannot tell `cotton shirt` from `shirt … cotton` — both raise the
  same term weights.
- **Positional retrieval (Part C)** keeps each term's **position list** and
  answers a *structural* question: *"do these terms occur adjacently
  (phrase) / within k positions (proximity)?"* It is boolean evidence, not a
  score, and returns the exact positions that satisfy the constraint.
- Concretely in this corpus, `D011` contains **both** `cotton` and `shirt`
  (so VSM/boolean co-occurrence would return it) but never as the adjacent
  phrase `cotton shirt`, so `phrase_search("cotton shirt")` correctly
  **excludes** it — demonstrating the two are not equivalent.

```bash
python src/positional_index.py   # builds the index, runs phrase/proximity demos
```

## Search interface (Part D)

`src/app.py` is a small **Streamlit** app — a clean demonstration of the IR
system, not a full website. It is a thin layer over the existing modules: it
imports `query_vsm` (Part B) and `phrase_search` / `proximity_search`
(Part C) and never re-implements preprocessing, indexing, or scoring. Indexes
and metadata are loaded once (cached) rather than rebuilt per interaction.

Two primary modes:

- **Free-text search** — ranked retrieval via `query_vsm()`. Shows the top 10
  results in a table (rank, docID, category, product title, and cosine score
  to 4 decimal places) in the exact required order. An optional novelty
  re-ranking control is present and will connect to the re-ranker once it is
  implemented (until then it transparently falls back to the cosine ranking).
- **Phrase / proximity search** — a selector between **exact phrase search**
  (`phrase_search()`) and **proximity search** (`proximity_search()`, with
  term 1 / term 2 / `k` / ordered-or-unordered controls). Both display the
  actual matching positions / satisfying position pairs, so the positional
  index is visibly in use.

The app also includes an accurate `lnc.ltc` explainer and handles empty
queries, unknown terms, no results, invalid `k`, and missing index files
gracefully (it never crashes on bad input).

### Run it

```bash
pip install -r requirements.txt   # installs streamlit (and nltk)
python src/index_builder.py       # builds output/inverted_index.json + doc_metadata.json
streamlit run src/app.py          # launches the interface at http://localhost:8501
```

### NLTK data

The pipeline uses the NLTK **English stop-word list** plus the Porter stemmer.
The stop-word corpus is downloaded automatically on first use by
`preprocess.ensure_nltk_stopwords()`, so no manual step is normally required.
If your environment blocks that automatic download, fetch it once beforehand:

```bash
python -c "import nltk; nltk.download('stopwords')"
```

## Status

Part A is in place: XML-style corpus parsing, a documented English stopword
policy with Porter stemming, and a deterministic inverted index (df + tf
postings) over all 100 documents. Part B is in place: exact `lnc.ltc` cosine
ranked retrieval in `src/vsm.py`. Part C is in place: a positional index with
exact phrase search and ordered/unordered `WITHIN/k` proximity search in
`src/positional_index.py` (`output/positional_index.json`). Part D is in
place: a Streamlit search interface (`src/app.py`) with free-text ranked
retrieval and a phrase/proximity mode that surfaces matching positions. Part E
(the evaluation harness) is still pending.
