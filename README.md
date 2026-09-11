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
  evaluate.py         Part E: reproducible evaluation + analysis harness
tests/
  test_ir.py          Part E: behavioural pytest suite (36 tests)
  run_evaluation.py   Part E: thin wrapper that calls src/evaluate.py
output/               generated indexes, evaluation results, and analysis
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
  to 4 decimal places) in the exact required order. An optional
  **"Use proximity-aware re-ranking"** toggle switches to the novelty
  re-ranker (see below); unchecked, it uses the plain `lnc.ltc` baseline.
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

## Novelty: Proximity-Aware Re-Ranking

`src/reranker.py` adds a **lightweight, explainable retrieval enhancement**
that *extends* the classical system rather than replacing it. It uses only
information already produced by the required parts — the **`lnc.ltc` VSM**
(Part B) and the **positional index** (Part C). It introduces **no** LLMs,
embeddings, neural networks, vector databases, semantic/external search APIs,
pretrained models, BM25, or any different ranking algorithm. It is a classical
IR extension that uses positional evidence — not "AI", "semantic search", or
"machine learning".

**Why proximity is a useful signal.** The baseline `lnc.ltc` VSM is a
*bag-of-words* model: it weighs term importance and vector similarity but
discards word order, so it cannot tell whether two query terms occur *next to
each other* or *at opposite ends* of a document. Two documents can earn nearly
identical cosine scores while one packs the query terms tightly together
(usually the better match) and the other scatters them. The positional index
already knows *where* every term occurs, so we can reward candidates in which
the query terms co-occur closely.

**How it works (plain language).**

1. **Baseline** — `lnc.ltc` ranks documents using weighted term similarity
   (cosine). This is left completely unchanged (`query_vsm` is *called*, never
   modified) so the two rankings can be compared.
2. **Positions** — the positional index knows where each query term occurs in
   each document.
3. **Proximity** — for every pair of *distinct, known* query terms that both
   occur in a candidate, we take the minimum absolute positional gap and add
   `pair_bonus = 1 / (1 + min_gap)`; `proximity_bonus` is the sum over all
   valid pairs. Closer terms → larger bonus; a pair that never co-occurs
   contributes `0` (no fabricated evidence).
4. **Re-rank the VSM candidate set only** — we take the top `candidate_k` (20)
   VSM candidates and re-order them; we never pull in arbitrary documents
   outside that set.
5. **Preserve the baseline** — `final_score = cosine_score + alpha × proximity_bonus`,
   sorted by `final_score` descending, `docID` ascending for ties. The classical
   lexical score stays the primary ranking force.
6. **`alpha`** (default `0.15`, a configurable parameter) controls how strongly
   proximity affects ranking. It is kept **small on purpose** so the positional
   signal is a *controlled secondary signal* that only nudges near-tied
   documents rather than overwhelming the required `lnc.ltc` similarity. Setting
   `alpha = 0` recovers the exact baseline ordering.

This is a **project-level retrieval enhancement** combining lexical weighting
from `lnc.ltc` with positional proximity information. It is **not** a new
research algorithm and is **not** claimed to universally improve retrieval —
for some queries the proximity signal changes nothing, which the system reports
honestly.

Public API (the UI does not need to know the internals):

```python
rerank_with_proximity(query_string, top_k=10, candidate_k=20, alpha=0.15)
compare_baseline_and_reranked(query_string, top_k=10, candidate_k=20, alpha=0.15)
```

Each re-ranked result carries enough to *explain* the ranking in a viva:
`docID`, `title`, `category`, `cosine_score`, `proximity_bonus`, `final_score`,
and the `closest_pair` (which term pair and gap best explains the bonus).

```bash
python src/reranker.py   # baseline vs. re-ranked comparison on real corpus queries
```

**Worked example (`cotton denim`).** The baseline ranks `D043` (cosine
`0.2060`) above `D053` (cosine `0.1787`). But in `D053` the words *cotton* and
*denim* occur adjacently (gap 1 → bonus `0.5`) while in `D043` they never
co-occur (bonus `0`). With `alpha = 0.15`, `D053`'s final score `0.2537`
overtakes `D043`'s `0.2060`, so `D053` rises above the three higher-cosine but
non-co-occurring documents — exactly the behavior the proximity signal is meant
to add.

In the **Free-text search** UI, tick **"Use proximity-aware re-ranking"** to
switch from the baseline to the novelty. When enabled it shows the baseline
cosine score, the proximity bonus, the final score, the final rank, each
document's movement versus the baseline, and a compact baseline-vs-reranked
ordering comparison.

## Evaluation & analysis (Part E)

`src/evaluate.py` is the complete, reproducible evaluation harness. It **only
drives** the retrieval code that already exists — `preprocess_text`,
`query_vsm`/`VectorSpaceModel` (Part B), `phrase_search`/`proximity_search`
(Part C), and `rerank_with_proximity`/`compare_baseline_and_reranked` (the
novelty). It re-implements none of them. Every docID, score, and position in
the generated outputs is computed at run time; only the query *strings* are
declared, and the positional-analysis cases are **discovered** from the live
positional index, not hard-coded.

### Run it

```bash
python -m src.evaluate        # from the repo root (preferred)
# or
python src/evaluate.py        # equivalent
# or
python tests/run_evaluation.py  # thin wrapper, same result
```

Each run regenerates two files under `output/` (overwriting deterministically):

- **`output/evaluation_results.json`** — machine-readable results with clearly
  separated sections: `free_text_queries`, `phrase_queries`,
  `proximity_queries`, `unknown_term_query`, `reranking_comparisons`, and
  `positional_analysis` (plus an `evaluation_setup` header).
- **`output/evaluation_report.md`** — a human-readable report with result
  tables (free-text, phrase, proximity, reranking), the unknown-term behavior,
  the two positional-impact cases, and honest observations.

### What is evaluated

- **Free-text queries (≥ 10):** single-term, multi-term, several clothing
  categories, and descriptive concepts. Each is run through the required
  lnc.ltc VSM (top 10, descending score, ascending docID for ties). The
  proximity-aware reranked ranking is recorded **separately** — it never
  replaces the required lnc.ltc baseline.
- **Exact phrase queries (≥ 5):** e.g. `cotton shirt`, `stretch denim`,
  `festive wear`, `winter wear`, `regular fit`, `high waist`,
  `breathable fabric`. Each match records the actual consecutive positions and
  is checked to be strictly `+1`-consecutive in query order.
- **Proximity queries (≥ 3, different k):** ordered `WITHIN/k` searches
  (`cotton W/3 shirt`, `stretch W/4 denim`, `winter W/2 wear`, `high W/1 waist`,
  `slim W/3 fit`) with the satisfying `(p1, p2)` pairs (`0 < p2 - p1 <= k`).
- **Unknown / absent term query:** e.g. `corduroy blazer` (all terms absent)
  and `cotton corduroy` (known + absent). Confirms VSM, phrase, and proximity
  all handle absent terms cleanly (no crash, no invented matches, in-range
  scores). Results are produced by executing the retrieval code, not hard-coded.

### How the positional analysis works

Two genuine corpus cases are **derived from the actual positional index**:

- **Case 1 — phrase vs. co-occurrence.** The harness finds a phrase where the
  set of documents containing all terms *somewhere* strictly exceeds the exact
  phrase matches, then reports a concrete co-occurrence-only document with the
  real term positions (e.g. for `cotton shirt`, **15** docs co-occur but only
  **10** match the phrase; **D011** has `cotton` at 11 and `shirt` at 3/8/13,
  never adjacent).
- **Case 2 — proximity re-ranking.** The harness finds the first multi-term
  query whose ranking the reranker changes, then reports the risen document,
  the higher-cosine document it overtook, and the positional evidence (e.g.
  `cotton denim`: **D053** has `cotton`/`denim` adjacent — gap 1, bonus 0.5 —
  and rises from baseline rank 6 to reranked rank 3, overtaking **D043** which
  has higher cosine but the terms never co-occur).

### How the reranker is evaluated

Section 8 of the report compares baseline lnc.ltc against lnc.ltc + proximity
re-ranking for several multi-term queries, listing each document's baseline
rank, reranked rank, cosine score, proximity bonus, and final score. It states
plainly whether each query's order changed (e.g. `cotton denim`,
`regular winter`, `jacket festive` change; `cotton shirt` does not, because the
proximity bonus is uniform across the top group). No improvement is fabricated.

### Automated tests

`tests/test_ir.py` is a behavioural pytest suite (36 tests) covering the
corpus (N = 100, unique IDs, required fields), preprocessing (lowercasing,
punctuation, stopwords, stemming, index/query consistency), the inverted index
(df/tf correctness, valid postings, df ≠ collection frequency), the VSM
(unknown-term safety, cosine range, determinism, docID tie-break, top-k, and a
hand-recomputed lnc.ltc single-term check with no idf on documents), the
positional index (positions recover their tokens, tf = len(positions), phrase
consecutiveness, phrase order sensitivity, proximity respects k), and the
reranker (baseline untouched, `alpha = 0` recovers the baseline order,
deterministic bonuses, no bonus without positional evidence, and the
`final = cosine + alpha·bonus` formula).

```bash
python -m pytest tests/ -q
```

## Status

Part A is in place: XML-style corpus parsing, a documented English stopword
policy with Porter stemming, and a deterministic inverted index (df + tf
postings) over all 100 documents. Part B is in place: exact `lnc.ltc` cosine
ranked retrieval in `src/vsm.py`. Part C is in place: a positional index with
exact phrase search and ordered/unordered `WITHIN/k` proximity search in
`src/positional_index.py` (`output/positional_index.json`). Part D is in
place: a Streamlit search interface (`src/app.py`) with free-text ranked
retrieval and a phrase/proximity mode that surfaces matching positions. The
**novelty** — proximity-aware re-ranking (`src/reranker.py`) — is in place and
wired into the free-text UI behind a toggle, keeping the `lnc.ltc` baseline
available unchanged for comparison. Part E is in place: a reproducible
evaluation and analysis harness (`src/evaluate.py`) that regenerates
`output/evaluation_results.json` and `output/evaluation_report.md`, plus a
behavioural pytest suite (`tests/test_ir.py`, 36 tests, all passing).
