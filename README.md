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

## Status

Part A is in place: XML-style corpus parsing, a documented English stopword
policy with Porter stemming, and a deterministic inverted index (df + tf
postings) over all 100 documents. Parts B–E are still pending.
