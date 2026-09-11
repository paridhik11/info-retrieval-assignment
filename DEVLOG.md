# Development Log

## Entry 1 — Initial project setup

I set up the skeleton for the CSD358 Information Retrieval Assignment 1
clothing search engine today. I started from a repository that only contained
the assignment brief and the supplied `corpus_100.txt` (which I verified holds
exactly 100 product documents in the `<DOC>/<DOCID>/<CATEGORY>/<TITLE>/<TEXT>`
format).

I moved the corpus into `data/corpus_100.txt` so the data lives in one clear
place, and I laid out a modular `src/` package that mirrors the assignment
parts: `preprocess.py` (Part A pipeline), `index_builder.py` (inverted index),
`vsm.py` (lnc.ltc Vector Space Model), `positional_index.py` (positional index
with phrase/proximity search), `reranker.py` (combining VSM with positional
evidence), and `app.py` (the Streamlit interface). I also added
`tests/run_evaluation.py` for the mandatory Part E queries, plus `output/` and
`screenshots/` directories for deliverables.

At this stage the source files only document their responsibilities — I have
not implemented any IR logic yet. I kept dependencies deliberately minimal
(`nltk` for Porter stemming and stop-words, `streamlit` for the UI) and wrote
the README, requirements, and `.gitignore`. Next I will implement Part A: the
pre-processing pipeline and the inverted index.
