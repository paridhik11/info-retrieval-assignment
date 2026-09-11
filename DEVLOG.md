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

## Entry 2 — Part A: preprocessing and inverted index

I implemented the corpus parser and the shared text pipeline in
`src/preprocess.py`, then built the inverted index in `src/index_builder.py`.

I parsed `data/corpus_100.txt` with ElementTree (wrapping the file in a
temporary root because it is a sequence of `<DOC>` blocks, not a rooted XML
document) so I would not depend on line offsets. TITLE + TEXT are the
indexable content; CATEGORY is kept only as display metadata, because the
title already names the garment type and indexing the category tag would
inflate tf for every document in that category.

The pipeline is lowercase → replace punctuation with spaces → whitespace
tokenize → NLTK English stopwords → Porter stem, in that order. I used the
stock English stopword list with no clothing-specific extras: terms such as
wear, fit, size, cotton, denim, shirt, kurta, dress, winter, and festive are
not English function words, so they stay searchable. I did hit one
stopword collision I chose not to special-case: size letters `s` and `m`
are in the NLTK list (from contractions), so they disappear, while `size`,
`l`, `xl`, and `xxl` remain. Porter also stems some product words more
aggressively than a clothing lexicon would (`festive` → `festiv`,
`leggings` → `leg`); I kept Porter as specified rather than adding
exceptions.

Running against the real corpus gave **N = 100**, no duplicate DOCIDs, and a
vocabulary of **124** stems. Spot-checks of cotton / denim / kurta matched
hand counts of stemmed TITLE+TEXT tokens (df is the posting-list length, not
collection frequency). Rebuilding the index twice produced identical JSON.
Outputs: `output/inverted_index.json` and `output/doc_metadata.json`.
