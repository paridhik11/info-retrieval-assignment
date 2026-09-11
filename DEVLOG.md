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

## Entry 3 — Part B: `lnc.ltc` Vector Space Model

I implemented ranked retrieval in `src/vsm.py` using the exact `lnc.ltc`
weighting scheme required by the assignment, spelled out by hand rather than
delegated to any TF-IDF/BM25/embedding library. Document weights are
`1 + log10(tf)` with **no IDF** (IDF is a collection-level property, applied
exactly once on the query side to avoid squaring the boost); query weights are
`(1 + log10(tf_query)) * log10(N / df)` with **N = 100** fixed and `df` read
straight from the inverted index. Both the document and query vectors are
cosine-normalized and the score is their dot product — I do not divide by the
norms a second time.

For efficiency I precompute every document's term weights and its cosine norm
once at construction time from the inverted index, then each query only scores
the **candidate documents** obtained by unioning the postings lists of the
query terms (the only documents that can have a non-zero cosine). Results are
the top 10, sorted by decreasing cosine with an explicit increasing-docID
secondary sort so ranking never depends on dict/set iteration order.

I reused the Part A `preprocess_text` pipeline verbatim for queries, so query
terms and document terms are the same stems. I handled the edge cases
deterministically: empty / punctuation-only / all-stopword queries and queries
whose every term is outside the corpus return an empty list cleanly; unknown
terms carry no `df` and contribute no score; repeated words and words that stem
to the same token simply raise that term's query `tf`.

I verified the math by hand for the single-term query `denim`: `df = 15`, so
`idf = log10(100/15) = 0.8239`; a one-term query normalizes to weight `1.0`, so
each document's score reduces to `(1 + log10(tf)) / doc_norm`. My independent
recomputation matched `query_vsm` exactly (top document `D023`, score
`0.2157`). Representative queries (`cotton kurta`, `denim jeans`, `winter
jacket for men`) return sensible category-consistent top-10 lists, and the
score ties (e.g. structurally identical `D034`/`D094`) break by ascending
docID as intended. The pure VSM baseline is kept independent so it remains
available for the later novelty-reranker comparison.
