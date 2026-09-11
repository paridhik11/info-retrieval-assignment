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

## Entry 4 — Part C: positional index, phrase & proximity search

I implemented `src/positional_index.py`: a positional index plus exact phrase
search and `WITHIN/k` proximity search. The hard requirement here was to
**reuse the exact Part A preprocessing pipeline** and not spawn a second
tokenizer. I did this by importing `process_documents` / `preprocess_text`
from `preprocess.py` and reading positions directly off each document's
processed token list `document.tokens` via `enumerate` — the very list Part A
counted tf/df from. To make the reuse provable rather than assumed, I added
`verify_consistency_with_inverted_index`, which asserts the positional index
has the **same vocabulary, same df, and same tf** as `output/inverted_index.json`;
it passes (124 stems, identical df/tf). I also added
`verify_positions_match_tokens`, which asserts every stored position `p`
recovers its own term via `document.tokens[p]` — the direct proof that
positions are processed-stream indices, not character offsets.

**Position convention (the one explicit decision):** positions are
**zero-based indices into the processed token stream** (after lowercase →
punctuation split → stopword removal → Porter stem). I picked this because the
processed list *is* what the pipeline emits, so `document.tokens[p]` recovers
the token with no off-by-one translation, and zero-based means the stored
number is literally the Python list index. Indexing, phrase matching,
proximity matching, and displayed positions therefore all share one coordinate
system. I explicitly do **not** use raw character offsets and do **not**
recompute positions from the original string after stemming/stopword removal,
because after those steps raw offsets no longer align with the indexed tokens.
Adjacency is defined on the processed stream, so a dropped stopword between two
surface words still leaves them adjacent (positions differ by 1).

**Phrase search** normalizes the phrase with the same pipeline, intersects the
candidate documents, and for an n-term phrase keeps only positions `p` where
term `i` sits at `p+i` for all `i`. It returns the actual consecutive positions
as evidence. I handled the required edge cases: empty/stopword-only phrase →
`[]`; any unknown term → `[]`; one-term phrase → each occurrence; repeated
terms like `cotton cotton` probed at `p` and `p+1`.

**Proximity search** takes `ordered` (default): `0 < p2 - p1 <= k`, and
unordered: `0 < |p1 - p2| <= k`, returning the satisfying `(p1, p2)` pairs. I
documented that **`k` is the maximum positional difference, not the count of
intervening tokens** (so `WITHIN/1` = adjacency, `WITHIN/3` = up to two tokens
between).

Testing against the assignment examples: `cotton shirt`, `stretch denim`,
`festive wear`, `winter wear`, and `regular fit` all return true consecutive
matches (e.g. `cotton shirt` at `[14, 15]` in D001; `regular fit` at `[1, 2]`
in D003). The proximity examples `cotton WITHIN/3 shirt`, `stretch WITHIN/4
denim`, and `winter WITHIN/3 wear` all return real position pairs, while
`festive WITHIN/4 kurta` returns **nothing** — I left that honest rather than
fabricating a hit, since the corpus is synthetic. Crucially, `cotton shirt`
gives a clean co-occurrence-vs-phrase counterexample: **15** documents contain
both `cotton` and `shirt` somewhere, but only **10** contain the adjacent
phrase — e.g. **D011** ("...Cotton Shirt..." title stems put `shirt` at
`[3, 8, 13]` and a later `cotton` at `[11]`, never adjacent), so a boolean/VSM
"both terms present" test would wrongly return it while phrase search correctly
excludes it. Phrase and proximity search operate purely on the positional
index; they never call the VSM. `output/positional_index.json` is regenerated
deterministically (rebuilding twice yields byte-identical JSON). I did not
touch the Part B `lnc.ltc` mathematics.

## Entry 5 — Part D: Streamlit search interface

I implemented the user-facing search app in `src/app.py`. The guiding
principle was that the UI is a **thin presentation layer**: it imports
`query_vsm` (Part B) and `phrase_search` / `proximity_search` (Part C) and
does not re-implement any preprocessing, indexing, scoring, or phrase/proximity
logic. I deliberately did not duplicate the pipeline in the app — keeping one
source of truth in `src/` was the whole point of the earlier modular layout.

The app has the two required primary modes, chosen with a radio at the top:

- **Free-text search** — a title, a short description, a query box, a Search
  button, and an optional novelty re-ranking checkbox. It calls
  `query_vsm(query, top_k=10)` and renders the top 10 in a clean table with
  rank, docID, category, product title, and the cosine score formatted to 4
  decimal places, in the exact order the VSM returns (I only enumerate for the
  rank column; I never re-sort). No raw dicts are shown. The novelty control
  tries to import `reranker.rerank_results`; since Part D+ isn't implemented
  yet, `_RERANKER_AVAILABLE` is `False`, so the control is present but
  transparently falls back to the plain cosine ranking with an info note. This
  keeps the wiring ready for the next step without faking behavior.
- **Phrase / proximity search** — a sub-selector between **exact phrase
  search** (one input → `phrase_search`) and **proximity search** (term 1,
  term 2, a numeric `k`, an ordered/unordered radio → `proximity_search`).
  Both display the **actual matching positions / satisfying position pairs**,
  which is the explicit assignment requirement that the positional index be
  visibly in use. I render positions as readable strings, never as Python
  lists-in-dicts.

I added an accurate `lnc.ltc` explainer in an expander: documents use log-tf
with no idf, queries use log-tf × idf, both cosine-normalized, idf applied once
on the query side, N = 100. I kept it truthful rather than the common
hand-wave that "tf-idf is applied to both sides".

Efficiency: metadata is loaded through `st.cache_data` and the VSM model /
positional index keep their existing module-level lazy caches, so indexes load
once per session instead of rebuilding the corpus on every click. Positional
results only carry docIDs, so I look up category/title from
`output/doc_metadata.json` for display, degrading to docID-only if metadata is
absent.

Error handling was a focus: empty query, empty phrase, or missing proximity
terms produce warnings; `k < 1` is rejected; unknown terms and no-result cases
show a friendly info message; a missing index file shows a "build the indexes"
error instead of a stack trace. The IR-module imports are wrapped so a broken
environment surfaces one clean error rather than a blank page.

I installed Streamlit (already listed in `requirements.txt`), byte-compiled the
app, and ran a smoke test driving the same functions the UI calls: `cotton
kurta` returns the expected Kurta top-10 (D034/D094 tied at 0.2467, broken by
docID); `cotton shirt` phrase search returns consecutive-position matches
(e.g. D001 `[14, 15]`); `cotton WITHIN/3 shirt` returns real pairs; and all
edge cases (empty, unknown term, `k = 0`) return `[]` without crashing. I also
launched the app headless to confirm it boots and serves with no runtime
errors. I documented the `streamlit run src/app.py` command and the NLTK
stop-word data note in the README. I did not modify any Part B/C module to
simplify the UI.
