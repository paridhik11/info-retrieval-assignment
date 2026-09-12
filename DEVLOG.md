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
delegated to any TF-IDF/embedding library. Document weights are
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

## Entry 6 — Novelty: proximity-aware re-ranking

I implemented the novelty in `src/reranker.py`. It is a deliberately
**lightweight, explainable** enhancement that *extends* the required classical
system rather than replacing it: it introduces no LLMs, embeddings, neural
networks, vector databases, semantic/external APIs, pretrained models, or any
different ranking algorithm. It reuses only what the assignment already
built — the `lnc.ltc` VSM (Part B) and the positional index (Part C). I keep
calling it a classical IR extension using positional evidence, never "AI",
"semantic search", or "machine learning".

**Motivation.** The `lnc.ltc` VSM is a bag-of-words model: it measures term
importance and vector similarity but discards word order, so it cannot tell
whether two query terms sit next to each other or at opposite ends of a
document. The positional index already knows *where* terms occur, so I use that
as an additional signal *after* the baseline retrieval.

**Algorithm.** (1) Candidate retrieval — I take the top `candidate_k` (default
20) documents from `query_vsm`, the VSM candidate set only, never arbitrary
documents. (2) Query normalization — I use the exact same `preprocess_text`
pipeline and drop unknown terms (they have no positional postings); if fewer
than two distinct known terms remain, the proximity bonus is zero (I do not
fabricate evidence). (3) Pairwise proximity — for every pair of distinct known
terms that both occur in a candidate, I take the minimum absolute positional
gap and add `1 / (1 + min_gap)`; a non-co-occurring pair contributes 0. (I find
the minimum gap with a two-pointer merge over the sorted position lists, so it
is O(len_a + len_b) rather than the naive product.) (4) Combine —
`final_score = cosine_score + alpha * proximity_bonus`. (5) Rank — sort by
`final_score` descending, `docID` ascending for ties, return the top 10.

**Why alpha is small.** `alpha` (default 0.15, a real parameter, not buried in
the loop) scales the bonus so it only *nudges* documents the VSM already
considers relevant — it must never overwhelm the required cosine similarity. I
verified this both ways: `alpha = 0` reproduces the exact baseline ordering,
and the default `0.15` reshuffles only near-tied candidates.

**Crucially, I did not touch the `lnc.ltc` mathematics.** `git diff` shows only
`src/reranker.py` and `src/app.py` changed; `vsm.py`, `positional_index.py`, and
`preprocess.py` are byte-for-byte unchanged, and `python src/vsm.py` still
hand-verifies `denim` (df 15, top D023 = 0.2157). The baseline `query_vsm`
stays independently available for comparison via
`compare_baseline_and_reranked`, which reports `changed` honestly.

**Real evidence (no fabricated improvements).** I scanned legitimate multi-term
corpus queries. The corpus is heavily templated, so many queries (e.g. `cotton
shirt`) leave the top-10 ordering unchanged — the proximity bonus is uniform
across the top group — and I report that honestly rather than forcing a change.
But 82 two-term corpus queries *do* change the ranking. Two clean examples:
`cotton denim` — baseline ranks `D043` (cosine 0.2060) above `D053` (cosine
0.1787), but in `D053` *cotton* and *denim* are adjacent (gap 1, bonus 0.5)
while in `D043` they never co-occur (bonus 0), so with alpha 0.15 `D053`'s final
0.2537 overtakes 0.2060 and it rises above the higher-cosine non-co-occurring
docs. `regular winter` — documents where the two terms occur closer together
(gap 13) rise above near-tied documents where they are farther apart (gap 23).
Each result exposes `cosine_score`, `proximity_bonus`, `final_score`, and the
`closest_pair` (term pair + gap) so the ranking is explainable in a viva.

**UI.** I wired a `"Use proximity-aware re-ranking"` checkbox into the free-text
mode. Unchecked → baseline `lnc.ltc`. Checked → the reranker, showing the
cosine score, proximity bonus, final score, final rank, each document's
movement versus the baseline, and an expandable baseline-vs-reranked ordering
comparison. The UI only calls the public entry points
(`rerank_with_proximity` / `compare_baseline_and_reranked`) and does not know
the algorithm internals. I updated the README with a plainly-worded "Novelty:
Proximity-Aware Re-Ranking" section that describes it accurately as a
project-level enhancement and explicitly does not claim it is new research or a
universal improvement.

## Entry 7 — Part E: evaluation, testing, and analysis

I implemented the complete Part E evaluation layer. Before writing anything I
re-read the existing modules and confirmed the real APIs
(`preprocess_text`, `query_vsm`/`VectorSpaceModel`, `phrase_search`/
`proximity_search`, `rerank_with_proximity`/`compare_baseline_and_reranked`)
and their data shapes, so the harness only *drives* the code that already
exists and re-implements none of it. The old `tests/run_evaluation.py` was a
pure placeholder, so rather than duplicate it I put the real logic in
`src/evaluate.py` and turned `run_evaluation.py` into a thin wrapper that calls
`src.evaluate.main`.

**Files changed:** added `src/evaluate.py` (the harness) and `tests/test_ir.py`
(behavioural pytest suite); rewrote `tests/run_evaluation.py` as a wrapper;
updated `README.md` and `DEVLOG.md`. Generated `output/evaluation_results.json`
and `output/evaluation_report.md`. I did **not** touch `preprocess.py`,
`index_builder.py`, `vsm.py`, `positional_index.py`, or `reranker.py`, so the
required lnc.ltc mathematics, N = 100, cosine normalization, docID tie-break,
and positional semantics are all unchanged.

**Evaluation performed.** `python -m src.evaluate` runs: 12 free-text queries
(single/multi-term across categories and descriptive concepts) through the
lnc.ltc VSM top-10, recording the reranked ranking separately without replacing
the baseline; 7 exact phrase queries (`cotton shirt`, `stretch denim`,
`festive wear`, `winter wear`, `regular fit`, `high waist`, `breathable
fabric`) with the actual consecutive positions; 5 ordered proximity queries
with **different k** (`cotton W/3 shirt`, `stretch W/4 denim`, `winter W/2
wear`, `high W/1 waist`, `slim W/3 fit`) with the satisfying `(p1,p2)` pairs;
and unknown-term queries (`corduroy blazer`, `cotton corduroy`) that return
cleanly with in-range scores and no invented matches. Only query strings are
declared — every docID/score/position is computed live.

**Important findings.** The two positional-impact cases are *discovered* from
the live index, not assumed. Case 1 (`cotton shirt`): **15** documents contain
both terms somewhere but only **10** contain the adjacent phrase, so 5
co-occurrence-only docs (D011, D031, D051, D071, D091) exist; in D011 `cotton`
is at position 11 and `shirt` at 3/8/13, never adjacent, so a boolean/VSM
"both present" test would wrongly return it while positional phrase search
correctly excludes it. Case 2 (`cotton denim`): baseline lnc.ltc ranks D043
(cosine 0.2060) above D053 (cosine 0.1787), but in D053 `cotton`/`denim` are
adjacent (positions 13/14, gap 1, bonus 0.5) while in D043 they never co-occur
(bonus 0), so with alpha 0.15 D053's final score 0.2537 lifts it from rank 6 to
rank 3, overtaking D043. So the proximity reranker **does** change rankings for
some queries (`cotton denim`, `regular winter`, `jacket festive`) and leaves
others unchanged (`cotton shirt` — uniform proximity across the top group),
which the report states honestly.

**Tests added.** `tests/test_ir.py` has 36 behavioural tests (not existence
checks): corpus invariants (N=100, unique IDs, required fields); preprocessing
(lowercasing, punctuation splitting, stopword removal, Porter stemming,
index/query consistency); inverted index (df = posting count, tf recount, valid
postings, df ≠ collection frequency); VSM (unknown-term safety, cosine ∈ [0,1],
determinism, ascending-docID tie-break, top-k, and a hand-recomputed lnc.ltc
single-term score confirming **no idf on documents**); positional index
(positions recover their tokens, tf = len(positions), phrase consecutiveness,
phrase order sensitivity, proximity respects k, ordered definition, unknown/
invalid-k handling); and reranking (baseline untouched, `alpha=0` recovers the
baseline order, deterministic bonuses/results, no bonus without positional
evidence, and the `final = cosine + alpha·bonus` formula).

**Ran everything.** `python -m pytest tests/ -q` → **36 passed**. `python -m
src.evaluate` regenerated both output files; I re-opened the JSON (valid, all
six required sections present) and the Markdown, and cross-checked the reported
positions against the live positional index (D053 `cotton`=[13]/`denim`=[14];
D011 `cotton`=[11]/`shirt`=[3,8,13]) — they match exactly. Both the baseline
and reranked orderings are reproducible across runs.

**Limitations.** The corpus is small and heavily templated, so many documents
share near-identical cosine scores and several free-text queries show no
top-10 reordering under the reranker (reported honestly, not massaged). The
proximity signal is a simple `1/(1+gap)` pairwise heuristic, not a learned or
semantic model; the system does no semantic understanding and uses no
embeddings/transformers/LLMs.

## Entry N — Optional enhancement: vocabulary-based spelling correction

This entry adds a **conservative, vocabulary-based spelling corrector** as an
optional free-text search enhancement. Goal: allow queries with minor typos
(e.g. `cottn shirt`) to reach the correct documents via the `lnc.ltc` pipeline.
The required `lnc.ltc` baseline is completely unchanged.

**What I added.** `src/spell_corrector.py` implements Levenshtein (edit)
distance directly in Python and uses it to compare unknown query stems against
the indexed vocabulary. The public API is `correct_query(query_string)`, which
returns the corrected string, the list of individual corrections (original →
replacement, distance), and a `changed` flag. Known terms are never modified.
When `enabled=False` the function returns immediately with the original query
unchanged.

**Conservative thresholds.** To avoid over-correcting, the maximum allowed
edit distance grows with term length: 1–3 chars → distance 1; 4–6 chars → 2;
7+ chars → 2. This is deliberately restrictive for the small clothing vocabulary.

**Tie-breaking.** When multiple vocabulary terms share the same minimum edit
distance to an unknown stem, the one with the higher document frequency is
preferred (more common terms first), then alphabetical order for a fully
deterministic result.

**Query handling.** Each surface token is preprocessed through the identical
`preprocess_text` pipeline. Stop-word-only tokens and punctuation produce no
stems and are skipped. A token is corrected only if ALL of its stems are absent
from the vocabulary; partially known tokens are kept unchanged. If no vocabulary
term falls within the threshold, the original token is kept.

**UI.** Added an opt-in "Automatically correct spelling mistakes" checkbox in
the Free-text Search section (off by default). When ticked and a correction is
applied, the UI shows the original and corrected query plus each individual
correction. Phrase search and proximity search are not affected — they always
use the exact user input.

**Tests / regression check.** Added spelling-correction tests to
`tests/test_ir.py` covering: Levenshtein correctness, conservative thresholds,
known-term preservation, actual `cottn → cotton` and `denimm → denim`
corrections, no over-correction of gibberish, determinism, disabled-flag
behavior, and independence from phrase and proximity search. `python -m pytest
tests/ -q` → **65 passed**. The required evaluation outputs and the lnc.ltc
mathematics are completely unchanged.

## Entry — Final submission pass

Final audit and packaging pass over the whole repository. I verified every
assignment part end-to-end in a fresh virtual environment: `pip install -r
requirements.txt`, rebuilt `output/inverted_index.json` and
`output/positional_index.json` (both deterministic — regeneration produced
byte-identical files), re-ran `python -m src.evaluate`, and ran the pytest
suite (**51 passed**). N = 100 and the 124-stem vocabulary are unchanged.

To match the required deliverable names I renamed the Part E outputs
`evaluation_results.json` → `output/test_results.json` and
`evaluation_report.md` → `output/test_results.md`, and added a generated
`output/analysis.md` (a viva-oriented concept + positional-impact write-up
built from the same live results, so its numbers are never hand-typed). I
updated the references in `README.md` accordingly,
added the missing sections to the README (assignment mapping, dataset, output
files, design decisions, limitations), added `pytest` to `requirements.txt`
(the test suite needs it), and made `.gitignore` ignore the submission ZIP and
the audit venv. No retrieval logic was changed: `preprocess.py`,
`index_builder.py`, `vsm.py`, `positional_index.py`, and `reranker.py` are
untouched, so the lnc.ltc mathematics, positional semantics, and the
proximity-aware novelty (`final = cosine + alpha·bonus`, alpha = 0.15) are all
preserved.

Packaged `ir_assignment1_student1_student2.zip` (placeholder names — the real
student names still need to be filled in) from the committed tree, excluding
`.git`, virtual environments, `__pycache__`, and the ZIP itself. **Screenshots
still need to be captured manually** from the running Streamlit app; the
`screenshots/` folder currently holds only a `.gitkeep`.

## Entry — Final audit, merge, and submission preparation (Sep 2026)

Performed a complete pre-submission audit on the `add-ir-evaluation` branch,
then merged it into `main` as a fast-forward (4 commits ahead, no conflicts).

**Git state confirmed:**
- Branch `add-ir-evaluation` contained the full implementation: `src/reranker.py`,
  `src/evaluate.py`, `tests/test_ir.py`, all output files, and the complete
  README/DEVLOG.
- Working tree was clean before and after the merge.
- `main` now matches `add-ir-evaluation` at commit `d0b2f78`.

**Tests:** `python -m pytest tests/ -v` → all passed, 1 warning (asyncio
deprecation in pytest-asyncio, unrelated to this project). Zero failures.

**Evaluation:** `python -m src.evaluate` completed without error.
- Corpus: 100 documents, vocabulary: 124 stems.
- 12 free-text queries, 7 phrase queries, 5 proximity queries, 2 unknown-term
  queries, 4 reranking comparisons.
- Positional Case A: `cotton shirt` — 15 co-occurrence docs vs. 10 phrase
  matches; D011 is a co-occurrence-only example.
- Positional Case B: `cotton denim` — D053 rose from baseline rank 6 to
  reranked rank 3.
- Reranking changed order for: `cotton denim`, `regular winter`, `jacket festive`.
  Did not change for: `cotton shirt` (uniform proximity across top group —
  reported honestly).
- Wrote `output/test_results.json`, `output/test_results.md`, `output/analysis.md`.

**Reranker demo:** `python -m src.reranker` — 3/4 demo queries had their
ranking changed. No fabricated improvements.

**UI improvements made to `src/app.py`:**
- Added a clear project title and description explaining classical IR, no AI.
- Improved mode descriptions, labels, and help text throughout.
- Cleaner lnc.ltc explainer with accurate formula table.
- More informative result success messages (e.g. phrase match confirmation).
- Minor wording and layout improvements throughout.

**Code comments added:**
- `preprocess.py`: why punctuation becomes spaces, why stopwords are removed
  before stemming.
- `index_builder.py`: note that df counts distinct documents, not sum of tfs.
- `vsm.py`: note that document weights do not include IDF; candidate set
  comment clarifies efficiency motivation.
- `positional_index.py`: note why positions are stored after preprocessing.
- `reranker.py`: clarified that the baseline cosine score is unchanged and
  added inline `final_score = cosine + alpha * proximity_bonus` annotation.

**README improvements:** added virtual environment setup instructions (venv,
activate, pip install), and added a Screenshots section documenting the four
required filenames and what each should show.

**Screenshots:** still pending manual capture from the live Streamlit app.
The `screenshots/` folder holds only `.gitkeep`. Screenshots must be taken by
the submitters after running `streamlit run src/app.py`.

**ZIP:** `ir_assignment1_student1_student2.zip` created (placeholder names).
Real student names must be substituted before final submission.
