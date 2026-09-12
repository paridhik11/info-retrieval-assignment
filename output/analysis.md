# IR Assignment 1 — Analysis & Concept Reference

This document explains *why* the system is built the way it is and answers the concepts most likely to come up in a viva. The numeric examples are read from the same live evaluation run that produced `test_results.json` / `test_results.md`, so they always match the current corpus.

## 1. What the system is

A classical Information Retrieval pipeline over 100 clothing product descriptions (124 distinct stems, N = 100 for idf):

- **Preprocessing** — one shared pipeline: lowercase → punctuation split → NLTK English stop-word removal → Porter stemming.
- **Inverted index** — `term → {df, postings: {docID: tf}}`.
- **Ranked retrieval** — Vector Space Model with the **lnc.ltc** weighting scheme and cosine similarity.
- **Positional index** — postings extended with token positions, supporting exact phrase search and ordered `WITHIN/k` proximity search.
- **Novelty** — proximity-aware re-ranking: an explainable extension that nudges the lnc.ltc ranking using positional evidence, without replacing the baseline.

There is **no** machine learning, no embeddings, and no LLM anywhere in the system — it is classical IR arithmetic that can be hand-checked.

## 2. Concept reference (viva questions)

**What is an inverted index?** A map from each vocabulary term to the list of documents that contain it (its *postings*). Instead of scanning every document for a query term, we jump straight to that term's postings. Here each posting also stores the term frequency (and, in the positional index, the positions).

**What is `df` (document frequency)?** The number of *distinct documents* a term occurs in — the length of its postings list. It is **not** the total number of occurrences (that is collection frequency). `df` drives idf: a term in few documents is more discriminating.

**What is `tf` (term frequency)?** The number of times a term occurs *within one document*. In lnc.ltc it is dampened with a logarithm (`1 + log10(tf)`) so a word occurring 10 times is worth more than once, but not ten times as much.

**Why is idf applied to the query but not the document in lnc.ltc?** idf (`log10(N/df)`) is a *collection-level* property of a term, not a property of the term inside one document. lnc.ltc applies it exactly **once**, on the query side, so a rare word still boosts the ranking without being squared by also multiplying it into every document weight. This also keeps document weights independent of any query, so they can be precomputed once. (The middle letter of the document scheme `lnc` is `n` = *no* idf.)

**Why normalize the vectors?** Without normalization, long documents accumulate larger raw dot products purely because they have more terms and higher tf — length, not relevance, would win. Dividing each vector by its Euclidean (cosine) norm projects every document and the query onto the unit sphere, so similarity depends on term *proportions*, not document size.

**Why cosine similarity?** After both vectors are length-normalized, their dot product is the cosine of the angle between them — a bounded `[0, 1]` measure of how similarly the query and document distribute weight across shared terms. Because the vectors are already unit length, the cosine *is* the dot product; we do not divide by the norms a second time.

**What does positional indexing add?** The plain VSM is a *bag-of-words* model: it knows *which* terms a document contains and how salient they are, but it discards word **order**. The positional index additionally stores *where* each term occurs, which lets us answer structural questions the VSM cannot — is this an exact phrase? are these two terms close together?

**Why is `cotton shirt` different from merely finding documents that contain both words?** Because a phrase requires the words to be **adjacent and in order**, not just present somewhere. In this corpus, `cotton shirt` occurs in **15** documents when we only ask that both terms appear *somewhere*, but in only **10** documents as the actual adjacent phrase. For example, **D011** contains both terms but never next to each other, so a boolean/VSM 'both present' test would wrongly return it while exact phrase search correctly excludes it.

**What does `k` mean in proximity search?** `k` is the **maximum positional difference** allowed between the two matched tokens — *not* the number of words in between. Ordered `term1 WITHIN/k term2` keeps pairs with `0 < p2 - p1 <= k`; adjacency is the `k = 1` case, and `k = 3` allows up to two tokens between them.

**How does the novelty combine VSM and positional information?** It runs the lnc.ltc VSM first and takes its top candidates. For each candidate it computes a `proximity_bonus` from the positional index: for every pair of distinct known query terms, `pair_bonus = 1 / (1 + min_gap)` (closer terms → larger bonus), summed over pairs. The candidates are then re-ordered by `final_score = cosine_score + alpha * proximity_bonus`. So the lexical score (VSM) decides *relevance* and the positional signal only *reshapes* the ordering.

**Why is `alpha` necessary?** `alpha` (default 0.15) scales the proximity bonus so it stays a *controlled secondary signal*. Cosine scores and proximity bonuses live on different scales; without a small weight the heuristic bonus could overwhelm the required cosine similarity. Keeping `alpha` small means proximity only nudges near-tied candidates, and setting `alpha = 0` recovers the exact baseline order — which is how we prove the baseline is preserved.

**Why did the novelty not replace lnc.ltc?** The assignment requires lnc.ltc ranked retrieval, and proximity is a weaker, heuristic signal that is only meaningful *among* already-relevant documents. Replacing the VSM would throw away the term-weighting that decides relevance in the first place. Instead the reranker *calls* `query_vsm` unchanged and layers on top of it, so the baseline stays independently available for comparison.

**What happens when a query contains an unknown term?** An unknown term has no `df` and no postings, so it contributes **no** query weight, cannot form a phrase, and cannot form a proximity pair. Retrieval continues on the remaining known terms (or returns an empty list if every term is unknown) and never crashes. A known + unknown query returns exactly what the known term alone would.

## 3. Positional impact — two concrete cases

These are the two required cases where positional information changes the result set or ordering. Both are **discovered** from the live index, not hard-coded.

### Case 1 — exact phrase vs. mere co-occurrence

- **Phrase:** `cotton shirt` (normalized ['cotton', 'shirt']).
- Documents containing all terms *somewhere*: **15**.
- Documents where the phrase actually matches (adjacent, in order): **10**.
- Co-occurrence-only example **D011** (T-Shirt: Men's Oversized Graphic T-Shirt - Mustard):
  - `cotton` at positions 11
  - `shirt` at positions 3, 8, 13
  - Closest forward gap between the terms: **2** (> 1, so not a phrase).

This is the textbook reason positional indexing matters: co-occurrence and phrase matching are **not** the same set.

### Case 2 — proximity-aware re-ranking changes the order

- **Query:** `cotton denim`.
- Baseline order: ['D023', 'D083', 'D043', 'D003', 'D063', 'D053', 'D058', 'D018', 'D078', 'D038']
- Reranked order: ['D023', 'D083', 'D053', 'D043', 'D003', 'D063', 'D058', 'D018', 'D078', 'D038']
- **D053** (Jeans: Men's Slim Fit Stretch Jeans - Grey) rises from baseline rank **6** to reranked rank **3**: cosine 0.1787, proximity bonus 0.5000, final 0.2537.
  - Closest term pair `cotton`/`denim` with minimum gap **1** (pair bonus 0.5000).
- It overtakes **D043** (cosine 0.2060, *higher*, but proximity bonus 0.0000) — the terms are not close together there.

For query 'cotton denim', D053 has cosine 0.1787 with the query terms close together (cotton/denim gap 1), earning proximity bonus 0.5000. With alpha=0.15 its final score 0.2537 lifts it from baseline rank 6 to reranked rank 3, overtaking D043 (higher cosine 0.2060 but proximity bonus 0.0000).

## 4. Baseline vs. proximity re-ranking — honest observations

- `cotton denim`: re-ranking **changed** the order (documents that moved: ['D053', 'D043', 'D003', 'D063']).
- `regular winter`: re-ranking **changed** the order (documents that moved: ['D040', 'D100', 'D060', 'D028', 'D088']).
- `jacket festive`: re-ranking **changed** the order (documents that moved: ['D048', 'D038']).
- `cotton shirt`: re-ranking **did not change** the order (the proximity bonus was uniform across the top group — reported honestly, not massaged).

The reranker helps for some multi-term queries and does nothing for others. We never fabricate an improvement; a no-change result is reported as such.

## 5. Limitations

- The corpus is small (100 documents) and heavily templated, so many documents share near-identical cosine scores; this both limits how often the proximity signal can matter and makes several queries tie.
- The proximity bonus is a simple `1/(1+gap)` pairwise heuristic, not a learned or semantic model. It rewards closeness, which is a proxy for — not a guarantee of — better relevance.
- Porter stemming is aggressive (e.g. `festive` → `festiv`, `leggings` → `legg`), which is standard but can merge or mangle a few product words.
- Single-letter size tokens `s`/`m` collide with NLTK stop-words (from contractions) and are dropped; this is an accepted consequence of using a consistent stock stop-word list.
- The system performs **no** semantic understanding: synonyms and paraphrases are not matched. It is a classical lexical + positional IR system by design.

