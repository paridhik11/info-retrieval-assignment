# ClothingSpree IR — Complete Local Setup & Testing Guide

This guide covers everything needed to reproduce the project from scratch:
clone the repo, set up the environment, run all automated tests, run the
evaluation harness, launch the Streamlit app, take the required screenshots,
and prepare the final submission ZIP.

**GitHub repo:** https://github.com/paridhik11/info-retrieval-assignment.git

---

## Part E — Requirements checklist (already satisfied by the code)

Before setting up, confirm what Part E requires and where each requirement is met:

| Requirement | Minimum | Covered in |
|---|---|---|
| Free-text queries | ≥ 10 | `evaluate.py` `FREE_TEXT_QUERIES` — **12 queries** |
| Exact phrase queries | ≥ 5 | `evaluate.py` `PHRASE_QUERIES` — **7 queries** |
| Proximity queries, different k | ≥ 3 | `evaluate.py` `PROXIMITY_QUERIES` — **5 queries, k = 1/2/3/4** |
| Query with absent term | ≥ 1 | `evaluate.py` `UNKNOWN_TERM_QUERIES` — **2 queries** |
| Top-10 results | yes | `evaluate.py` reports top 10 per query in `output/test_results.md` |
| Positional Case 1 (phrase vs co-occurrence) | yes | discovered live from the positional index |
| Positional Case 2 (proximity reranking) | yes | discovered live from the reranker |
| No hard-coded doc IDs | yes | only query *strings* are declared; all results computed at runtime |

All requirements are satisfied. Running `python -m src.evaluate` regenerates
the evidence automatically.

---

## 1. Prerequisites

- **Python 3.9 or higher** (3.10 / 3.11 / 3.12 / 3.14 all work)
- **Git**
- A terminal: PowerShell on Windows, Terminal on macOS/Linux

Check your Python version:

```powershell
python --version
```

---

## 2. Clone the repository

```powershell
git clone https://github.com/paridhik11/info-retrieval-assignment.git
cd info-retrieval-assignment
git branch --show-current      # should print: main
git log --oneline -5           # confirm recent commits
```

---

## 3. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If you get an execution-policy error on Windows, run this first (once):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Your prompt should now show `(.venv)` at the left.

---

## 4. Install dependencies

```powershell
pip install -r requirements.txt
```

This installs: `nltk`, `streamlit`, `pytest`, and their dependencies.
It also downloads the NLTK English stopword list automatically on first use.

If the automatic NLTK download is blocked in your environment, run this once:

```powershell
python -c "import nltk; nltk.download('stopwords')"
```

---

## 5. Build the indexes

```powershell
python src/index_builder.py
python src/positional_index.py
```

Expected output from `index_builder.py`:

```
documents parsed N = 100
unique DOCIDs     = 100
vocabulary size   = 124
indexed terms     = 124
wrote output/inverted_index.json
wrote output/doc_metadata.json
```

Expected output from `positional_index.py`:

```
documents indexed  : 100
vocabulary size    : 124
wrote              : output/positional_index.json
position convention: ZERO-BASED index into processed token stream
pipeline check     : positional index matches output/inverted_index.json (same vocabulary, df, and tf)
```

Confirm the files now exist:

```powershell
dir output\
```

You should see: `inverted_index.json`, `doc_metadata.json`, `positional_index.json`.

---

## 6. Run the automated test suite

```powershell
python -m pytest tests/ -q
```

**Expected result:** `65 passed` — no failures, no errors.

For verbose output (shows each test name):

```powershell
python -m pytest tests/ -v
```

The 65 tests cover:
- Corpus: N = 100, unique IDs, required fields
- Preprocessing: lowercasing, punctuation, stopwords, stemming, pipeline consistency
- Inverted index: df/tf correctness, valid postings, df ≠ collection frequency
- VSM (lnc.ltc): unknown-term safety, cosine range [0,1], determinism, docID tie-break, hand-recomputed formula
- Positional index: positions recover tokens, tf = len(positions), phrase consecutiveness, phrase order sensitivity, proximity respects k
- Reranker: baseline untouched, alpha=0 recovers baseline, determinism, no bonus without evidence, formula check
- Spelling corrector: Levenshtein correctness, conservative thresholds, known-term preservation, cottn→cotton, determinism, disabled-flag behavior

---

## 7. Run the Part E evaluation harness

```powershell
python -m src.evaluate
```

**Expected console output:**

```
Running IR evaluation suite (Part E)...
  corpus size          : 100
  vocabulary size      : 124
  free-text queries    : 12
  phrase queries       : 7
  proximity queries    : 5
  unknown-term queries : 2
  rerank comparisons   : 4
  positional Case A    : phrase 'cotton shirt' — 15 co-occur vs 10 phrase matches (example D011)
  positional Case B    : query 'cotton denim' — D053 rose 6->3
  rerank changed order : ['cotton denim', 'regular winter', 'jacket festive']
  wrote output/test_results.json
  wrote output/test_results.md
  wrote output/analysis.md
```

Three files are regenerated under `output/`:

| File | Contents |
|---|---|
| `test_results.json` | Machine-readable results for every query |
| `test_results.md` | Human-readable tables (free-text, phrase, proximity, reranking) |
| `analysis.md` | Viva-oriented concept explanations and positional-impact cases |

Open `output/test_results.md` in any Markdown viewer and confirm:
- Section 2 has a full table with 12 free-text queries and top-10 results
- Section 3 has exact phrase match results with consecutive positions
- Section 4 has proximity results with satisfying (p1, p2) pairs
- Section 5 shows unknown-term handling (no crash, in-range scores)
- Sections 6 and 7 show the two positional-impact cases

---

## 8. Launch the Streamlit app

```powershell
streamlit run src/app.py
```

If `streamlit` is not recognised as a command:

```powershell
python -m streamlit run src/app.py
```

The app opens at **http://localhost:8501** in your browser.

If port 8501 is in use on your machine:

```powershell
streamlit run src/app.py --server.port 8502
```

---

## 9. Screenshot checklist

You need **4 screenshots** minimum. Take them from the running app.
Save them in the `screenshots/` folder using the exact filenames below.

### Screenshot 1 — `01_free_text_baseline.png`

**What to capture:** Free-text search results table (lnc.ltc baseline, no re-ranking).

Steps:
1. Select **Free-text search** mode (radio at top)
2. Make sure **"Use proximity-aware ranking"** is **unchecked**
3. Make sure **"Automatically correct spelling"** is **unchecked**
4. Type: `cotton shirt`
5. Click **Search**
6. Wait for the results table to appear
7. Screenshot the whole page (title, query box, options, and results table)

What you should see: a table with columns Rank, Doc ID, Title, Category, Cosine score; top result should have a cosine score around 0.3–0.5; "Top 10 result(s) — lnc.ltc baseline ranking." success message.

---

### Screenshot 2 — `02_proximity_reranking.png`

**What to capture:** The same query with proximity-aware re-ranking enabled, showing the extra columns.

Steps:
1. Stay in **Free-text search** mode
2. Tick **"Use proximity-aware ranking"**
3. Type: `cotton denim`
4. Click **Search**
5. Screenshot the results table

What you should see: a wider table with columns Rank, Doc ID, Title, Category, Cosine score, Proximity bonus, Final score, Closest pair, Δ vs baseline. The success/info message should say whether the ranking changed. Open the **"Compare baseline vs. re-ranked order"** expander and include it in the screenshot if possible.

---

### Screenshot 3 — `03_exact_phrase_positions.png`

**What to capture:** Exact phrase search showing matching positions.

Steps:
1. Select **Phrase / proximity search** mode (radio at top)
2. The sub-selector should default to **Exact phrase search**
3. Type: `cotton shirt`
4. Click **Search phrase**
5. Screenshot the results table

What you should see: a table with columns Doc ID, Title, Category, Matching positions. Position groups should look like `[14, 15]` — two consecutive numbers. The success message should say "Found the exact phrase in N document(s). Each position group is [start, start+1, ...] — strictly consecutive."

---

### Screenshot 4 — `04_proximity_positions.png`

**What to capture:** Proximity (WITHIN/k) search showing satisfying position pairs.

Steps:
1. Stay in **Phrase / proximity search** mode
2. Select **Proximity search (WITHIN/k)** in the sub-selector
3. Term 1: `cotton`
4. Term 2: `shirt`
5. k: `3`
6. Order: **Ordered (term 1 before term 2)**
7. Click **Search proximity**
8. Screenshot the results table

What you should see: a table with columns Doc ID, Title, Category, Satisfying position pairs. Pairs look like `(14, 15)` — the second number minus the first must be between 1 and 3. The success message should say "Found N document(s) where the terms are within 3 position(s) of each other (ordered)."

---

### Optional extra screenshots (recommended)

These are not mandatory but improve the submission:

| Filename | Query to use |
|---|---|
| `05_spell_correction.png` | Free-text `cottn shirt`, spell correction ticked — shows correction notice |
| `06_how_it_works.png` | Open the "How does this search work?" expander at the bottom |
| `07_unknown_term.png` | Free-text `corduroy blazer` — shows "No matching documents found" info |
| `08_proximity_high_k.png` | Proximity `winter` / `wear` / k=3 ordered |

---

## 10. Manual verification queries

Run these in the app to confirm everything works end-to-end:

### Free-text search (baseline)

| Query | What to check |
|---|---|
| `cotton shirt` | Top 10 results, all clothing items, cosine scores between 0 and 1 |
| `denim jeans` | Jeans-category results at the top |
| `winter jacket` | Jacket-category results |
| `kurta` | Kurta results |
| `corduroy blazer` | Should return "No matching documents" (absent terms) |
| `cottn shirt` | Without spell correction: may return no/wrong results; with correction ticked: returns cotton shirt results |

### Phrase search

| Query | Expected behavior |
|---|---|
| `cotton shirt` | Returns only docs where "cotton" sits immediately before "shirt" |
| `stretch denim` | Returns docs with these two stems adjacent |
| `regular fit` | Returns docs with these two adjacent |
| `winter wear` | Returns matches |
| `festive wear` | Returns matches |

### Proximity search

| Term 1 | Term 2 | k | Expected |
|---|---|---|---|
| `cotton` | `shirt` | 1 | Only adjacent pairs (gap exactly 1) |
| `cotton` | `shirt` | 3 | More results; all pairs have gap ≤ 3 |
| `winter` | `wear` | 2 | Pairs with gap ≤ 2 |
| `stretch` | `denim` | 4 | Pairs with gap ≤ 4 |

---

## 11. Confirm the app looks correct visually

Open http://localhost:8501 and check:

- [ ] White and blue default Streamlit theme (no purple)
- [ ] Title shows "Clothing Product Search Engine"
- [ ] Page title in browser tab shows "ClothingSpree Search"
- [ ] Two mode options: "Free-text search" and "Phrase / proximity search"
- [ ] Free-text mode has two checkboxes (proximity-aware ranking, spell correction)
- [ ] No BM25 controls anywhere in the UI
- [ ] "How does this search work?" expander at the bottom opens correctly and shows the lnc.ltc formula table
- [ ] Results table has no raw Python dicts — only clean text and numbers
- [ ] Empty query shows a warning (not a crash)
- [ ] Unknown term query shows an info message (not a crash)

---

## 12. Final submission steps

### Fill in student names

Edit `README.md` — replace the Authors placeholder:

```markdown
## Authors

Student 1 — [student1_id]
Student 2 — [student2_id]
```

with the real names and student IDs.

### Confirm all output files exist

```powershell
dir output\
```

Expected files:
- `inverted_index.json`
- `doc_metadata.json`
- `positional_index.json`
- `test_results.json`
- `test_results.md`
- `analysis.md`

### Confirm screenshots exist

```powershell
dir screenshots\
```

Expected files at minimum:
- `01_free_text_baseline.png`
- `02_proximity_reranking.png`
- `03_exact_phrase_positions.png`
- `04_proximity_positions.png`

### Commit and push screenshots and author update

```powershell
git add README.md screenshots/
git commit -m "Add screenshots and update authors"
git push origin main
```

### Build the submission ZIP

From the repo root, run the appropriate command for your OS.

**Windows PowerShell:**

```powershell
# Replace student1 and student2 with the real names
$zipName = "ir_assignment1_student1_student2.zip"
Compress-Archive -Path data, src, tests, output, screenshots, README.md, DEVLOG.md, requirements.txt -DestinationPath $zipName -Force
```

**macOS / Linux:**

```bash
zip -r ir_assignment1_student1_student2.zip \
  data/ src/ tests/ output/ screenshots/ \
  README.md DEVLOG.md requirements.txt \
  --exclude "*.pyc" --exclude "__pycache__/*" --exclude ".venv/*"
```

Check the ZIP:

```powershell
# Windows
Expand-Archive -Path ir_assignment1_student1_student2.zip -DestinationPath zip_check -Force
dir zip_check\
Remove-Item -Recurse -Force zip_check
```

Make sure the ZIP does **not** include `.venv/`, `.venv_audit/`, `.git/`, or `__pycache__/`.

---

## 13. Clean-clone smoke test (optional but recommended)

To be sure the ZIP is self-contained, do a fresh test in a temporary folder:

```powershell
mkdir C:\Temp\ir_test
cd C:\Temp\ir_test
# Extract the ZIP here, then:
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/index_builder.py
python src/positional_index.py
python -m pytest tests/ -q
python -m src.evaluate
python -m streamlit run src/app.py
```

If all of the above work, the submission is complete.

---

## Summary of required CLI commands

```powershell
# Once after cloning
python -m venv .venv
.venv\Scripts\Activate.ps1              # Windows
# source .venv/bin/activate             # macOS/Linux
pip install -r requirements.txt

# Build indexes (run once, or whenever corpus changes)
python src/index_builder.py
python src/positional_index.py

# Tests (Part E automated)
python -m pytest tests/ -q

# Evaluation harness (regenerates output/*.md and output/*.json)
python -m src.evaluate

# Streamlit app
streamlit run src/app.py
# or:
python -m streamlit run src/app.py
```
