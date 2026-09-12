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
  
                       
