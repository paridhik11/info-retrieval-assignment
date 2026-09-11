"""
app.py
======

Part D of the assignment: the Streamlit search interface.

Responsibilities (to be implemented in a later step):
    * Free-text query mode: show the top 10 ranked results with docID,
      title/category, and cosine score (from ``vsm.py``).
    * A second mode for phrase / proximity queries backed by the positional
      index (from ``positional_index.py``).
    * Display matching term positions for at least one positional query as
      evidence that the positional index is being used.

Design notes (kept explicit for the viva):
    * The UI is a thin layer over the IR modules; all retrieval logic lives in
      the ``src`` modules, not in the UI code.

Run (once implemented):
    streamlit run src/app.py

NOTE: UI logic is intentionally NOT implemented yet in this commit.
"""

# Implementation intentionally deferred.
