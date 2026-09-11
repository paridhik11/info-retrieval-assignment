"""
index_builder.py
================

Part A (dictionary / inverted index) of the assignment.

Responsibilities (to be implemented in a later step):
    * Build a standard inverted index from the pre-processed documents.
    * For every term store at least:
          - document frequency (df)
          - a postings list of (docID, term frequency) entries
    * Serialize the inverted index to ``output/inverted_index.json`` so it can
      be inspected and submitted as a deliverable.

Design notes (kept explicit for the viva):
    * This module produces the NON-positional inverted index used by the
      Vector Space Model. The richer positional index lives in
      ``positional_index.py`` so the two representations can be compared.
    * Term/document statistics (df, tf, N) are the inputs to lnc.ltc weighting.

NOTE: IR logic is intentionally NOT implemented yet in this commit.
"""

# Implementation intentionally deferred.
