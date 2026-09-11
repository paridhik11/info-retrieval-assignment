"""
positional_index.py
====================

Part C of the assignment: positional index plus phrase and proximity search.

Responsibilities (to be implemented in a later step):
    * Extend the inverted index so every posting stores term positions:
          term -> df -> [(docID, tf, [p1, p2, ...]), ...]
    * Reuse the exact tokenization/normalization/stemming pipeline from
      ``preprocess.py`` so positions line up with document terms.
    * Serialize to ``output/positional_index.json``.
    * Exact phrase search (e.g. "cotton shirt", "stretch denim") using
      consecutive positions rather than mere co-occurrence.
    * Ordered proximity search (e.g. ``cotton WITHIN/3 shirt``) where the
      second term must appear within k positions AFTER the first term.

Design notes (kept explicit for the viva):
    * Phrase search checks that positions differ by exactly 1 (adjacency).
    * Proximity search enforces order and a maximum gap k.
    * This is what lets us compare plain VSM retrieval against positional
      retrieval (Part C / Part E discussion).

NOTE: IR logic is intentionally NOT implemented yet in this commit.
"""

# Implementation intentionally deferred.
