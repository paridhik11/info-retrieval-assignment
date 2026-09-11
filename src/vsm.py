"""
vsm.py
======

Part B of the assignment: ranked retrieval with the Vector Space Model (VSM)
using the lnc.ltc weighting scheme.

Responsibilities (to be implemented in a later step):
    * Document weights (lnc):
          w(d,t) = 1 + log10(tf)   for tf > 0   (no idf on documents)
    * Query weights (ltc):
          w(q,t) = (1 + log10(tf)) * log10(N / df),   N = 100
    * Cosine-normalize both document and query vectors.
    * Compute cosine similarity between the query and each matching document.
    * Return up to 10 docIDs sorted by decreasing similarity, breaking ties by
      increasing docID.

Design notes (kept explicit for the viva):
    * lnc.ltc is implemented from first principles (log-tf + idf + cosine
      normalization). It is NOT delegated to TF-IDF/BM25/embedding libraries.
    * Document length normalization values are pre-computed from the inverted
      index so queries stay fast.

NOTE: IR logic is intentionally NOT implemented yet in this commit.
"""

# Implementation intentionally deferred.
