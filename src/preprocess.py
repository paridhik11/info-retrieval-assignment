"""
preprocess.py
=============

Part A of the assignment: corpus reading and the text pre-processing pipeline.

Responsibilities (to be implemented in a later step):
    * Parse ``data/corpus_100.txt`` into structured documents.
      Each document in the corpus uses the tag layout:
          <DOC>
            <DOCID>D001</DOCID>
            <CATEGORY>...</CATEGORY>
            <TITLE>...</TITLE>
            <TEXT>...</TEXT>
          </DOC>
    * Tokenize text and normalize case (lower-casing).
    * Remove punctuation.
    * Apply Porter stemming (via ``nltk.stem.PorterStemmer``).
    * Apply a documented, consistent stop-word policy using the standard
      NLTK English stop-word list.

Design notes (kept explicit for the viva):
    * The SAME pipeline must be reused everywhere (index building, VSM,
      positional index, and query parsing) so that document terms and query
      terms are always comparable.
    * Token positions produced here are what the positional index relies on,
      so tokenization order must be deterministic.

NOTE: IR logic is intentionally NOT implemented yet in this commit.
"""

# Implementation intentionally deferred. This module currently only documents
# the responsibilities of the pre-processing stage.
