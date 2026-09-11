"""
preprocess.py
=============

Part A: corpus reading and the single text-normalization pipeline.

This module is the source of truth for how a string becomes a list of index
terms. The inverted index, Vector Space Model, positional index, and query
processor must all call ``preprocess_text`` (or ``tokenize`` + the same
stopword/stemming steps) so document terms and query terms stay comparable.

Stopword policy
---------------
We use the **standard NLTK English stopword list** (``nltk.corpus.stopwords``
words for ``'english'``) with **no domain-specific additions or deletions**.

Why a stock list, not a clothing-specific one:
    Clothing terms such as *wear*, *fit*, *size*, *cotton*, *denim*, *shirt*,
    *kurta*, *dress*, *winter*, and *festive* are meaningful retrieval cues
    in this collection (fabric, garment type, season, occasion). They are
    **not** English function words. None of them appear in the NLTK English
    list, so they remain searchable after filtering. We deliberately do
    **not** drop them just because they are frequent in a clothing corpus.

Consequence we accept rather than special-case:
    Single-letter size tokens ``s`` and ``m`` collide with NLTK entries that
    exist for contracted English (``it's`` -> ``s``, ``I'm`` -> ``m``). The
    word *size* itself is kept. We keep the stock list consistent rather
    than carving out clothing sizes.

What is indexed
---------------
Only TITLE + TEXT. CATEGORY is stored as display metadata and is **not**
added to the token stream: the title already names the garment type, and
indexing CATEGORY would inflate tf uniformly for every document in a
category, which is not a signal we want baked into the index.
"""

from __future__ import annotations

import re
import string
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Punctuation is replaced by spaces (not deleted) so hyphenated product
# language ("t-shirt", "easy-care") splits into parts instead of becoming
# an opaque glued token ("tshirt"). Tokenization then splits on whitespace.
_PUNCT_RE = re.compile(rf"[{re.escape(string.punctuation)}]+")

_STEMMER: PorterStemmer | None = None
_STOPWORDS: set[str] | None = None


@dataclass
class Document:
    """One <DOC> record with original metadata preserved for later display."""

    doc_id: str
    category: str
    title: str
    text: str
    # Filled by process_documents(); empty until then.
    tokens: list[str] = field(default_factory=list)


def ensure_nltk_stopwords() -> None:
    """Load the NLTK English stopword list, downloading it if needed."""
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)


def get_stemmer() -> PorterStemmer:
    """Return the shared Porter stemmer (one instance is enough and deterministic)."""
    global _STEMMER
    if _STEMMER is None:
        _STEMMER = PorterStemmer()
    return _STEMMER


def get_stopwords() -> set[str]:
    """Return the cached NLTK English stopword set (lowercase, as NLTK stores it)."""
    global _STOPWORDS
    if _STOPWORDS is None:
        ensure_nltk_stopwords()
        _STOPWORDS = set(stopwords.words("english"))
    return _STOPWORDS


def load_corpus(path: str | Path) -> str:
    """Read the raw corpus file as UTF-8 text."""
    return Path(path).read_text(encoding="utf-8")


def _tag_text(element: ET.Element, tag: str) -> str:
    """Extract stripped text from a child tag; missing/empty tags become ''."""
    child = element.find(tag)
    if child is None or child.text is None:
        return ""
    return child.text.strip()


def parse_documents(raw_text: str) -> list[Document]:
    """
    Parse every <DOC> using ElementTree (real XML tags, not line offsets).

    The supplied file is a *sequence* of <DOC> elements, not a single rooted
    XML document, so we wrap it in a temporary <CORPUS> root. Document IDs
    and order come from the tags themselves; we never assume D001..D100.
    """
    stripped = raw_text.strip()
    wrapped = f"<CORPUS>{stripped}</CORPUS>"
    try:
        root = ET.fromstring(wrapped)
    except ET.ParseError as exc:
        raise ValueError(f"corpus is not well-formed XML: {exc}") from exc

    documents: list[Document] = []
    for doc_el in root.findall("DOC"):
        documents.append(
            Document(
                doc_id=_tag_text(doc_el, "DOCID"),
                category=_tag_text(doc_el, "CATEGORY"),
                title=_tag_text(doc_el, "TITLE"),
                text=_tag_text(doc_el, "TEXT"),
            )
        )
    return documents


def tokenize(text: str) -> list[str]:
    """
    Lowercase, strip punctuation, then split on whitespace.

    Stopword removal and stemming are *not* done here so callers that only
    need raw tokens (or that want to inspect the pre-stopword stream) share
    the same first stage as ``preprocess_text``.
    """
    normalized = _PUNCT_RE.sub(" ", text.lower())
    return [token for token in normalized.split() if token]


def preprocess_text(text: str) -> list[str]:
    """
    Full index/query pipeline: tokenize -> drop English stopwords -> Porter stem.

    Stemming runs *after* stopword removal, as specified: the stopword check
    is against surface forms ("this", "from"), not stems.
    """
    english_stopwords = get_stopwords()
    stemmer = get_stemmer()
    tokens: list[str] = []
    for token in tokenize(text):
        if token in english_stopwords:
            continue
        tokens.append(stemmer.stem(token))
    return tokens


def indexable_content(document: Document) -> str:
    """TITLE + TEXT only. CATEGORY is metadata, not indexable content."""
    return f"{document.title} {document.text}".strip()


def process_documents(documents: list[Document]) -> list[Document]:
    """Attach the preprocessed token list to every parsed document (in place)."""
    for document in documents:
        document.tokens = preprocess_text(indexable_content(document))
    return documents


def index_lookup_term(raw_term: str) -> str | None:
    """
    Map a surface term (e.g. 'cotton') to its index key.

    Needed because the vocabulary stores Porter stems, so a raw lookup of
    'festive' would miss 'festiv'. Returns None if the term is empty or is
    only stopwords/punctuation after normalization.
    """
    tokens = preprocess_text(raw_term)
    if not tokens:
        return None
    # Example lookups are single content words; if punctuation splits a
    # term we still return the first surviving stem so callers can probe.
    return tokens[0]
