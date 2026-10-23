"""Hybrid retrieval: semantic embeddings + BM25 keyword search.

* Semantic search (sentence-transformers) understands meaning:
  "get my money back" matches "Approved refunds return to the original payment method".
* BM25 keyword search is exact: it never misses "2FA", "UPI" or an e-mail address.
* Scores are combined:  final = alpha * semantic + (1 - alpha) * bm25_normalised.

If sentence-transformers is not installed or the model cannot be downloaded,
a TF-IDF embedder is used automatically, so the project always runs.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from . import config
from .loader import Sentence

log = logging.getLogger(__name__)

TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = set("""a an the is are was were be been to of and or in on at for with by from as it its this that
these those i you your we our my me do does did can could how what when where which who why will would should
if then there their them they have has had not so any about into than also just please tell more get want know
need""".split())


def tokenize(text: str) -> list[str]:
    out = []
    for word in TOKEN.findall(text.lower()):
        if word in STOPWORDS:
            continue
        for ending in ("ing", "ed", "es", "s"):
            if len(word) > 4 and word.endswith(ending):
                word = word[: -len(ending)]
                break
        out.append(word)
    return out


# --------------------------------------------------------------------------- embedders

class SentenceTransformerEmbedder:
    name = "sentence-transformers"

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer  # imported lazily

        self.model = SentenceTransformer(model_name)

    def fit(self, texts: list[str]) -> None:  # nothing to train
        pass

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False))


class TfidfEmbedder:
    """Fallback that needs only numpy. Same interface as the semantic embedder."""

    name = "tfidf"

    def fit(self, texts: list[str]) -> None:
        docs = [tokenize(t) for t in texts]
        self.vocab = {w: i for i, w in enumerate(sorted({w for d in docs for w in d}))}
        df = Counter(w for d in docs for w in set(d))
        n = len(docs)
        self.idf = np.zeros(len(self.vocab))
        for w, i in self.vocab.items():
            self.idf[i] = math.log((n + 1) / (df[w] + 1)) + 1

    def encode(self, texts: list[str]) -> np.ndarray:
        mat = np.zeros((len(texts), len(self.vocab)))
        for row, text in enumerate(texts):
            for w, c in Counter(tokenize(text)).items():
                if w in self.vocab:
                    mat[row, self.vocab[w]] = c * self.idf[self.vocab[w]]
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return mat / norms


def load_embedder(backend: str = "auto"):
    """backend: 'auto' (try semantic, fall back), 'sentence-transformers' or 'tfidf'."""
    if backend in ("auto", "sentence-transformers"):
        try:
            return SentenceTransformerEmbedder()
        except Exception as exc:  # not installed, offline, ...
            if backend == "sentence-transformers":
                raise
            log.warning("Semantic model unavailable (%s). Falling back to TF-IDF.", exc)
    return TfidfEmbedder()


# --------------------------------------------------------------------------- BM25

class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [tokenize(t) for t in texts]
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, self.docs)) / max(len(self.docs), 1)
        self.tf = [Counter(d) for d in self.docs]
        df = Counter(w for d in self.docs for w in set(d))
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(len(self.docs))
        for i, (tf, doc) in enumerate(zip(self.tf, self.docs)):
            denom_norm = self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl)
            for w in q:
                if w in tf:
                    out[i] += self.idf[w] * tf[w] * (self.k1 + 1) / (tf[w] + denom_norm)
        return out


# --------------------------------------------------------------------------- hybrid retriever

@dataclass
class Hit:
    sentence: Sentence
    score: float      # combined score used for ranking
    semantic: float   # raw cosine similarity (used for "is this in the docs at all?")


class HybridRetriever:
    def __init__(self, sentences: list[Sentence], embedder, alpha: float = config.HYBRID_ALPHA):
        self.sentences = sentences
        self.embedder = embedder
        self.alpha = alpha
        texts = [s.search_text for s in sentences]
        embedder.fit(texts)
        self.matrix = embedder.encode(texts)
        self.bm25 = BM25(texts)

    @property
    def backend(self) -> str:
        return self.embedder.name

    def embed(self, text: str) -> np.ndarray:
        return self.embedder.encode([text])[0]

    def search(self, query: str, top_k: int = 15) -> list[Hit]:
        semantic = self.matrix @ self.embed(query)
        keyword = self.bm25.scores(query)
        if keyword.max() > 0:
            keyword = keyword / keyword.max()
        combined = self.alpha * semantic + (1 - self.alpha) * keyword
        order = np.argsort(-combined)[:top_k]
        return [Hit(self.sentences[i], float(combined[i]), float(semantic[i])) for i in order]
