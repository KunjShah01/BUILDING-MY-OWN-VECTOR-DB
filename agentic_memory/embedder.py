"""Embedding backends for Agentic Memory.

Two backends, both implemented here (no memory-layer SDK):
  - SemanticEmbedder: sentence-transformers (all-MiniLM-L6-v2, 384-d). Real
    semantic similarity, handles paraphrase. Needs the model (downloads once).
  - CharNGramEmbedder: pure-stdlib char n-gram hashing (256-d). Lexical only,
    zero deps, fully offline. Used as fallback when the model is unavailable.

`get_embedder()` picks the backend from MEM_EMBEDDING_BACKEND
(semantic | lexical | auto, default auto) and falls back to lexical if the
semantic model can't load.
"""

import hashlib
import math
import os
from typing import List


def _stable_hash(token: str) -> int:
    """Deterministic hash. Python's builtin hash() is salted per process
    (PYTHONHASHSEED), which would make embeddings non-reproducible across
    runs and silently break persisted vectors after a restart. md5 of the
    utf-8 bytes is stable across processes and platforms.
    """
    return int.from_bytes(hashlib.md5(token.encode("utf-8")).digest()[:8], "big")


class CharNGramEmbedder:
    """Character n-gram hashing embedding. No external dependencies.

    Converts text to a fixed-dimension vector using hashed character
    n-gram and word-level features. Normalized to unit length.
    """

    name = "char-ngram"

    def __init__(self, dimensions: int = 256, ngram_range: tuple = (2, 4)):
        self.dimensions = dimensions
        self.ngram_range = ngram_range

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dimensions
        text_lower = text.lower()

        # Character n-gram features
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(text_lower) - n + 1):
                ngram = text_lower[i:i + n]
                idx = _stable_hash(ngram) % self.dimensions
                sign = 1 if _stable_hash(ngram + "_s") % 2 == 0 else -1
                vec[idx] += sign * 1.0

        # Word-level features (word presence hashing)
        words = text_lower.split()
        for word in words:
            idx = _stable_hash("w_" + word) % self.dimensions
            sign = 1 if _stable_hash(word + "_t") % 2 == 0 else -1
            vec[idx] += sign * 0.5

        # Normalize to unit vector
        magnitude = math.sqrt(sum(v * v for v in vec))
        if magnitude > 0:
            vec = [v / magnitude for v in vec]
        return vec

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


class SemanticEmbedder:
    """sentence-transformers backend. Real semantic similarity.

    Model is loaded lazily on first embed so importing this module stays cheap.
    Vectors are L2-normalized, so cosine reduces to a dot product (same
    contract as CharNGramEmbedder).
    """

    name = "semantic"

    def __init__(self, model_name: str = ""):
        self.model_name = model_name or os.environ.get(
            "MEM_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )
        self._model = None
        self.dimensions = 0

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self.dimensions = self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        self._ensure_model()
        vecs = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return vecs.tolist()


def get_embedder(backend: str = ""):
    """Return an embedder per MEM_EMBEDDING_BACKEND (semantic|lexical|auto).

    auto/semantic try the model first; if it can't load (offline, no cache),
    fall back to the zero-dep lexical embedder so retrieval always works.
    """
    backend = (backend or os.environ.get("MEM_EMBEDDING_BACKEND", "auto")).lower()

    if backend == "lexical":
        return CharNGramEmbedder()

    if backend in ("semantic", "auto"):
        try:
            emb = SemanticEmbedder()
            emb._ensure_model()  # surface load errors now, not mid-request
            return emb
        except Exception as e:
            if backend == "semantic":
                raise
            import sys
            print(
                f"[embedder] semantic backend unavailable ({type(e).__name__}); "
                "falling back to char-ngram lexical backend.",
                file=sys.stderr,
            )
            return CharNGramEmbedder()

    raise ValueError(f"Unknown MEM_EMBEDDING_BACKEND: {backend!r}")


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot  # Both vectors are unit-normalized
