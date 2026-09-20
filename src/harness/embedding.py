"""Text encoding for the semantic memory implementations.

One protocol, so a memory provider depends on "something that turns text into unit vectors"
rather than on sentence-transformers. That is what lets the unit tests encode without a
model download and keeps the network out of CI.

The shipped encoder pins a revision as well as a model name. An embedding model silently
updated under a fixed name would change retrieval quality mid-sweep, and the results would
show a difference that no condition caused - which is exactly the class of bug the
provenance header exists to catch.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

Vectors = npt.NDArray[np.float32]

#: Below this norm a vector is treated as zero and left alone rather than divided by.
MIN_NORM = 1e-12


class Encoder(Protocol):
    """Returns one L2-normalised row per input text, shape (len(texts), dim)."""

    def encode(self, texts: Sequence[str]) -> Vectors: ...


def l2_normalise(matrix: Vectors) -> Vectors:
    """Scale each row to unit length, leaving all-zero rows untouched.

    Normalising here rather than inside cosine similarity means the similarity is a plain
    dot product, which keeps the retrieval hot path a single matrix multiply.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms < MIN_NORM, 1.0, norms)
    return np.asarray(matrix / safe, dtype=np.float32)


class SentenceTransformerEncoder:
    """The shipped encoder. Loads the model on first use, never at import.

    Importing sentence-transformers costs seconds and touches the filesystem, and most of
    the test suite never needs it. Loading lazily keeps `pytest` fast and keeps a model
    download off the critical path of a config validation error.
    """

    def __init__(self, model: str, revision: str) -> None:
        self._model_name = model
        self._revision = revision
        self._model: Any | None = None

    def encode(self, texts: Sequence[str]) -> Vectors:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        raw = self._loaded().encode(list(texts), convert_to_numpy=True)
        return l2_normalise(np.asarray(raw, dtype=np.float32))

    def _loaded(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, revision=self._revision)
        return self._model
