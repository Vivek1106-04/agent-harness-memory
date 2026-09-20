from __future__ import annotations

import sys
from typing import Any, ClassVar

import numpy as np
import pytest

from harness.embedding import MIN_NORM, SentenceTransformerEncoder, l2_normalise


class _StubModel:
    """Stands in for a loaded SentenceTransformer, recording how it was constructed."""

    built: ClassVar[list[_StubModel]] = []

    def __init__(self, name: str, revision: str) -> None:
        self.name = name
        self.revision = revision
        type(self).built.append(self)

    def encode(self, texts: list[str], convert_to_numpy: bool = True) -> Any:
        return np.array([[float(len(text)), 1.0] for text in texts], dtype=np.float32)


@pytest.fixture
def stub_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> type[_StubModel]:
    """Install a stub module so no test ever downloads a model or touches the network."""
    _StubModel.built = []
    module = type("_Module", (), {"SentenceTransformer": _StubModel})
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return _StubModel


def test_l2_normalise_gives_every_row_unit_length() -> None:
    # Arrange
    matrix = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)

    # Act
    normalised = l2_normalise(matrix)

    # Assert
    assert np.linalg.norm(normalised, axis=1) == pytest.approx([1.0, 1.0])


def test_l2_normalise_leaves_a_zero_row_alone_rather_than_dividing_by_zero() -> None:
    # Arrange
    matrix = np.array([[0.0, 0.0], [0.0, 2.0]], dtype=np.float32)

    # Act
    normalised = l2_normalise(matrix)

    # Assert
    assert normalised[0].tolist() == [0.0, 0.0]
    assert np.linalg.norm(normalised[1]) == pytest.approx(1.0)


def test_a_vector_under_the_norm_floor_is_left_untouched() -> None:
    matrix = np.array([[MIN_NORM / 10, 0.0]], dtype=np.float32)

    assert l2_normalise(matrix)[0].tolist() == matrix[0].tolist()


def test_encoding_nothing_does_not_load_the_model() -> None:
    """Config validation must not pay for a model download to discover a typo."""
    # Arrange
    _StubModel.built = []
    encoder = SentenceTransformerEncoder("some/model", "abc123")

    # Act
    result = encoder.encode([])

    # Assert
    assert result.shape == (0, 0)
    assert _StubModel.built == []


def test_the_model_is_loaded_once_and_reused(
    stub_sentence_transformers: type[_StubModel],
) -> None:
    # Arrange
    encoder = SentenceTransformerEncoder("some/model", "abc123")

    # Act
    encoder.encode(["one"])
    encoder.encode(["two"])

    # Assert
    assert len(stub_sentence_transformers.built) == 1


def test_encoded_rows_come_back_normalised(
    stub_sentence_transformers: type[_StubModel],
) -> None:
    """A provider doing cosine as a dot product depends on this, not on politeness."""
    # Arrange
    encoder = SentenceTransformerEncoder("some/model", "abc123")

    # Act
    vectors = encoder.encode(["alpha", "a much longer piece of text"])

    # Assert
    assert np.linalg.norm(vectors, axis=1) == pytest.approx([1.0, 1.0])


def test_the_pinned_revision_reaches_the_model(
    stub_sentence_transformers: type[_StubModel],
) -> None:
    """An embedding model updated under a fixed name would change results mid-sweep."""
    # Arrange
    encoder = SentenceTransformerEncoder("some/model", "pinned-revision")

    # Act
    encoder.encode(["text"])

    # Assert
    assert stub_sentence_transformers.built[0].revision == "pinned-revision"
    assert stub_sentence_transformers.built[0].name == "some/model"
