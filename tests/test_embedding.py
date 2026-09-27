from __future__ import annotations

import sys
from typing import Any, ClassVar

import numpy as np
import pytest

from harness.embedding import MIN_NORM, SentenceTransformerEncoder, l2_normalise, load_model


class _StubModel:
    """Stands in for a loaded SentenceTransformer, recording how it was constructed."""

    built: ClassVar[list[_StubModel]] = []

    prompts: ClassVar[dict[str, str]] = {"query": "Query:", "document": ""}

    def __init__(self, name: str, revision: str) -> None:
        self.name = name
        self.revision = revision
        self.prompt_names: list[str | None] = []
        type(self).built.append(self)

    def encode(
        self, texts: list[str], convert_to_numpy: bool = True, prompt_name: str | None = None
    ) -> Any:
        self.prompt_names.append(prompt_name)
        return np.array([[float(len(text)), 1.0] for text in texts], dtype=np.float32)


@pytest.fixture
def stub_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> type[_StubModel]:
    """Install a stub module so no test ever downloads a model or touches the network."""
    _StubModel.built = []
    load_model.cache_clear()
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


def test_queries_carry_the_models_query_prompt_and_documents_do_not(
    stub_sentence_transformers: type[_StubModel],
) -> None:
    # Arrange
    encoder = SentenceTransformerEncoder("some/model", "abc123")

    # Act
    encoder.encode(["a document"])
    encoder.encode_queries(["a question"])

    # Assert
    assert stub_sentence_transformers.built[0].prompt_names == [None, "query"]


def test_a_model_without_a_query_prompt_encodes_queries_plainly(
    stub_sentence_transformers: type[_StubModel], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setattr(_StubModel, "prompts", {})
    encoder = SentenceTransformerEncoder("some/model", "abc123")

    # Act
    encoder.encode_queries(["a question"])

    # Assert
    assert stub_sentence_transformers.built[0].prompt_names == [None]


def test_the_loaded_model_is_shared_across_encoders(
    stub_sentence_transformers: type[_StubModel],
) -> None:
    """The builder makes an encoder per run; the model must not reload per run."""
    # Act
    SentenceTransformerEncoder("some/model", "abc123").encode(["one"])
    SentenceTransformerEncoder("some/model", "abc123").encode(["two"])

    # Assert
    assert len(stub_sentence_transformers.built) == 1
