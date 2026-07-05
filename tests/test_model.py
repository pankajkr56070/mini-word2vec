import numpy as np
import pytest

from src.model import Word2VecModel


def build_model() -> Word2VecModel:
    return Word2VecModel(vocab_size=16, embedding_dim=4)


def test_forward_shape() -> None:
    model = build_model()

    assert model.forward(10).shape == (16,)


def test_forward_batch_shape() -> None:
    model = build_model()

    assert model.forward_batch(np.array([1, 10])).shape == (2, 16)


def test_empty_batch() -> None:
    model = build_model()

    assert model.forward_batch([]).shape == (0, 16)


def test_invalid_token() -> None:
    model = build_model()

    with pytest.raises(IndexError, match="Invalid token id"):
        model.forward(16)


def test_input_output_matrix_shape() -> None:
    model = build_model()

    assert model.input_matrix.shape == (16, 4)
    assert model.output_matrix.shape == (16, 4)


def test_forward_equals_batch_single() -> None:
    model = build_model()

    single = model.forward(10)
    batch = model.forward_batch([10])

    assert np.allclose(single, batch[0])
