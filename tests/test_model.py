import numpy as np
import pytest

from src.losses import CrossEntropyLoss
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


def test_backward_updates_output_matrix_and_center_row_only() -> None:
    model = build_model()
    center_id, other_id = 3, 0
    original_output = model.output_matrix.copy()
    original_other_row = model.input_embeddings.embeddings[other_id].copy()

    gradient_logits = np.zeros(model.vocab_size, dtype=np.float32)
    gradient_logits[5] = 1.0

    model.backward(center_id, gradient_logits, learning_rate=0.1)

    assert not np.allclose(model.output_matrix, original_output)
    assert not np.allclose(model.input_embeddings.embeddings[center_id], np.zeros(model.embedding_dim))
    assert np.allclose(model.input_embeddings.embeddings[other_id], original_other_row)


def test_backward_matches_numerical_gradient() -> None:
    model = build_model()
    loss_fn = CrossEntropyLoss()
    center_id, target = 2, 7
    eps = 1e-4

    def loss_at() -> float:
        return loss_fn.forward(model.forward(center_id), target)

    gradient_logits = loss_fn.backward(model.forward(center_id), target)
    hidden = model.input_embeddings.lookup(center_id).copy()
    analytic_output_grad = np.outer(gradient_logits, hidden)
    analytic_hidden_grad = model.output_matrix.T @ gradient_logits

    numeric_output_grad = np.zeros_like(model.output_matrix)
    for i in range(model.vocab_size):
        for j in range(model.embedding_dim):
            original = model.output_matrix[i, j]
            model.output_matrix[i, j] = original + eps
            loss_plus = loss_at()
            model.output_matrix[i, j] = original - eps
            loss_minus = loss_at()
            model.output_matrix[i, j] = original
            numeric_output_grad[i, j] = (loss_plus - loss_minus) / (2 * eps)

    numeric_hidden_grad = np.zeros(model.embedding_dim)
    for j in range(model.embedding_dim):
        original = model.input_embeddings.embeddings[center_id, j]
        model.input_embeddings.embeddings[center_id, j] = original + eps
        loss_plus = loss_at()
        model.input_embeddings.embeddings[center_id, j] = original - eps
        loss_minus = loss_at()
        model.input_embeddings.embeddings[center_id, j] = original
        numeric_hidden_grad[j] = (loss_plus - loss_minus) / (2 * eps)

    assert np.allclose(numeric_output_grad, analytic_output_grad, atol=1e-2)
    assert np.allclose(numeric_hidden_grad, analytic_hidden_grad, atol=1e-2)


def test_backward_step_reduces_loss_for_target() -> None:
    model = build_model()
    loss_fn = CrossEntropyLoss()
    center_id, target = 4, 9

    logits_before = model.forward(center_id)
    loss_before = loss_fn.forward(logits_before, target)

    gradient_logits = loss_fn.backward(logits_before, target)
    model.backward(center_id, gradient_logits, learning_rate=0.5)

    loss_after = loss_fn.forward(model.forward(center_id), target)

    assert loss_after < loss_before


def test_backward_rejects_non_positive_learning_rate() -> None:
    model = build_model()
    gradient_logits = np.zeros(model.vocab_size, dtype=np.float32)

    with pytest.raises(ValueError, match="learning_rate must be positive"):
        model.backward(0, gradient_logits, learning_rate=0.0)


def test_backward_rejects_wrong_gradient_shape() -> None:
    model = build_model()
    gradient_logits = np.zeros(model.vocab_size - 1, dtype=np.float32)

    with pytest.raises(ValueError, match="gradient_logits must have shape"):
        model.backward(0, gradient_logits, learning_rate=0.1)


def test_backward_rejects_invalid_center_id() -> None:
    model = build_model()
    gradient_logits = np.zeros(model.vocab_size, dtype=np.float32)

    with pytest.raises(IndexError, match="Invalid token id"):
        model.backward(model.vocab_size, gradient_logits, learning_rate=0.1)
