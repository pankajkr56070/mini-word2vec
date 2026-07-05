import numpy as np
import pytest

from src.losses import CrossEntropyLoss


def test_softmax_converts_logits_to_probabilities() -> None:
    loss = CrossEntropyLoss()

    probabilities = loss.softmax(np.array([0.10, 0.37, 0.51, 0.24]))

    assert np.allclose(probabilities, [0.20, 0.26, 0.30, 0.23], atol=0.01)
    assert np.isclose(probabilities.sum(), 1.0)


def test_softmax_is_stable_for_large_logits() -> None:
    loss = CrossEntropyLoss()

    probabilities = loss.softmax(np.array([10_000.0, 10_001.0]))

    assert np.all(np.isfinite(probabilities))
    assert np.isclose(probabilities.sum(), 1.0)


def test_forward_is_stable_for_extreme_logits() -> None:
    loss = CrossEntropyLoss()
    logits = np.array([1000, 999, 998], dtype=np.float32)

    loss_value = loss.forward(logits, 0)

    assert np.isfinite(loss_value)


def test_forward_returns_scalar_cross_entropy() -> None:
    loss = CrossEntropyLoss()

    value = loss.forward(np.zeros(4), target=0)

    assert isinstance(value, float)
    assert value == pytest.approx(np.log(4.0))


def test_forward_matches_manual_cross_entropy() -> None:
    loss = CrossEntropyLoss()
    logits = np.array([0.10, 0.37, 0.51, 0.24])
    shifted = logits - np.max(logits)
    expected = float(np.log(np.sum(np.exp(shifted))) - shifted[0])

    assert loss.forward(logits, target=0) == pytest.approx(expected)


def test_forward_rejects_invalid_target() -> None:
    loss = CrossEntropyLoss()

    with pytest.raises(IndexError, match="Invalid target id"):
        loss.forward(np.zeros(4), target=4)


def test_forward_rejects_invalid_logits() -> None:
    loss = CrossEntropyLoss()

    with pytest.raises(ValueError, match="non-empty 1D array"):
        loss.forward(np.empty((0, 4)), target=0)
