import numpy as np
import pytest

from src.embeddings import EmbeddingLayer


def test_embedding_shape() -> None:
    layer = EmbeddingLayer(vocab_size=3, embedding_dim=2)
    assert layer.shape == (3, 2)


def test_lookup() -> None:
    layer = EmbeddingLayer(vocab_size=3, embedding_dim=2)
    assert layer.lookup(0).shape == (2,)


def test_lookup_invalid() -> None:
    layer = EmbeddingLayer(vocab_size=3, embedding_dim=2)

    with pytest.raises(IndexError, match="Invalid token id"):
        layer.lookup(999)


def test_batch_lookup() -> None:
    layer = EmbeddingLayer(vocab_size=3, embedding_dim=2)
    batch = layer.lookup_batch([0, 2])
    assert batch.shape == (2, 2)


def test_empty_batch() -> None:
    layer = EmbeddingLayer(vocab_size=3, embedding_dim=2)
    assert layer.lookup_batch([]).shape == (0, 2)


def test_save(tmp_path) -> None:
    layer = EmbeddingLayer(vocab_size=2, embedding_dim=3)
    path = tmp_path / "embeddings.npy"

    layer.save(path)

    assert path.exists()


def test_load(tmp_path) -> None:
    layer = EmbeddingLayer(vocab_size=2, embedding_dim=3)
    path = tmp_path / "embeddings.npy"
    layer.save(path)
    reloaded = EmbeddingLayer.load(path)

    assert reloaded.shape == layer.shape
    assert np.allclose(reloaded.embeddings, layer.embeddings)


def test_invalid_vocab_size() -> None:
    with pytest.raises(ValueError, match="vocab_size must be positive"):
        EmbeddingLayer(vocab_size=0, embedding_dim=2)


def test_invalid_embedding_dim() -> None:
    with pytest.raises(ValueError, match="embedding_dim must be positive"):
        EmbeddingLayer(vocab_size=3, embedding_dim=0)
