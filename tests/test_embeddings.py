import numpy as np

from src.embeddings import EmbeddingLayer
from src.vocabulary import Vocabulary


def test_embeddings_lookup_uses_token_ids_and_tokens() -> None:
    vocab = Vocabulary(min_count=1)
    vocab.build(["hello", "world"])

    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    embeddings = EmbeddingLayer(vocab_size=vocab.size, embedding_dim=2, vectors=vectors)

    assert np.allclose(embeddings.lookup(1), np.array([1.0, 0.0]))
    assert np.allclose(embeddings.lookup(2), np.array([0.0, 1.0]))
    assert np.allclose(embeddings.lookup_batch([1, 2]), np.array([[1.0, 0.0], [0.0, 1.0]]))
