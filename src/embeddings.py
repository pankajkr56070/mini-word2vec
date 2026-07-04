from pathlib import Path
from typing import Optional, Union

import numpy as np

class EmbeddingLayer:
    def __init__(
        self,
        vocab_size: Optional[int] = None,
        embedding_dim: Optional[int] = None,
        vocabulary=None,
        vectors: Optional[np.ndarray] = None,
    ) -> None:
        if vocabulary is not None:
            vocab_size = vocabulary.size
        if vocab_size is None or embedding_dim is None:
            raise TypeError("EmbeddingLayer requires vocab_size and embedding_dim")

        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.vector_size = self.embedding_dim
        self.embeddings = np.zeros((self.vocab_size, self.embedding_dim), dtype=float)

        if vectors is not None:
            vectors = np.asarray(vectors, dtype=float)
            expected_shape = (self.vocab_size, self.embedding_dim)
            if vectors.shape == (self.vocab_size - 1, self.embedding_dim):
                vectors = np.vstack([np.zeros((1, self.embedding_dim), dtype=float), vectors])
            if vectors.shape != expected_shape:
                raise ValueError(f"Expected embedding matrix with shape {expected_shape}, got {vectors.shape}")
            self.embeddings = vectors

    def lookup(self, token_id: int) -> np.ndarray:
        if not isinstance(token_id, (int, np.integer)):
            raise TypeError("token_id must be an integer")

        index = int(token_id)
        if index < 0 or index >= self.vocab_size:
            return np.zeros(self.embedding_dim, dtype=float)
        return self.embeddings[index].astype(float)

    def lookup_batch(self, token_ids: list[int]) -> np.ndarray:
        return np.vstack([self.lookup(token_id) for token_id in token_ids])

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.embeddings)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EmbeddingLayer":
        path = Path(path)
        data = np.load(path)
        if data.ndim != 2:
            raise ValueError("Expected a 2D embedding matrix")
        layer = cls(vocab_size=data.shape[0], embedding_dim=data.shape[1])
        layer.embeddings = data.astype(float)
        return layer


Embeddings = EmbeddingLayer
