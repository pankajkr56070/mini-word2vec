from pathlib import Path
from typing import List, Union

import numpy as np


class EmbeddingLayer:
    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.embeddings = np.random.uniform(-0.1, 0.1, size=(self.vocab_size, self.embedding_dim)).astype(np.float32)

    @property
    def shape(self) -> tuple[int, int]:
        return self.embeddings.shape

    def lookup(self, token_id: int) -> np.ndarray:
        if not isinstance(token_id, (int, np.integer)):
            raise TypeError("token_id must be an integer")

        index = int(token_id)
        if index < 0 or index >= self.vocab_size:
            raise IndexError(f"Invalid token id: {index}")
        return self.embeddings[index]

    def lookup_batch(self, token_ids: Union[List[int], np.ndarray]) -> np.ndarray:
        token_ids = np.asarray(token_ids)
        if token_ids.size == 0:
            return np.empty((0, self.embedding_dim), dtype=float)
        return self.embeddings[token_ids]

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
        layer.embeddings = data.astype(np.float32)
        return layer


def save_embeddings(model, vocabulary, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, token in enumerate(vocabulary.index_to_token):
            vector = model.input_embeddings.lookup(index)
            values = " ".join(str(value) for value in vector)
            handle.write(f"{token} {values}\n")
