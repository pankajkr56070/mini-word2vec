from typing import List, Tuple

import numpy as np

from .model import Word2VecModel
from .vocabulary import Vocabulary


class Inference:
    """Read-only queries over a trained Word2VecModel + Vocabulary."""

    def __init__(self, model: Word2VecModel, vocabulary: Vocabulary) -> None:
        self.model = model
        self.vocabulary = vocabulary

    def embedding(self, word: str) -> np.ndarray:
        if not self.vocabulary.contains(word):
            raise ValueError(f"Unknown word: {word!r}")
        return self.model.input_embeddings.lookup(self.vocabulary.to_index(word)).copy()

    def similarity(self, word1: str, word2: str) -> float:
        return self._cosine_similarity(self.embedding(word1), self.embedding(word2))

    def most_similar(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        neighbors = self.nearest(self.embedding(word), top_k=top_k + 1)
        return [(token, score) for token, score in neighbors if token != word][:top_k]

    def nearest(self, vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        similarities = [
            (token, self._cosine_similarity(vector, self.model.input_embeddings.lookup(idx)))
            for idx, token in enumerate(self.vocabulary.index_to_token)
            if token != self.vocabulary.unk_token
        ]
        similarities.sort(key=lambda item: item[1], reverse=True)
        return similarities[:top_k]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
