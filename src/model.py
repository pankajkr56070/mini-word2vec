from typing import List

import numpy as np


class TinyWord2Vec:
    def __init__(self, embedding_dim: int, vocabulary):
        self.embedding_dim = embedding_dim
        self.vocabulary = vocabulary
        self.embeddings = np.zeros((self.vocabulary.size, embedding_dim), dtype=float)
        self.vector_size = embedding_dim

    def fit_from_pairs(self, pairs: List[tuple[str, str]]) -> None:
        if not pairs:
            return

        cooccurrence = np.zeros((self.vocabulary.size, self.vocabulary.size), dtype=float)
        for source, target in pairs:
            source_idx = self.vocabulary.to_index(source)
            target_idx = self.vocabulary.to_index(target)
            if source_idx == self.vocabulary.unk_index or target_idx == self.vocabulary.unk_index:
                continue
            cooccurrence[source_idx, target_idx] += 1.0

        if np.allclose(cooccurrence, 0):
            return

        cooccurrence = cooccurrence + 1.0
        _, singular_values, right_singular_vectors = np.linalg.svd(cooccurrence, full_matrices=False)
        rank = min(self.embedding_dim, len(singular_values))
        self.embeddings = right_singular_vectors[:, :rank].T.astype(float)
        if rank < self.embedding_dim:
            padding = np.zeros((self.embedding_dim - rank, self.vocabulary.size), dtype=float)
            self.embeddings = np.vstack([self.embeddings, padding])
        self.embeddings = self.embeddings[:, : self.vocabulary.size].T

    def get_vector(self, token: str) -> np.ndarray:
        index = self.vocabulary.to_index(token)
        if index == self.vocabulary.unk_index:
            return np.zeros(self.embedding_dim, dtype=float)
        return self.embeddings[index].astype(float)
