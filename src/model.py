from typing import List, Union

import numpy as np

from .dataset_builder import SkipGramPair


class TinyWord2Vec:
    def __init__(self, embedding_dim: int, vocabulary):
        self.embedding_dim = embedding_dim
        self.vocabulary = vocabulary
        self.embeddings = np.zeros((self.vocabulary.size, embedding_dim), dtype=float)
        self.vector_size = embedding_dim

    def fit_from_pairs(self, pairs: List[Union[SkipGramPair, tuple[int, int]]]) -> None:
        """Fit embeddings from token ID pairs (NOT string pairs).
        
        Args:
            pairs: List of SkipGramPair or (center_id, context_id) tuples.
                   IDs must already be encoded by the vocabulary.
        """
        if not pairs:
            return

        cooccurrence = np.zeros((self.vocabulary.size, self.vocabulary.size), dtype=float)
        for pair in pairs:
            # Extract IDs from SkipGramPair or tuple
            if hasattr(pair, 'center'):  # SkipGramPair dataclass
                center_id, context_id = pair.center, pair.context
            else:  # tuple of ints
                center_id, context_id = pair
            cooccurrence[center_id, context_id] += 1.0

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
