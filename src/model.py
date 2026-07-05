from typing import List, Union

import numpy as np

from .embeddings import EmbeddingLayer
from .losses import CrossEntropyLoss


class Word2VecModel:
    """Compute raw context-word scores for center-word token IDs."""

    @property
    def input_matrix(self) -> np.ndarray:
        return self.input_embeddings.embeddings

    @property
    def output_matrix(self) -> np.ndarray:
        return self.output_embeddings.embeddings

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.input_embeddings = EmbeddingLayer(vocab_size, embedding_dim)
        self.output_embeddings = EmbeddingLayer(vocab_size, embedding_dim)

    def forward(self, token_id: int) -> np.ndarray:
        """Returns logits."""
        center_vector = self.input_embeddings.lookup(token_id)
        return self.output_matrix @ center_vector

    def forward_batch(self, token_ids: Union[List[int], np.ndarray]) -> np.ndarray:
        """Returns logits for a batch of token IDs."""
        token_ids = np.asarray(token_ids)
        if token_ids.size == 0:
            return np.empty((0, self.vocab_size), dtype=np.float32)

        center_vectors = self.input_embeddings.lookup_batch(token_ids)
        return center_vectors @ self.output_matrix.T


    def backward(self, center_id : int, gradient_logits: np.ndarray, learning_rate : float) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        gradient_logits = np.asarray(gradient_logits, dtype=np.float32)

        if gradient_logits.shape != (self.vocab_size,):
            raise ValueError("gradient_logits must have shape (self.vocab_size)")
        hidden = self.input_embeddings.lookup(center_id)
        output_embeddings = self.output_embeddings.embeddings
        input_embeddings = self.input_embeddings.embeddings

        gradient_output_embeddings = np.outer(gradient_logits, hidden)
        gradient_hidden = output_embeddings.T @ gradient_logits
        output_embeddings -= learning_rate * gradient_output_embeddings
        input_embeddings[center_id] -= learning_rate * gradient_hidden

