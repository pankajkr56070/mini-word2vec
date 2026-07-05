from pathlib import Path
from typing import List, Tuple, Union

import numpy as np

from .config import EMBEDDING_DIM, EPOCHS, LEARNING_RATE, MIN_FREQUENCY, VOCAB_SIZE, WINDOW_SIZE
from .dataset_builder import DatasetBuilder, SkipGramPair
from .model import Word2VecModel
from .preprocessor import load_corpus
from .tokenizer import Tokenizer
from .vocabulary import Vocabulary


def train_word2vec(
    corpus: Union[str, Path, List[str]],
    embedding_dim: int = EMBEDDING_DIM,
    epochs: int = EPOCHS,
    window_size: int = WINDOW_SIZE,
    min_count: int = MIN_FREQUENCY,
    max_vocab: int = VOCAB_SIZE,
    learning_rate: float = LEARNING_RATE,
) -> Tuple[Word2VecModel, Vocabulary]:
    """Train a tiny embedding model using co-occurrence statistics."""
    if isinstance(corpus, (str, Path)):
        corpus_path = Path(corpus)
        if corpus_path.exists():
            tokens = load_corpus(corpus_path)
        else:
            tokens = Tokenizer().tokenize(str(corpus))
    else:
        tokens = list(corpus)

    vocab = Vocabulary(min_count=min_count, max_vocab=max_vocab)
    vocab.fit(tokens)

    token_ids = vocab.encode(tokens)
    dataset_builder = DatasetBuilder()
    pairs = dataset_builder.build_skipgram_pairs(token_ids, window_size=window_size)
    model = Word2VecModel(vocab_size=vocab.size, embedding_dim=embedding_dim)
    embeddings = _fit_embeddings_from_pairs(pairs, vocab.size, embedding_dim)
    model.input_embeddings.embeddings = embeddings
    model.output_embeddings.embeddings = embeddings.copy()
    return model, vocab


def _fit_embeddings_from_pairs(
    pairs: List[SkipGramPair],
    vocab_size: int,
    embedding_dim: int,
) -> np.ndarray:
    """Fit co-occurrence embeddings outside the model's forward pass."""
    cooccurrence = np.zeros((vocab_size, vocab_size), dtype=float)
    for pair in pairs:
        cooccurrence[pair.center, pair.context] += 1.0

    if not np.any(cooccurrence):
        return np.zeros((vocab_size, embedding_dim), dtype=np.float32)

    _, singular_values, right_singular_vectors = np.linalg.svd(cooccurrence, full_matrices=False)
    rank = min(embedding_dim, len(singular_values))
    embeddings = right_singular_vectors[:rank].T * np.sqrt(singular_values[:rank])
    if rank < embedding_dim:
        embeddings = np.pad(embeddings, ((0, 0), (0, embedding_dim - rank)))
    return embeddings.astype(np.float32)
