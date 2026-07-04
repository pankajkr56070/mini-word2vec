from pathlib import Path
from typing import List, Union

from .config import EMBEDDING_DIM, EPOCHS, LEARNING_RATE, MIN_FREQUENCY, VOCAB_SIZE, WINDOW_SIZE
from .dataset_builder import DatasetBuilder
from .model import TinyWord2Vec
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
) -> TinyWord2Vec:
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
    model = TinyWord2Vec(embedding_dim=embedding_dim, vocabulary=vocab)
    model.fit_from_pairs(pairs)
    return model
