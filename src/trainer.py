from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .config import EMBEDDING_DIM, EPOCHS, LEARNING_RATE, MIN_FREQUENCY, VOCAB_SIZE, WINDOW_SIZE
from .dataset_builder import DatasetBuilder, SkipGramPair
from .losses import CrossEntropyLoss
from .model import Word2VecModel
from .preprocessor import load_corpus
from .tokenizer import Tokenizer
from .vocabulary import Vocabulary


class Trainer:
    """Runs SGD training over (center, context) pairs.

    Model-agnostic: only calls `model.forward`, `criterion.forward`,
    `criterion.backward`, and `model.backward`. Knows nothing about
    embeddings, softmax, vocabulary, or tokenization.

    `average_loss()` is the running average over every step across all
    epochs; `epoch_losses` holds each completed epoch's own average loss.
    """

    def __init__(
        self,
        model: Word2VecModel,
        criterion: CrossEntropyLoss,
        dataset: List[SkipGramPair],
        learning_rate: float,
    ) -> None:
        if model is None:
            raise ValueError("model must not be None")
        if criterion is None:
            raise ValueError("criterion must not be None")
        if not dataset:
            raise ValueError("dataset must not be empty")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        self.model = model
        self.criterion = criterion
        self.dataset = dataset
        self.learning_rate = learning_rate

        self.steps_completed = 0
        self.pairs_processed = 0
        self.current_loss: Optional[float] = None
        self.epoch_losses: List[float] = []
        self._loss_sum = 0.0

    def train(self, epochs: int) -> None:
        if epochs <= 0:
            raise ValueError("epochs must be positive")

        for epoch in range(1, epochs + 1):
            epoch_loss = self.train_epoch()
            self._log_epoch(epoch, epochs, epoch_loss)

    def train_epoch(self) -> float:
        epoch_loss_sum = 0.0
        for pair in self.dataset:
            epoch_loss_sum += self._train_step(pair)

        epoch_loss = epoch_loss_sum / len(self.dataset)
        self.epoch_losses.append(epoch_loss)
        return epoch_loss

    def _train_step(self, pair: SkipGramPair) -> float:
        logits = self.model.forward(pair.center)
        loss_value = self.criterion.forward(logits, pair.context)
        gradient_logits = self.criterion.backward(logits, pair.context)
        self.model.backward(pair.center, gradient_logits, self.learning_rate)

        self.steps_completed += 1
        self.pairs_processed += 1
        self._loss_sum += loss_value
        self.current_loss = loss_value
        return loss_value

    def average_loss(self) -> float:
        """Running average loss across every step completed so far, all epochs combined."""
        if self.steps_completed == 0:
            return 0.0
        return self._loss_sum / self.steps_completed

    def _log_epoch(self, epoch: int, epochs: int, epoch_loss: float) -> None:
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'Pairs processed':<16} : {self.pairs_processed:,}")
        print(f"{'Average Loss':<16} : {epoch_loss:.2f}")
        print(f"{'Learning Rate':<16} : {self.learning_rate:g}")

    @property
    def metrics(self) -> Dict[str, object]:
        return {
            "steps_completed": self.steps_completed,
            "pairs_processed": self.pairs_processed,
            "current_loss": self.current_loss,
            "average_loss": self.average_loss(),
            "epoch_losses": list(self.epoch_losses),
        }


def train_word2vec(
    corpus: Union[str, Path, List[str]],
    embedding_dim: int = EMBEDDING_DIM,
    epochs: int = EPOCHS,
    window_size: int = WINDOW_SIZE,
    min_count: int = MIN_FREQUENCY,
    max_vocab: int = VOCAB_SIZE,
    learning_rate: float = LEARNING_RATE,
) -> Tuple[Word2VecModel, Vocabulary]:
    """Convenience wrapper: builds a vocab/dataset/model and runs a Trainer over it."""
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
    pairs = DatasetBuilder().build_skipgram_pairs(token_ids, window_size=window_size)

    model = Word2VecModel(vocab_size=vocab.size, embedding_dim=embedding_dim)
    trainer = Trainer(model, CrossEntropyLoss(), pairs, learning_rate)
    trainer.train(epochs)

    return trainer.model, vocab
