from dataclasses import dataclass
from typing import Optional


VOCAB_SIZE = 5000
WINDOW_SIZE = 5
EMBEDDING_DIM = 32
LEARNING_RATE = 0.05
EPOCHS = 10
MIN_FREQUENCY = 2


@dataclass
class PipelineConfig:
    min_count: int = MIN_FREQUENCY
    max_vocab: Optional[int] = None
    window_size: int = WINDOW_SIZE


@dataclass
class TrainingConfig:
    embedding_dim: int = EMBEDDING_DIM
    epochs: int = EPOCHS
    window_size: int = WINDOW_SIZE
    min_count: int = MIN_FREQUENCY
    max_vocab: int = VOCAB_SIZE
    learning_rate: float = LEARNING_RATE
