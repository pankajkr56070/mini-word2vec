from pathlib import Path

from src.config import EMBEDDING_DIM, EPOCHS, LEARNING_RATE, VOCAB_SIZE, WINDOW_SIZE
from src.inference import Inference
from src.trainer import train_word2vec

WORDS = [
    "holmes",
    "watson",
    "sherlock",
    "doctor",
    "inspector",
    "police",
    "murder",
    "crime",
    "mystery",
    "case",
    "clue",
    "letter",
    "london",
    "street",
    "house",
    "room",
    "door",
    "window",
    "night",
    "voice",
]


def main() -> None:
    corpus_path = Path("data/raw/sherlock.txt")
    model, vocabulary = train_word2vec(
        corpus_path,
        embedding_dim=EMBEDDING_DIM,
        epochs=EPOCHS,
        window_size=WINDOW_SIZE,
        min_count=3,
        max_vocab=VOCAB_SIZE,
        learning_rate=LEARNING_RATE,
    )

    inference = Inference(model, vocabulary)

    print()
    print("Nearest neighbors")
    print("=================")
    for word in WORDS:
        if not vocabulary.contains(word):
            print(f"{word:12s} -> (not in vocabulary)")
            continue

        neighbors = inference.most_similar(word, top_k=5)
        formatted = ", ".join(f"{token} ({score:.2f})" for token, score in neighbors)
        print(f"{word:12s} -> {formatted}")


if __name__ == "__main__":
    main()
