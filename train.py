from pathlib import Path

from src.config import EMBEDDING_DIM, EPOCHS, VOCAB_SIZE, WINDOW_SIZE
from src.embeddings import save_embeddings
from src.trainer import train_word2vec


def main() -> None:
    corpus_path = Path("data/raw/sherlock.txt")
    model, vocabulary = train_word2vec(
        corpus_path,
        embedding_dim=EMBEDDING_DIM,
        epochs=EPOCHS,
        window_size=WINDOW_SIZE,
        min_count=3,
        max_vocab=VOCAB_SIZE,
    )
    output_path = Path("models/sherlock_embeddings.txt")
    save_embeddings(model, vocabulary, output_path)
    print(f"Saved embeddings to {output_path}")


if __name__ == "__main__":
    main()
