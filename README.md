# mini-word2vec

A compact Word2Vec-style embedding project that reads raw text, preprocesses it, tokenizes it, builds a vocabulary, creates skip-gram pairs, and trains a small embedding layer.

## Project structure

- `data/raw/`: raw source text files
- `outputs/`: generated artifacts and checkpoints
- `src/`: project source code
- `tests/`: unit and integration tests
- `train.py`: training entry point

## Architecture

The project follows a small, focused separation of responsibilities:

- **Preprocessor**: cleans and normalizes text
- **Tokenizer**: converts text into tokens
- **Vocabulary**: builds token-to-ID mappings
- **DatasetBuilder**: builds skip-gram training pairs from token IDs
- **EmbeddingLayer**: stores and looks up dense vector representations by token ID
- **TinyWord2Vec**: trains embeddings from co-occurrence data and exposes vector lookup
- **Pipeline**: orchestrates the workflow and writes artifacts

## What it does

1. Reads a corpus from `data/raw/sherlock.txt`
2. Cleans and normalizes the text
3. Tokenizes the cleaned text
4. Builds a vocabulary and token-ID mappings
5. Generates skip-gram training pairs
6. Trains a small embedding matrix and saves it for later use

## Embedding layer

The embedding layer is intentionally small and focused:

```python
from src.embeddings import EmbeddingLayer

layer = EmbeddingLayer(vocab_size=100, embedding_dim=32)
vector = layer.lookup(5)
batch = layer.lookup_batch([1, 5, 10])
layer.save("outputs/embeddings.npy")
reloaded = EmbeddingLayer.load("outputs/embeddings.npy")
```

It supports:

- `lookup(token_id)` for a single vector
- `lookup_batch(token_ids)` for a batch of vectors
- `save(path)` and `load(path)` for persistence

## Training data formats

### Skip-gram pairs

The `DatasetBuilder.build_skipgram_pairs()` method generates pairs of token IDs:

```python
from src.dataset_builder import DatasetBuilder

builder = DatasetBuilder()
pairs = builder.build_skipgram_pairs(token_ids, window_size=2)
```

## Saved artifacts

The pipeline writes to `outputs/processed/`:

- `clean_text.txt`: preprocessed text
- `vocabulary.json`: vocabulary metadata and counts
- `word_to_id.json`: word → ID mappings
- `id_to_word.json`: ID → word mappings
- `token_ids.npy`: encoded token IDs
- `skipgram_pairs.json`: skip-gram training pairs

## Usage

```bash
cd /Users/corleone/Projects/mini-word2vec
python train.py
```

Or use the pipeline directly from Python:

```python
from src.config import PipelineConfig
from src.pipeline import Pipeline

pipeline = Pipeline(PipelineConfig(min_count=1, max_vocab=5000, window_size=2))
output = pipeline.run(
    input_path="data/raw/sherlock.txt",
    output_dir="outputs",
)

skipgram_pairs = output["skipgram_pairs"]
vocabulary = output["vocabulary"]
```

## Testing

Run all tests with:

```bash
PYTHONPATH=. .venv/bin/pytest -q tests
```

## GitHub

This repository is initialized with Git and ready to push to GitHub as `pankajkr56070/mini-word2vec`.

