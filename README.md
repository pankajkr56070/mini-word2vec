# mini-word2vec

A small Word2Vec-style embedding pipeline that reads raw text, preprocesses it, tokenizes it, builds a vocabulary, and saves the resulting artifacts.

## Project structure

- `data/raw/`: raw source text files
- `data/processed/`: generated artifacts
- `src/`: project source code
- `tests/`: unit and integration tests
- `train.py`: training entry point for embeddings

## What it does

1. Reads the raw corpus from `data/raw/sherlock.txt`
2. Cleans and normalizes the text
3. Tokenizes the cleaned text
4. Builds a vocabulary and token-ID mappings
5. Saves artifacts to `data/processed/`

## Saved artifacts

The pipeline writes:

- `data/processed/clean_text.txt`
- `data/processed/vocabulary.json`
- `data/processed/word_to_id.json`
- `data/processed/id_to_word.json`
- `data/processed/token_ids.npy`

## Usage

```bash
cd /Users/corleone/Projects/mini-word2vec
python train.py
```

Or use the pipeline directly from Python:

```python
from src.config import PipelineConfig
from src.pipeline import Pipeline

pipeline = Pipeline(PipelineConfig(min_count=1, max_vocab=5000))
output = pipeline.run(
    input_path="data/raw/sherlock.txt",
    output_dir="data/processed",
)
```

## Testing

Run all tests with:

```bash
PYTHONPATH=. .venv/bin/pytest -q tests
```

## GitHub

This repository is initialized with Git and ready to push to GitHub as `pankajkr56070/mini-word2vec`.

