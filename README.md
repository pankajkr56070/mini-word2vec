# mini-word2vec

A small Word2Vec-style embedding pipeline that reads raw text, preprocesses it, tokenizes it, builds a vocabulary, and generates training data.

## Project structure

- `data/raw/`: raw source text files
- `outputs/`: generated artifacts and checkpoints
- `src/`: project source code
- `tests/`: unit and integration tests
- `train.py`: training entry point for embeddings

## Architecture

The pipeline follows **Single Responsibility Principle**:

- **Preprocessor**: Cleans and normalizes text
- **Tokenizer**: Converts text to tokens
- **Vocabulary**: Builds token-ID mappings
- **DatasetBuilder**: Generates training data structures (skip-gram pairs, CBOW samples) — **no file I/O**
- **Pipeline**: Orchestrates the workflow and handles file persistence

## What it does

1. Reads the raw corpus from `data/raw/sherlock.txt`
2. Cleans and normalizes the text
3. Tokenizes the cleaned text
4. Builds a vocabulary and token-ID mappings
5. Generates skip-gram training pairs from token IDs
6. Saves artifacts to `outputs/processed/`

## Training data formats

### Skip-gram pairs

The `DatasetBuilder.build_skipgram_pairs()` method generates pairs of (center_word, context_word):

```python
from src.dataset_builder import SkipGramPair

# Returns: [SkipGramPair(center=0, context=1), SkipGramPair(center=1, context=0), ...]
pairs = builder.build_skipgram_pairs(token_ids, window_size=2)
```

### CBOW samples

The `DatasetBuilder.build_cbow_samples()` method (coming soon) will generate (context_words, target_word) tuples.

## Saved artifacts

The pipeline writes to `outputs/processed/`:

- `clean_text.txt`: Preprocessed text
- `vocabulary.json`: Full vocabulary (for reference)
- `word_to_id.json`: Word → ID mappings
- `id_to_word.json`: ID → Word mappings
- `token_ids.npy`: Encoded token IDs (numpy format)
- `skipgram_pairs.json`: Skip-gram training pairs (center_id, context_id)

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

# Access generated training data
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

