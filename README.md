# mini-word2vec

A compact Word2Vec-style embedding project that reads raw text, preprocesses it, tokenizes it, builds a vocabulary, creates skip-gram pairs, and fits a small embedding layer from co-occurrence statistics.

## Project structure

- `data/raw/`: raw source text files
- `models/`: trained embeddings saved by `train.py` (plaintext, one token + vector per line)
- `outputs/`: generated artifacts and checkpoints
- `src/`: project source code
- `tests/`: unit and integration tests
- `train.py`: training entry point

## Architecture

The project follows a small, focused separation of responsibilities:

- **Preprocessor**: cleans and normalizes text
- **Tokenizer**: converts text into tokens
- **Vocabulary**: builds token-to-ID mappings, keeping the `max_vocab` most frequent tokens
- **DatasetBuilder**: builds skip-gram training pairs from token IDs
- **EmbeddingLayer**: stores and looks up dense vector representations by token ID
- **Word2VecModel**: holds separate input/output `EmbeddingLayer`s; `forward`/`forward_batch` compute raw context logits from center-word IDs, and `backward` applies one full-softmax gradient-descent step (given `dL/dlogits` and a learning rate) to both embedding matrices
- **CrossEntropyLoss**: `forward` computes numerically-stable softmax/cross-entropy over a logits vector; `backward` returns `dL/dlogits` (`softmax(logits) - onehot(target)`) for a target context word
- **Pipeline**: orchestrates preprocessing (clean → tokenize → vocab → skip-gram pairs) and writes intermediate artifacts
- **Trainer** (`train_word2vec`): the actual model-producing path — rebuilds vocab/skip-gram pairs and fits the embedding matrix from the center/context co-occurrence matrix via SVD, then copies it into a `Word2VecModel`
- **Inference** (`most_similar`): ranks vocabulary words by cosine similarity to a query word's embedding

`Pipeline` and `train_word2vec` are independent: `Pipeline` only produces preprocessing artifacts (it does not train anything), while `train_word2vec` re-derives its own vocab/pairs and produces the trained `Word2VecModel`. `train.py` uses `train_word2vec` directly, not `Pipeline`.

`Word2VecModel.forward`/`backward` and `CrossEntropyLoss` provide the building blocks for gradient-descent training (verified correct against numerical gradients in `tests/test_model.py`), but `train_word2vec` does not call them yet — it still fits embeddings via SVD, not SGD. `backward` updates the *entire* output embedding matrix per call (full softmax, no negative sampling), so wiring it into a real training loop over skip-gram pairs would need that considered for larger vocabularies.

## What it does

Preprocessing (`Pipeline`):

1. Reads a corpus from `data/raw/sherlock.txt`
2. Cleans and normalizes the text
3. Tokenizes the cleaned text
4. Builds a vocabulary and token-ID mappings
5. Generates skip-gram training pairs

Training (`train_word2vec`, used by `train.py`):

1. Repeats steps 2-5 above independently of `Pipeline`
2. Fits an embedding matrix by taking the SVD of the center/context co-occurrence matrix (not gradient descent)
3. Saves the resulting embeddings to `models/`

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

## Gradient-descent building blocks

`Word2VecModel` and `CrossEntropyLoss` together support one manual SGD step for a `(center, target)` pair:

```python
from src.model import Word2VecModel
from src.losses import CrossEntropyLoss

model = Word2VecModel(vocab_size=100, embedding_dim=32)
loss_fn = CrossEntropyLoss()

logits = model.forward(center_id)
gradient_logits = loss_fn.backward(logits, target=target_id)
model.backward(center_id, gradient_logits, learning_rate=0.05)
```

This path is exercised by tests but is not currently used by `train_word2vec`, which fits embeddings via SVD instead (see Architecture above).

## Training data formats

### Skip-gram pairs

The `DatasetBuilder.build_skipgram_pairs()` method generates pairs of token IDs:

```python
from src.dataset_builder import DatasetBuilder

builder = DatasetBuilder()
pairs = builder.build_skipgram_pairs(token_ids, window_size=2)
```

## Saved artifacts

`Pipeline.run()` writes preprocessing artifacts to `outputs/processed/` (it does not train a model):

- `clean_text.txt`: preprocessed text
- `vocabulary.json`: vocabulary metadata and counts
- `word_to_id.json`: word → ID mappings
- `id_to_word.json`: ID → word mappings
- `token_ids.npy`: encoded token IDs
- `skipgram_pairs.json`: skip-gram training pairs

`train.py` writes trained embeddings to `models/sherlock_embeddings.txt` (one token and its vector per line).

## Usage

Train embeddings on the Sherlock corpus and save them to `models/`:

```bash
cd /Users/corleone/Projects/mini-word2vec
python train.py
```

Or call the trainer directly from Python:

```python
from src.trainer import train_word2vec

model, vocabulary = train_word2vec(
    "data/raw/sherlock.txt",
    embedding_dim=32,
    window_size=2,
    min_count=3,
    max_vocab=5000,
)
```

To run only the preprocessing pipeline (clean text, vocabulary, skip-gram pairs) without training:

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

## Finding similar words

Once you have a trained model and vocabulary, `most_similar` ranks vocabulary words by cosine similarity:

```python
from src.inference import most_similar

most_similar(model, vocabulary, "holmes", top_k=5)
```

## Testing

Run all tests with:

```bash
PYTHONPATH=. .venv/bin/pytest -q tests
```

## GitHub

This repository is initialized with Git and ready to push to GitHub as `pankajkr56070/mini-word2vec`.
