# mini-word2vec

A compact Word2Vec-style embedding project that reads raw text, preprocesses it, tokenizes it, builds a vocabulary, creates skip-gram pairs, and trains a small embedding layer via gradient descent.

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
- **Trainer**: orchestrates SGD training over a dataset of `(center, context)` pairs by calling `model.forward` → `criterion.forward` → `criterion.backward` → `model.backward` for each pair, once per epoch. It is model-agnostic — it knows nothing about embeddings, softmax, vocabulary, or tokenization, only the `forward`/`backward` interface
- **Pipeline**: orchestrates preprocessing (clean → tokenize → vocab → skip-gram pairs) and writes intermediate artifacts
- **`train_word2vec`**: a convenience wrapper that builds a vocab, skip-gram pairs, and a `Word2VecModel`, then hands them to a `Trainer` and runs it
- **Inference**: a small query API over a trained model + vocabulary — `embedding(word)`, `similarity(word1, word2)`, `most_similar(word, top_k)`, and `nearest(vector, top_k)`, all sharing one cosine-similarity implementation

`Pipeline` and `Trainer`/`train_word2vec` are independent: `Pipeline` only produces preprocessing artifacts (it does not train anything), while `train_word2vec` re-derives its own vocab/pairs and produces the trained `Word2VecModel`. `train.py` uses `train_word2vec` directly, not `Pipeline`.

Each training step updates the *entire* output embedding matrix for a single `(center, context)` pair (full softmax, no negative sampling) — on the full Sherlock corpus (vocab≈3100, ~424k pairs) this runs in about 3 minutes for 3 epochs; a much larger vocab or corpus would need negative sampling or hierarchical softmax to stay fast.

## What it does

Preprocessing (`Pipeline`):

1. Reads a corpus from `data/raw/sherlock.txt`
2. Cleans and normalizes the text
3. Tokenizes the cleaned text
4. Builds a vocabulary and token-ID mappings
5. Generates skip-gram training pairs

Training (`train_word2vec`, used by `train.py`):

1. Repeats steps 2-5 above independently of `Pipeline`
2. Builds a `Word2VecModel` and a `Trainer` over the skip-gram pairs
3. Runs `Trainer.train(epochs)`: for each epoch, for each `(center, context)` pair, does `forward` → loss → `backward` → weight update, printing `Epoch i/N  Loss = ...`
4. Saves the resulting embeddings to `models/`

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

## Trainer

`Trainer` orchestrates the SGD loop over a list of `SkipGramPair`s. It never computes embeddings itself — the model owns its parameters; `Trainer` just calls `forward`/`backward`:

```python
from src.model import Word2VecModel
from src.losses import CrossEntropyLoss
from src.trainer import Trainer

model = Word2VecModel(vocab_size=100, embedding_dim=32)
trainer = Trainer(model, CrossEntropyLoss(), skipgram_pairs, learning_rate=0.05)

trainer.train(epochs=5)
# Epoch 1/5
# Pairs processed : 183,452
# Average Loss    : 5.83
# Learning Rate   : 0.025

trainer.metrics  # {"steps_completed", "pairs_processed", "current_loss", "average_loss", "epoch_losses"}
```

`train_epoch()` is also public if you want finer control than `train(epochs)` (e.g. custom logging, or stopping early between epochs) — the per-pair `_train_step` is private, since a single gradient step isn't meant to be driven manually outside of `train`/`train_epoch`. Checkpointing (`save`/`load`/resume) isn't implemented yet. `Trainer(...)` validates eagerly: it rejects a `None` model/criterion, an empty dataset, and a non-positive `learning_rate`; `train(epochs)` rejects a non-positive `epochs`.

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

`Inference` wraps a trained model and vocabulary with a small, composable query API — `embedding`/`similarity`/`most_similar` are all built on top of `nearest`, and there's no duplicated cosine-similarity math:

```python
from src.inference import Inference

inference = Inference(model, vocabulary)

inference.embedding("holmes")               # -> np.ndarray (a copy — safe to mutate)
inference.similarity("holmes", "watson")    # -> float
inference.most_similar("holmes", top_k=5)   # -> List[Tuple[str, float]], excludes "holmes" itself
inference.nearest(some_vector, top_k=5)     # -> nearest vocabulary words to an arbitrary vector
```

`embedding(word)` raises `ValueError` for a word not in the vocabulary rather than silently falling back to `<UNK>`'s vector, and returns a defensive copy so callers can't mutate the model's embeddings in place. `most_similar`/`nearest` both reject a non-positive `top_k`.

Two things noted for later, not done now: `nearest` recomputes every candidate's norm on each call (fine at this vocab size; a cache of normalized vectors would speed up repeated queries at larger scale), and `nearest`/`most_similar` work in terms of words rather than raw indices internally (revisit if/when analogy search or sentence embeddings need the index-level primitive).

`analogy(positive, negative, top_k=5)` (e.g. "king − man + woman ≈ queen") is deliberately not implemented yet — once `embedding` and `nearest` exist, it's a thin composition of the two rather than special-cased logic.

## Testing

Run all tests with:

```bash
PYTHONPATH=. .venv/bin/pytest -q tests
```

## GitHub

This repository is initialized with Git and ready to push to GitHub as `pankajkr56070/mini-word2vec`.
