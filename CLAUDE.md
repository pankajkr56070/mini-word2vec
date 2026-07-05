# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the full test suite
PYTHONPATH=. .venv/bin/pytest -q tests

# Run a single test file / test
PYTHONPATH=. .venv/bin/pytest -q tests/test_losses.py
PYTHONPATH=. .venv/bin/pytest -q tests/test_losses.py::test_softmax_is_stable_for_large_logits

# Verify sources compile (what CI runs before tests)
python -m compileall -q src train.py

# Train embeddings on the Sherlock corpus (entry point)
python train.py
```

CI (`.github/workflows/ci.yml`) runs on Python 3.9 and 3.12: `compileall` then `pytest`, with only `numpy`
and `pytest` installed — do not introduce dependencies beyond these without updating CI.

## Architecture

Data flows through a fixed pipeline of small, single-responsibility classes in `src/`:

```
Preprocessor -> Tokenizer -> Vocabulary -> DatasetBuilder -> (EmbeddingLayer / Word2VecModel)
```

- **Preprocessor** (`preprocessor.py`): strips Project Gutenberg header/footer, lowercases, strips
  punctuation. `load_corpus()` also caches the cleaned text next to the source as `<name>.cleaned`.
- **Tokenizer** (`tokenizer.py`): trivial whitespace split/join.
- **Vocabulary** (`vocabulary.py`): builds token<->ID maps, ordered by descending frequency (ties broken
  by first occurrence). Index `0` is always reserved for `<UNK>`; `min_count`/`max_vocab` control pruning.
- **DatasetBuilder** (`dataset_builder.py`): generates skip-gram `(center, context)` ID pairs within a
  window. `build_cbow_samples` is a stub that always returns `[]` — not implemented.
- **EmbeddingLayer** (`embeddings.py`): a plain `(vocab_size, embedding_dim)` numpy matrix with
  lookup/save/load. `save_embeddings()` writes a word2vec-style plaintext file (`token v1 v2 ...` per line),
  distinct from `EmbeddingLayer.save()`'s `.npy` format.
- **Word2VecModel** (`model.py`): holds *two* separate `EmbeddingLayer`s (`input_embeddings`,
  `output_embeddings`). `forward`/`forward_batch` compute raw logits (`output_matrix @ center_vector`) —
  there is no backprop path wired to these forward methods.
- **CrossEntropyLoss** (`losses.py`): numerically-stable softmax/forward over a single logits vector.
  Currently only exercised by tests — `trainer.py` does not call into it.

Two independent orchestration paths exist, and it's easy to conflate them:

- **`Pipeline`** (`pipeline.py`): preprocesses a raw text file end-to-end and writes intermediate
  artifacts to `outputs/processed/` (`clean_text.txt`, `vocabulary.json`, `word_to_id.json`,
  `id_to_word.json`, `token_ids.npy`, `skipgram_pairs.json`). It does **not** train a model.
- **`train_word2vec`** (`trainer.py`): the actual model-producing path. It re-derives vocab/skip-gram
  pairs itself (does not use `Pipeline`), then fits embeddings via **SVD of the center/context
  co-occurrence matrix** (`_fit_embeddings_from_pairs`) — not gradient descent — and copies the result
  into both `input_embeddings` and `output_embeddings` of a `Word2VecModel`. `train.py` calls this, then
  writes the plaintext embeddings to `models/`.

When changing training behavior, be clear about which of the two paths (`Pipeline` artifacts vs.
`trainer.train_word2vec` model output) you're touching — they don't share state.

Tests in `tests/` mirror `src/` modules 1:1 (`test_<module>.py`); keep new tests colocated the same way.
