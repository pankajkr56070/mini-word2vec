import numpy as np
import pytest

from src.inference import Inference
from src.model import Word2VecModel
from src.vocabulary import Vocabulary


def build_inference():
    vocab = Vocabulary(min_count=1)
    vocab.build(["king", "queen", "man", "woman"])
    model = Word2VecModel(vocab_size=vocab.size, embedding_dim=2)

    vectors = {
        vocab.unk_token: [0.0, 0.0],
        "king": [1.0, 0.0],
        "queen": [0.9, 0.1],
        "man": [0.0, 1.0],
        "woman": [-1.0, 0.0],
    }
    for token, vector in vectors.items():
        model.input_embeddings.embeddings[vocab.to_index(token)] = vector

    return Inference(model, vocab), vocab


def test_embedding_returns_vector_for_known_word() -> None:
    inference, _ = build_inference()

    assert np.allclose(inference.embedding("king"), [1.0, 0.0])


def test_embedding_raises_for_unknown_word() -> None:
    inference, _ = build_inference()

    with pytest.raises(ValueError, match="Unknown word"):
        inference.embedding("dragon")


def test_embedding_returns_a_copy_not_a_reference() -> None:
    inference, _ = build_inference()

    vector = inference.embedding("king")
    vector[:] = 0.0

    assert np.allclose(inference.embedding("king"), [1.0, 0.0])


def test_similarity_of_identical_vectors_is_one() -> None:
    inference, _ = build_inference()

    assert inference.similarity("king", "king") == pytest.approx(1.0)


def test_similarity_of_orthogonal_vectors_is_zero() -> None:
    inference, _ = build_inference()

    assert inference.similarity("king", "man") == pytest.approx(0.0, abs=1e-6)


def test_similarity_of_opposite_vectors_is_negative_one() -> None:
    inference, _ = build_inference()

    assert inference.similarity("king", "woman") == pytest.approx(-1.0)


def test_most_similar_excludes_query_word() -> None:
    inference, _ = build_inference()

    results = inference.most_similar("king", top_k=3)

    assert "king" not in [token for token, _ in results]


def test_most_similar_ranks_by_cosine_similarity() -> None:
    inference, _ = build_inference()

    results = inference.most_similar("king", top_k=1)

    assert results[0][0] == "queen"


def test_nearest_excludes_unk_token() -> None:
    inference, vocab = build_inference()

    results = inference.nearest(np.array([1.0, 0.0]), top_k=10)

    assert vocab.unk_token not in [token for token, _ in results]


def test_nearest_returns_top_k_sorted_by_similarity_desc() -> None:
    inference, _ = build_inference()

    results = inference.nearest(np.array([1.0, 0.0]), top_k=2)

    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0][0] == "king"


def test_most_similar_rejects_non_positive_top_k() -> None:
    inference, _ = build_inference()

    with pytest.raises(ValueError, match="top_k must be positive"):
        inference.most_similar("king", top_k=0)


def test_nearest_rejects_non_positive_top_k() -> None:
    inference, _ = build_inference()

    with pytest.raises(ValueError, match="top_k must be positive"):
        inference.nearest(np.array([1.0, 0.0]), top_k=0)
