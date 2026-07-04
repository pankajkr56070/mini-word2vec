from src.vocabulary import Vocabulary


def test_vocabulary_builds_ids_and_round_trips(tmp_path) -> None:
    vocab = Vocabulary(min_count=1, max_vocab=2)
    vocab.build(["hello", "hello", "world"])

    assert vocab.size == 3
    assert vocab.word_to_id("hello") == 1
    assert vocab.encode(["hello", "missing", "world"]) == [1, 0, 2]
    assert vocab.decode([1, 0, 2]) == ["hello", vocab.unk_token, "world"]

    save_path = tmp_path / "vocab.json"
    vocab.save(save_path)
    reloaded = Vocabulary.load(save_path)

    assert reloaded.word_to_id("world") == 2
    assert reloaded.decode([2]) == ["world"]
