from src.dataset_builder import DatasetBuilder, SkipGramPair


def test_dataset_builder_builds_skipgram_pairs() -> None:
    builder = DatasetBuilder()
    pairs = builder.build_skipgram_pairs([0, 1, 2, 3], window_size=1)

    assert pairs == [
        SkipGramPair(center=0, context=1),
        SkipGramPair(center=1, context=0),
        SkipGramPair(center=1, context=2),
        SkipGramPair(center=2, context=1),
        SkipGramPair(center=2, context=3),
        SkipGramPair(center=3, context=2),
    ]


def test_dataset_builder_builds_cbow_samples() -> None:
    builder = DatasetBuilder()
    samples = builder.build_cbow_samples([0, 1, 2, 3], window_size=1)

    assert samples == []
