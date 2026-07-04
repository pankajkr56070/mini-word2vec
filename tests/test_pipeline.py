from src.dataset import build_skipgram_pairs
from src.model import TinyWord2Vec
from src.config import PipelineConfig
from src.pipeline import Pipeline
from src.preprocessor import load_corpus, preprocess_text, save_cleaned_text
from src.tokenizer import Tokenizer
from src.trainer import train_word2vec
from src.vocabulary import Vocabulary


def test_tokenize_text_and_vocabulary():
    text = "hello world hello there"
    tokens = Tokenizer().tokenize(text)
    assert tokens == ["hello", "world", "hello", "there"]

    vocab = Vocabulary(min_count=1)
    vocab.build(tokens)

    assert vocab.size >= 4
    assert vocab.encode(["hello", "missing"])[0] >= 0
    assert vocab.encode(["missing"])[0] == vocab.unk_index


def test_vocabulary_builds_round_trips_and_persists(tmp_path):
    vocab = Vocabulary(min_count=2, max_vocab=3)
    vocab.build(["hello", "hello", "world", "world", "there"])

    assert vocab.size == 3
    assert vocab.contains("hello")
    assert vocab.word_to_id("hello") == 1
    assert vocab.id_to_word(1) == "hello"
    assert vocab.encode(["hello", "missing", "world"]) == [1, 0, 2]
    assert vocab.decode([1, 0, 2]) == ["hello", vocab.unk_token, "world"]

    save_path = tmp_path / "vocab.json"
    vocab.save(save_path)
    reloaded = Vocabulary.load(save_path)

    assert reloaded.word_to_id("hello") == 1
    assert reloaded.decode([1, 0, 2]) == ["hello", reloaded.unk_token, "world"]


def test_pipeline_runs_and_writes_artifacts(tmp_path):
    input_path = tmp_path / "sample.txt"
    input_path.write_text("Hello world hello there\n", encoding="utf-8")
    output_dir = tmp_path / "processed"

    pipeline = Pipeline(PipelineConfig(min_count=1, max_vocab=10))
    result = pipeline.run(input_path=input_path, output_dir=output_dir)

    assert result["vocabulary"].size >= 3
    assert result["token_ids"] == result["vocabulary"].encode(["hello", "world", "hello", "there"])
    assert (output_dir / "processed" / "clean_text.txt").exists()
    assert (output_dir / "processed" / "vocabulary.json").exists()
    assert (output_dir / "processed" / "token_ids.npy").exists()


def test_build_skipgram_pairs():
    tokens = ["the", "cat", "sat", "on", "the", "mat"]
    pairs = build_skipgram_pairs(tokens, window_size=1)

    assert len(pairs) > 0
    assert ("cat", "sat") in pairs
    assert ("sat", "cat") in pairs


def test_training_smoke():
    corpus = "one two three one two"
    model = train_word2vec(
        corpus,
        embedding_dim=8,
        epochs=2,
        window_size=1,
        min_count=1,
        max_vocab=10,
        learning_rate=0.1,
    )

    assert isinstance(model, TinyWord2Vec)
    assert model.vector_size == 8
    assert model.get_vector("one").shape == (8,)


def test_load_corpus_reads_text_file(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha beta\nalpha", encoding="utf-8")

    tokens = load_corpus(path)

    assert tokens == ["alpha", "beta", "alpha"]


def test_preprocess_text_cleans_gutenberg_text(tmp_path):
    raw_text = """*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***
Hello, World!
This is a test.
*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***"""

    cleaned = preprocess_text(raw_text)
    assert "hello" in cleaned
    assert "world" in cleaned
    assert "project" not in cleaned
    assert "gutenberg" not in cleaned
    assert "  " not in cleaned

    output_path = tmp_path / "cleaned.txt"
    save_cleaned_text(cleaned, output_path)
    assert output_path.read_text(encoding="utf-8") == cleaned
