from src.preprocessor import Preprocessor, load_corpus, preprocess_text, save_cleaned_text


def test_preprocess_text_removes_gutenberg_markers_and_punctuation() -> None:
    raw_text = """*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***
Hello, World!
This is a test.
*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***"""

    cleaned = preprocess_text(raw_text)

    assert cleaned == "hello world this is a test"
    assert "gutenberg" not in cleaned
    assert "project" not in cleaned


def test_preprocessor_saves_cleaned_text(tmp_path) -> None:
    preprocessor = Preprocessor()
    output_path = tmp_path / "cleaned.txt"

    preprocessor.save("hello world", output_path)

    assert output_path.read_text(encoding="utf-8") == "hello world"


def test_load_corpus_reads_and_writes_cleaned_file(tmp_path) -> None:
    input_path = tmp_path / "sample.txt"
    input_path.write_text("Alpha beta\nAlpha", encoding="utf-8")

    tokens = load_corpus(input_path)

    assert tokens == ["alpha", "beta", "alpha"]
    assert input_path.with_suffix(input_path.suffix + ".cleaned").exists()
