from src.tokenizer import Tokenizer


def test_tokenizer_splits_text_into_tokens() -> None:
    tokenizer = Tokenizer()

    assert tokenizer.tokenize("hello world  again") == ["hello", "world", "again"]


def test_tokenizer_detokenizes_tokens() -> None:
    tokenizer = Tokenizer()

    assert tokenizer.detokenize(["hello", "world"]) == "hello world"
