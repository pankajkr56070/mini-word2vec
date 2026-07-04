import re
from pathlib import Path
from typing import List, Union

from .tokenizer import Tokenizer


class Preprocessor:
    def process(self, text: str) -> str:
        return preprocess_text(text)

    def save(self, text: str, path: Union[str, Path]) -> None:
        save_cleaned_text(text, path)


def preprocess_text(text: str) -> str:
    """Clean text by removing Gutenberg markers, normalizing whitespace, lowercasing, and removing punctuation."""
    cleaned = _remove_gutenberg_header_footer(text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.lower()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _remove_gutenberg_header_footer(text: str) -> str:
    """Remove common Project Gutenberg header/footer markers."""
    lines = text.splitlines()
    start_index = 0
    end_index = len(lines)

    for index, line in enumerate(lines):
        stripped = line.strip().lower()
        if stripped.startswith("*** start of the project gutenberg") or stripped.startswith("start of the project gutenberg"):
            start_index = index + 1
            break

    for index in range(len(lines) - 1, -1, -1):
        stripped = lines[index].strip().lower()
        if stripped.startswith("*** end of the project gutenberg") or stripped.startswith("end of the project gutenberg"):
            end_index = index
            break

    return "\n".join(lines[start_index:end_index]).strip()


def save_cleaned_text(text: str, path: Union[str, Path]) -> None:
    """Save cleaned text to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_corpus(path: Union[str, Path]) -> List[str]:
    """Load tokens from a text file after cleaning."""
    path = Path(path)
    if not path.exists():
        return Tokenizer().tokenize(str(path))

    text = path.read_text(encoding="utf-8")
    cleaned = preprocess_text(text)
    save_cleaned_text(cleaned, path.with_suffix(path.suffix + ".cleaned"))
    return Tokenizer().tokenize(cleaned)
