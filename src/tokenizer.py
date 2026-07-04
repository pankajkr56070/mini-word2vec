from typing import List


class Tokenizer:
    def tokenize(self, text: str) -> List[str]:
        """Convert cleaned text into a sequence of tokens."""
        if not text:
            return []
        return text.split()

    def detokenize(self, tokens: List[str]) -> str:
        """Join tokens back into a single string."""
        return " ".join(tokens)
