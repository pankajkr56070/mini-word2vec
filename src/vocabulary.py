import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


class Vocabulary:
    def __init__(self, min_count: int = 1, max_vocab: Optional[int] = None):
        self.min_count = min_count
        self.max_vocab = max_vocab
        self.unk_token = "<UNK>"
        self.unk_index = 0
        self.token_to_index: Dict[str, int] = {self.unk_token: self.unk_index}
        self.index_to_token: List[str] = [self.unk_token]
        self.counts: Counter = Counter()

    @property
    def size(self) -> int:
        return len(self.index_to_token)

    @property
    def statistics(self) -> Dict[str, int]:
        return {
            "size": self.size,
            "total_tokens": sum(self.counts.values()),
            "unique_tokens": len(self.counts),
        }

    def build(self, tokens: List[str]) -> None:
        self.counts = Counter(tokens)
        filtered = {token for token, count in self.counts.items() if count >= self.min_count}

        first_seen: Dict[str, int] = {}
        for index, token in enumerate(tokens):
            if token in filtered and token not in first_seen:
                first_seen[token] = index

        ordered = sorted(
            filtered,
            key=lambda token: (-self.counts[token], first_seen[token]),
        )
        if self.max_vocab is not None:
            ordered = ordered[: self.max_vocab]

        self.token_to_index = {self.unk_token: self.unk_index}
        self.index_to_token = [self.unk_token]
        for token in ordered:
            self.token_to_index[token] = len(self.index_to_token)
            self.index_to_token.append(token)

    def fit(self, tokens: List[str]) -> None:
        self.build(tokens)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.word_to_id(token) for token in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.id_to_word(index) for index in ids]

    def contains(self, token: str) -> bool:
        return token in self.token_to_index

    def word_to_id(self, token: str) -> int:
        return self.token_to_index.get(token, self.unk_index)

    def id_to_word(self, index: int) -> str:
        if 0 <= index < len(self.index_to_token):
            return self.index_to_token[index]
        return self.unk_token

    def frequency(self, token: str) -> int:
        return self.counts.get(token, 0)

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "min_count": self.min_count,
            "max_vocab": self.max_vocab,
            "unk_token": self.unk_token,
            "unk_index": self.unk_index,
            "token_to_index": self.token_to_index,
            "index_to_token": self.index_to_token,
            "counts": dict(self.counts),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def save_mappings(self, output_dir) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        (output_dir / "vocabulary.json").write_text(
            json.dumps(
                {
                    "min_count": self.min_count,
                    "max_vocab": self.max_vocab,
                    "unk_token": self.unk_token,
                    "unk_index": self.unk_index,
                    "token_to_index": self.token_to_index,
                    "index_to_token": self.index_to_token,
                    "counts": dict(self.counts),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (output_dir / "word_to_id.json").write_text(
            json.dumps(self.token_to_index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "id_to_word.json").write_text(
            json.dumps(self.index_to_token, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        vocab = cls(
            min_count=payload.get("min_count", 1),
            max_vocab=payload.get("max_vocab"),
        )
        vocab.unk_token = payload.get("unk_token", vocab.unk_token)
        vocab.unk_index = payload.get("unk_index", vocab.unk_index)
        vocab.token_to_index = payload.get("token_to_index", {vocab.unk_token: vocab.unk_index})
        vocab.index_to_token = payload.get("index_to_token", [vocab.unk_token])
        vocab.counts = Counter(payload.get("counts", {}))
        return vocab

    def to_index(self, token: str) -> int:
        return self.word_to_id(token)

    def to_token(self, index: int) -> str:
        return self.id_to_word(index)

    def __contains__(self, token: str) -> bool:
        return self.contains(token)
