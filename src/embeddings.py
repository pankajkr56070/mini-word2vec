from pathlib import Path
from typing import Union


def save_embeddings(model, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, token in enumerate(model.vocabulary.index_to_token):
            vector = model.get_vector(token)
            values = " ".join(str(value) for value in vector)
            handle.write(f"{token} {values}\n")
