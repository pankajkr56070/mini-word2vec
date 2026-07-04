from typing import List, Tuple

import numpy as np


def most_similar(model, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
    target = model.get_vector(word)
    similarities: List[Tuple[str, float]] = []
    for idx, token in enumerate(model.vocabulary.index_to_token):
        if token == model.vocabulary.unk_token:
            continue
        vector = model.get_vector(token)
        similarity = float(np.dot(target, vector) / (np.linalg.norm(target) * np.linalg.norm(vector) + 1e-12))
        similarities.append((token, similarity))

    similarities.sort(key=lambda item: item[1], reverse=True)
    return similarities[:top_k]
