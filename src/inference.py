from typing import List, Tuple

import numpy as np


def most_similar(model, vocabulary, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
    target = model.input_embeddings.lookup(vocabulary.to_index(word))
    similarities: List[Tuple[str, float]] = []
    for idx, token in enumerate(vocabulary.index_to_token):
        if token == vocabulary.unk_token:
            continue
        vector = model.input_embeddings.lookup(idx)
        similarity = float(np.dot(target, vector) / (np.linalg.norm(target) * np.linalg.norm(vector) + 1e-12))
        similarities.append((token, similarity))

    similarities.sort(key=lambda item: item[1], reverse=True)
    return similarities[:top_k]
