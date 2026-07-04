from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SkipGramPair:
    center: int
    context: int


class DatasetBuilder:
    def build_skipgram_pairs(self, token_ids: List[int], window_size: int) -> List[SkipGramPair]:
        if window_size <= 0:
            return []

        training_pairs: List[SkipGramPair] = []
        for center_index, center_id in enumerate(token_ids):
            for offset in range(1, window_size + 1):
                left_index = center_index - offset
                right_index = center_index + offset
                if left_index >= 0:
                    training_pairs.append(SkipGramPair(center=center_id, context=token_ids[left_index]))
                if right_index < len(token_ids):
                    training_pairs.append(SkipGramPair(center=center_id, context=token_ids[right_index]))
        return training_pairs

    def build_cbow_samples(self, token_ids: List[int], window_size: int) -> List[Tuple[List[int], int]]:
        if window_size <= 0:
            return []

        samples: List[Tuple[List[int], int]] = []
        return samples
