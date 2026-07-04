from typing import List, Tuple

from .config import WINDOW_SIZE


def build_skipgram_pairs(tokens: List[str], window_size: int = WINDOW_SIZE) -> List[Tuple[str, str]]:
    """Create center-context pairs from token sequences."""
    if window_size <= 0:
        return []

    pairs: List[Tuple[str, str]] = []
    for center_index, center_token in enumerate(tokens):
        for offset in range(1, window_size + 1):
            left_index = center_index - offset
            right_index = center_index + offset
            if left_index >= 0:
                pairs.append((center_token, tokens[left_index]))
            if right_index < len(tokens):
                pairs.append((center_token, tokens[right_index]))
    return pairs
