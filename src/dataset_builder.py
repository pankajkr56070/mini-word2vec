from pathlib import Path
from typing import List, Union

import numpy as np


class DatasetBuilder:
    def build(self, token_ids: List[int], output_path: Union[str, Path]) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, np.asarray(token_ids, dtype=np.int64))
