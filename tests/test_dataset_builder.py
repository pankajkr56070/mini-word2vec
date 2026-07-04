import numpy as np

from src.dataset_builder import DatasetBuilder


def test_dataset_builder_saves_token_ids_as_numpy_array(tmp_path) -> None:
    builder = DatasetBuilder()
    output_path = tmp_path / "token_ids.npy"

    builder.build([1, 2, 3], output_path)

    saved = np.load(output_path)
    assert saved.tolist() == [1, 2, 3]
