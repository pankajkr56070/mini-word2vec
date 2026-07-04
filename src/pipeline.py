from pathlib import Path
from typing import Optional, Union

from .config import PipelineConfig
from .dataset_builder import DatasetBuilder
from .preprocessor import Preprocessor
from .tokenizer import Tokenizer
from .vocabulary import Vocabulary


class Pipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.preprocessor = Preprocessor()
        self.tokenizer = Tokenizer()
        self.vocabulary = Vocabulary(min_count=self.config.min_count, max_vocab=self.config.max_vocab)
        self.dataset_builder = DatasetBuilder()

    def run(self, input_path: Union[str, Path], output_dir: Union[str, Path]):
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        processed_dir = output_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        text = input_path.read_text(encoding="utf-8")
        clean_text = self.preprocessor.process(text)
        self.preprocessor.save(clean_text, processed_dir / "clean_text.txt")

        tokens = self.tokenizer.tokenize(clean_text)
        self.vocabulary.build(tokens)
        token_ids = self.vocabulary.encode(tokens)

        self.vocabulary.save_mappings(processed_dir)
        self.dataset_builder.build(token_ids, processed_dir / "token_ids.npy")

        return {"vocabulary": self.vocabulary, "token_ids": token_ids, "clean_text": clean_text}
