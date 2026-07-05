import numpy as np


class CrossEntropyLoss:
    """Evaluate a target token against a vector of model logits."""

    def softmax(self, logits: np.ndarray) -> np.ndarray:
        """Convert logits into probabilities."""
        logits = self._validate_logits(logits)
        shifted = logits - np.max(logits)
        exponentials = np.exp(shifted)
        return exponentials / np.sum(exponentials)

    def forward(self, logits: np.ndarray, target: int) -> float:
        """Compute cross-entropy loss for one target token ID."""
        logits = self._validate_logits(logits)
        target = self._validate_target(target, logits.size)

        # log-sum-exp form avoids taking log(0) for very unlikely targets.
        shifted = logits - np.max(logits)
        return float(np.log(np.sum(np.exp(shifted))) - shifted[target])

    @staticmethod
    def _validate_logits(logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float32)
        if logits.ndim != 1 or logits.size == 0:
            raise ValueError("logits must be a non-empty 1D array")
        if not np.all(np.isfinite(logits)):
            raise ValueError("logits must contain only finite values")
        return logits

    @staticmethod
    def _validate_target(target: int, vocab_size: int) -> int:
        if not isinstance(target, (int, np.integer)) or isinstance(target, bool):
            raise TypeError("target must be an integer")
        target = int(target)
        if target < 0 or target >= vocab_size:
            raise IndexError(f"Invalid target id: {target}")
        return target
