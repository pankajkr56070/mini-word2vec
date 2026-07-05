import pytest

from src.dataset_builder import SkipGramPair
from src.losses import CrossEntropyLoss
from src.model import Word2VecModel
from src.trainer import Trainer


def build_dataset() -> list:
    return [
        SkipGramPair(center=0, context=1),
        SkipGramPair(center=1, context=0),
        SkipGramPair(center=2, context=3),
        SkipGramPair(center=3, context=2),
    ]


def build_trainer(dataset=None, learning_rate: float = 0.5) -> Trainer:
    model = Word2VecModel(vocab_size=8, embedding_dim=4)
    criterion = CrossEntropyLoss()
    return Trainer(model, criterion, dataset or build_dataset(), learning_rate)


def test_train_runs_one_step_per_pair_per_epoch() -> None:
    dataset = build_dataset()
    trainer = build_trainer(dataset)

    trainer.train(epochs=3)

    assert trainer.steps_completed == 3 * len(dataset)
    assert trainer.pairs_processed == 3 * len(dataset)
    assert len(trainer.epoch_losses) == 3


def test_train_tracks_current_and_average_loss() -> None:
    trainer = build_trainer()

    trainer.train(epochs=1)

    assert trainer.current_loss is not None
    assert trainer.average_loss() > 0
    assert trainer.current_loss > 0


def test_average_loss_is_zero_before_any_step() -> None:
    trainer = build_trainer()

    assert trainer.average_loss() == 0.0


def test_train_step_updates_counters_and_returns_loss() -> None:
    trainer = build_trainer()
    pair = SkipGramPair(center=0, context=1)

    loss_value = trainer._train_step(pair)

    assert loss_value > 0
    assert trainer.steps_completed == 1
    assert trainer.pairs_processed == 1
    assert trainer.current_loss == loss_value
    assert trainer.average_loss() == loss_value


def test_train_epoch_returns_average_loss_and_records_history() -> None:
    dataset = build_dataset()
    trainer = build_trainer(dataset)

    epoch_loss = trainer.train_epoch()

    assert trainer.steps_completed == len(dataset)
    assert trainer.epoch_losses == [epoch_loss]


def test_metrics_reflects_current_state() -> None:
    trainer = build_trainer()

    trainer.train(epochs=2)
    metrics = trainer.metrics

    assert metrics["steps_completed"] == trainer.steps_completed
    assert metrics["pairs_processed"] == trainer.pairs_processed
    assert metrics["current_loss"] == trainer.current_loss
    assert metrics["average_loss"] == trainer.average_loss()
    assert metrics["epoch_losses"] == trainer.epoch_losses


def test_train_reduces_average_epoch_loss_over_time() -> None:
    trainer = build_trainer(learning_rate=0.5)

    trainer.train(epochs=20)

    assert trainer.epoch_losses[-1] < trainer.epoch_losses[0]


def test_train_prints_epoch_progress(capsys) -> None:
    trainer = build_trainer()

    trainer.train(epochs=2)

    output = capsys.readouterr().out
    assert "Epoch 1/2" in output
    assert "Epoch 2/2" in output
    assert "Pairs processed" in output
    assert "Average Loss" in output
    assert "Learning Rate" in output


def test_trainer_does_not_mutate_dataset() -> None:
    dataset = build_dataset()
    trainer = build_trainer(dataset)

    trainer.train(epochs=2)

    assert trainer.dataset == dataset


def test_rejects_none_model() -> None:
    with pytest.raises(ValueError, match="model must not be None"):
        Trainer(None, CrossEntropyLoss(), build_dataset(), learning_rate=0.1)


def test_rejects_none_criterion() -> None:
    model = Word2VecModel(vocab_size=8, embedding_dim=4)

    with pytest.raises(ValueError, match="criterion must not be None"):
        Trainer(model, None, build_dataset(), learning_rate=0.1)


def test_rejects_empty_dataset() -> None:
    model = Word2VecModel(vocab_size=8, embedding_dim=4)

    with pytest.raises(ValueError, match="dataset must not be empty"):
        Trainer(model, CrossEntropyLoss(), [], learning_rate=0.1)


def test_rejects_non_positive_learning_rate() -> None:
    model = Word2VecModel(vocab_size=8, embedding_dim=4)

    with pytest.raises(ValueError, match="learning_rate must be positive"):
        Trainer(model, CrossEntropyLoss(), build_dataset(), learning_rate=0.0)


def test_train_rejects_non_positive_epochs() -> None:
    trainer = build_trainer()

    with pytest.raises(ValueError, match="epochs must be positive"):
        trainer.train(epochs=0)
