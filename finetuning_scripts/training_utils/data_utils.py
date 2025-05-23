from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset

RANDOM_SEED = 4213

if TYPE_CHECKING:
    import pandas as pd


class TabularDataset(Dataset):
    """Tabular dataset.

    This class is used to load tabular data.

    Here one sample is equal to one split of the data.

    Arguments:
    ----------
    X_train: torch.Tensor (n_samples, n_features)
        Input features.
    y_train: torch.Tensor (n_samples, 1)
        Target labels.
    max_steps: int
        Maximum number of steps (splits of the data).
    stratify_split: bool
        Whether the task is classification or regression.
    cross_val_splits: int
        Number of cross-validation splits.
    """

    def __init__(
        self,
        *,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        max_steps: int,
        is_classification: bool,
        cross_val_splits: int | None = 10,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.max_steps = max_steps
        self.cross_val_splits = cross_val_splits
        self.is_classification = is_classification
        self._splits_generator = self.splits_generator(
            X_train=X_train,
            y_train=y_train,
            cross_val_splits=cross_val_splits,
            stratify_split=is_classification,
            seed=RANDOM_SEED,
        )
        self._rng = np.random.RandomState(RANDOM_SEED)

    @staticmethod
    def splits_generator(
        *,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        cross_val_splits: int,
        stratify_split: bool,
        seed: int,
    ):
        """Endless generator for splits to perform repeated cross-validation."""
        rng = np.random.RandomState(seed)
        splitter = StratifiedKFold if stratify_split else KFold

        while True:
            splits = splitter(
                n_splits=cross_val_splits,
                random_state=rng.random_integers(0, int(np.iinfo(np.int32).max)),
                shuffle=True,
            ).split(
                X=X_train,
                y=y_train.cpu().detach().numpy() if stratify_split else None,
            )
            yield from splits

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_splits_generator"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._splits_generator = self.splits_generator(
            X_train=self.X_train,
            y_train=self.y_train,
            cross_val_splits=self.cross_val_splits,
            stratify_split=self.is_classification,
            seed=RANDOM_SEED,
        )

    def __len__(self):
        return self.max_steps

    def get_splits(self) -> tuple[np.ndarray, np.ndarray]:
        """Get train and test indices for next batch."""
        train_idx, test_idx = next(self._splits_generator)
        return train_idx, test_idx

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        train_idx, test_idx = self.get_splits()

        # Correct for equal batch size
        expected_test_size = len(self.X_train) // self.cross_val_splits
        if len(test_idx) != expected_test_size:
            train_idx = np.concatenate(
                [train_idx, test_idx[: len(test_idx) - expected_test_size]],
            )
            test_idx = test_idx[len(test_idx) - expected_test_size :]

        return dict(
            X_train=self.X_train[train_idx],
            X_test=self.X_train[test_idx],
            y_train=self.y_train[train_idx],
            y_test=self.y_train[test_idx],
        )


def get_data_loader(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_steps: int,
    torch_rng: torch.Generator,
    batch_size: int,
    is_classification: bool,
    num_workers: int,
) -> DataLoader:
    """Get data loader.

    This function is used to get data loader.

    Arguments:
    ----------
    X_train: pd.DataFrame
        Input features.
    y_train: pd.Series
        Target labels.
    max_steps: int
        Maximum number of steps (splits of the data).
    torch_rng: torch.Generator
        Torch random number generator for splits and similar.
    batch_size: int
        Batch size. How many splits to load at a time.
    is_classification: bool
        Whether the task is classification or regression.
    num_workers: int
        Number of workers for data loader.

    Returns:
    --------
    DataLoader
        Data loader.
    """
    X_train = torch.tensor(X_train.copy().values).float()
    y_train = torch.tensor(y_train.copy().values).reshape(-1, 1).float()
    dataset = TabularDataset(
        X_train=X_train,
        y_train=y_train,
        max_steps=max_steps * batch_size,
        is_classification=is_classification,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
        generator=torch_rng,
        persistent_workers=False,
    )
