from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from finetuning_scripts.constant_utils import SupportedDevice, TaskType
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from finetuning_scripts.metric_utils.ag_metrics import Scorer
    from tabpfn.model.transformer import PerFeatureTransformer
    from tabpfn import TabPFNClassifier, TabPFNRegressor


def create_val_data(
    *,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    rng: np.random.RandomState,
    n_samples: int,
    is_classification: bool,
) -> tuple[
    pd.DataFrame | np.ndarray,
    pd.DataFrame | np.ndarray,
    pd.Series | np.ndarray,
    pd.Series | np.ndarray,
]:
    # Split data ourselves
    if n_samples < 10000:
        test_size = 0.33
    elif n_samples < 500000:
        test_size = 0.2
    elif n_samples < 1000000:
        test_size = 0.1
    else:
        test_size = 0.05

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=test_size,
        random_state=rng,
        stratify=y_train if is_classification else None,
    )
    return X_train, X_val, y_train, y_val


def validate_tabpfn(
    *,
    X_train: torch.Tensor,  # (n_samples, batch_size, n_features)
    y_train: torch.Tensor,  # (n_samples, batch_size, 1)
    X_val: torch.Tensor,  # (n_samples, batch_size, 1)
    y_val: torch.Tensor,  # (n_samples, batch_size, 1)
    validation_metric: Scorer,
    model: PerFeatureTransformer,
    model_forward_fn: Callable,
    task_type: TaskType,
    device: SupportedDevice,
    use_sklearn_interface_for_validation: bool = False,
    model_for_validation: TabPFNClassifier | TabPFNRegressor = None,
) -> float:
    """Validate the TabPFN model and return a loss (lower is better).

    This code assumes that batch_size for validation is 1. Otherwise,
    need to write a loop, I guess?
    """
    if use_sklearn_interface_for_validation:
        if model_for_validation is None:
            raise ValueError(
                f"Model for validation is required when validating with full TabPFN preprocessing.")
        if not model_for_validation.fit_mode == 'fit_preprocessors':
            raise ValueError(
                f"fit_mode for model_for_validation must be 'fit_preprocessors' when validating with full TabPFN preprocessing.")
        if model_for_validation.memory_saving_mode:
            raise ValueError(
                f"memory_saving_mode for model_for_validation must be False when validating with full TabPFN preprocessing.")

        estimator_type = model_for_validation.__sklearn_tags__().estimator_type
        if task_type == TaskType.REGRESSION and not estimator_type == 'regressor':
            raise ValueError(
                f"model_for_validation must be TabPFNRegressor but is: {type(model_for_validation)}")
        if task_type in {TaskType.MULTICLASS_CLASSIFICATION,
                           TaskType.BINARY_CLASSIFICATION} and not estimator_type == 'classifier':
            raise ValueError(
                f"model_for_validation must be TabPFNClassifier but is: {type(model_for_validation)}")

        from tabpfn import TabPFNClassifier, TabPFNRegressor

        X_val = X_val.cpu().detach().numpy()[:, 0, :]
        y_true = y_val.flatten().cpu().detach().numpy()

        if not hasattr(model_for_validation, 'executor_'):
            X_train = X_train.cpu().detach().numpy()[:, 0, :]
            y_train = y_train.flatten().cpu().detach().numpy()
            model_for_validation.fit(X_train, y_train)

        model_for_validation.model_ = model
        model_for_validation.executor_.model = model

        if task_type == TaskType.REGRESSION:
            y_pred = model_for_validation.predict(X_val, output_type="mean")
        else:
            y_pred = model_for_validation.predict_proba(X_val)

        # model is moved to cpu after inference by the TabPFN* models
        model.to(device)
    else:
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        X_val = X_val.to(device)
        y_val = y_val.to(device)

        pred_logits = model_forward_fn(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_val,
            forward_for_validation=True,
        )

        match task_type:
            case TaskType.REGRESSION:
                y_pred = pred_logits.float().flatten().cpu().detach().numpy()
                y_true = y_val.float().flatten().cpu().detach().numpy()
            case TaskType.BINARY_CLASSIFICATION:
                # TODO: check that this works / is exhaustive.
                if validation_metric.needs_threshold or validation_metric.needs_proba:
                    y_pred = (
                        torch.nn.functional.sigmoid(pred_logits[:, 0, 1])
                        .cpu()
                        .detach()
                        .numpy()
                    )
                else:
                    # Required to get the correct classes for the metrics
                    y_pred = (
                        torch.nn.functional.softmax(pred_logits[:, 0, :],
                                                    dim=-1)
                        .cpu()
                        .detach()
                        .numpy()
                    )
                y_true = y_val.long().flatten().cpu().detach().numpy()
            case TaskType.MULTICLASS_CLASSIFICATION:
                y_pred = (
                    torch.nn.functional.softmax(pred_logits[:, 0, :], dim=-1)
                    .cpu()
                    .detach()
                    .numpy()
                )
                y_true = y_val.long().flatten().cpu().detach().numpy()
            case _:
                raise ValueError(f"Task type {task_type} not supported.")

        X_train.cpu()
        y_train.cpu()
        X_val.cpu()
        y_val.cpu()

    score = validation_metric(y_true=y_true, y_pred=y_pred)

    return validation_metric.convert_score_to_error(score=score)
