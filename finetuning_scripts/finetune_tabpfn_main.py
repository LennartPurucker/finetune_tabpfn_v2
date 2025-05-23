from __future__ import annotations

import logging
import random
import time
import warnings
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Union, Sequence

import numpy as np
import pandas as pd
import torch
import wandb

from finetuning_scripts.constant_utils import (
    SupportedDevice,
    SupportedValidationMetric,
    TaskType,
)
from finetuning_scripts.data_classes import FineTuneSetup, FineTuneStepResults
from finetuning_scripts.metric_utils.ag_metrics import get_metric
from finetuning_scripts.training_utils.ag_early_stopping import AdaptiveES
from finetuning_scripts.training_utils.data_utils import get_data_loader
from finetuning_scripts.training_utils.model_utils import save_model
from finetuning_scripts.training_utils.training_loss import compute_loss, get_loss
from finetuning_scripts.training_utils.validation_utils import validate_tabpfn
from schedulefree import AdamWScheduleFree
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.base import load_model_criterion_config
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.nn import DataParallel
from tqdm import tqdm

if TYPE_CHECKING:
    from tabpfn.model.transformer import PerFeatureTransformer
    from torch.nn.modules.loss import _Loss
    from torch.optim.optimizer import Optimizer

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*input value tensor is non-contiguous.*",
)


def fine_tune_tabpfn(
    *,
    path_to_base_model: Path | Literal["auto"] = "auto",
    save_path_to_fine_tuned_model: Path,
    # Finetuning HPs
    time_limit: int,
    finetuning_config: dict,
    validation_metric: SupportedValidationMetric,
    # Input Data
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    categorical_features_index: list[int] | None,
    task_type: TaskType,
    device: SupportedDevice,
    use_multiple_gpus: bool = False,
    multiple_device_ids: Sequence[Union[int, torch.device]] | None  = None,
    X_val: pd.DataFrame | np.ndarray | None = None,
    y_val: pd.Series | np.ndarray | None = None,
    random_seed: int = 42,
    # Other
    logger_level: int = 20,
    show_training_curve: bool = False,
    use_wandb: bool = False,
    use_sklearn_interface_for_validation: bool = False,
    model_for_validation: TabPFNClassifier | TabPFNRegressor = None
) -> None:
    """Fine-tune a TabPFN model.

    Run a simple fine-tuning loop for a TabPFN model on one dataset.
    Saves the best model based on the validation loss to disk under `save_path_to_fine_tuned_model`.

    Arguments:
    ----------
    time_limit: int
        The maximum time limit in seconds for the fine-tuning.
    finetuning_config: dict
        The configuration for the fine-tuning such as learning rate, batch size, etc.
        See _setup_tuning for possible learning HPs.
    X_train: pd.DataFrame | np.ndarray
        The training features.
    y_train: pd.Series | np.ndarray
        The training target.
    categorical_features_index: list[int] | None
        The indices of the categorical features.
    path_to_base_model: Path | Literal["auto"]
        Path to the base model that shall be fine-tuned. Same logic as for the TabPFN
        model loading.
    save_path_to_fine_tuned_model: Path
        Output path to save the fine-tuned model.
    validation_metric: SupportedValidationMetric
        The validation metric to use for early stopping and validation.
    task_type: TaskType
        The task type of the problem.
    device: SupportedDevice
        The device to use for fine-tuning.
    use_multiple_gpus: bool
        If True, will use multiple GPUs for fine-tuning.
    multiple_device_ids: Sequence[Union[int, torch.device]] | None
        GPU ids to use when use_multiple_gpus is True.
        Will use all available GPUs if None.
    X_val: pd.DataFrame | np.ndarray | None
        Optional validation features. If not provided, will be split from the training set.
    y_val: pd.Series | np.ndarray | None
        Optional validation target. If not provided, will be split from the training set.
    random_seed: int
        The random seed to control the randomness.
    logger_level: int
        The logger level to use for output during fine-tuning.
    show_training_curve: bool
        If True, show a training curve plot after fine-tuning.
    use_wandb: bool
        If True, log the fine-tuning process to Weights & Biases.
        Log in via the CLI if not already done: `wandb login`.
    use_sklearn_interface_for_validation: bool
        If True, will create and run TabPFN default sklearn preprocessing pipeline
        for validation metric calculation.
    model_for_validation: TabPFNClassifier | TabPFNRegressor
        Optional TabPFN model which will be used for validation if use_sklearn_interface_for_validation is True.
        The passed model should not be fitted, it is used to configure the
        preprocessing pipeline.
    """
    st_time = time.time()

    # Coerce input data into pandas structures
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)

    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)

    if X_val is not None and not isinstance(X_val, pd.DataFrame):
        X_val = pd.DataFrame(X_val)

    if y_val is not None and not isinstance(y_val, pd.Series):
        y_val = pd.Series(y_val)

    # Control logging
    logger.setLevel(logger_level)
    disable_progress_bar = logger_level >= 20

    # Control randomness
    rng = np.random.RandomState(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch_rng = torch.Generator()
    torch_rng.manual_seed(random_seed)

    # Meta
    is_classification = task_type != TaskType.REGRESSION
    use_autocast = False
    if device == SupportedDevice.GPU:
        # Autocast on CPU too slow for unsupported hardware + env: https://github.com/pytorch/pytorch/issues/118499
        use_autocast = True
    # If True, it is likely that the first ~5 steps will have NaNs and no change
    #   The code below compensates for this fact.
    use_grad_scaler = use_autocast

    # Load base model
    if isinstance(path_to_base_model, str) and path_to_base_model == "auto":
        model_path = None  # type: ignore
    else:
        model_path = path_to_base_model
    model, criterion, checkpoint_config = load_model_criterion_config(
        model_path=model_path,
        check_bar_distribution_criterion=False,
        cache_trainset_representation=False,
        which="classifier" if is_classification else "regressor",
        version="v2",
        download=True,
        model_seed=random_seed,
    )
    model.criterion = criterion
    checkpoint_config = checkpoint_config.__dict__
    is_data_parallel = False
    if device == 'cuda' and use_multiple_gpus and torch.cuda.device_count() > 1:
        model = DataParallel(model, device_ids=multiple_device_ids)
        is_data_parallel = True
    model.to(device)
    if use_wandb:
        wandb.watch(model, log_freq=1, log="all")

    # Setup validation
    create_val_data = (X_val is None) and (y_val is None)
    n_classes = len(np.unique(y_train)) if is_classification else None
    n_samples = len(X_train)
    if not create_val_data:
        n_samples += len(X_val)
    else:
        from finetuning_scripts.training_utils.validation_utils import create_val_data

        X_train, X_val, y_train, y_val = create_val_data(
            X_train=X_train,
            y_train=y_train,
            rng=rng,
            n_samples=n_samples,
            is_classification=is_classification,
        )
    val_report = f"""
    === Basic / Validation State ===
        \tTime Limit: {time_limit}
        \tEarly Stopping Metric: {validation_metric}
        \tVal Samples: {len(X_val) if X_val is not None else 0} | Total Samples: {n_samples}
        \tModel #parameter: {sum(p.numel() for p in model.parameters())}
    """
    logger.debug(val_report)

    # Setup learning HPs
    fts = _setup_tuning(
        **finetuning_config,
        model=model,
        task_type=task_type,
        is_classification=is_classification,
        is_data_parallel=is_data_parallel,
    )
    logger.debug(fts.report_str)

    # Setup Forward Pass Function
    categorical_features_index = (
        [int(i) for i in categorical_features_index]
        if categorical_features_index is not None
        else None
    )
    scaler = GradScaler(
        enabled=use_grad_scaler,
        growth_interval=100,
    )
    model_forward_fn = partial(
        _model_forward,
        n_classes=n_classes,
        categorical_features_index=categorical_features_index,
        use_autocast=use_autocast,
        device=device,
        is_data_parallel=is_data_parallel,
    )

    # Setup validation function
    adaptive_es, optimizer = fts.adaptive_es, fts.optimizer
    validation_metric = get_metric(metric=validation_metric, problem_type=task_type)
    if use_sklearn_interface_for_validation:
        if model_for_validation is not None:
            if hasattr(model_for_validation, 'executor_'):
                raise ValueError(f"model_for_validation must NOT be fitted")
        else:
            model_for_validation = TabPFNRegressor() if task_type == TaskType.REGRESSION else TabPFNClassifier()
        # this is required as memory_saving_mode can not be used during training
        model_for_validation.memory_saving_mode = False
    validate_tabpfn_fn = partial(
        validate_tabpfn,
        X_train=torch.tensor(X_train.values)
        .reshape(X_train.shape[0], 1, X_train.shape[1])
        .float(),
        y_train=torch.tensor(y_train.values).reshape(y_train.shape[0], 1, 1).float(),
        X_val=torch.tensor(X_val.values)
        .reshape(X_val.shape[0], 1, X_val.shape[1])
        .float(),
        y_val=torch.tensor(y_val.values).reshape(y_val.shape[0], 1, 1).float(),
        validation_metric=validation_metric,
        model_forward_fn=model_forward_fn,
        task_type=task_type,
        device=device,
        use_sklearn_interface_for_validation=use_sklearn_interface_for_validation,
        model_for_validation=model_for_validation
    )
    model.eval()
    optimizer.eval()
    with torch.no_grad():
        best_validation_loss = validate_tabpfn_fn(
            model=model,
        )  # Initial validation loss
    adaptive_es.update(cur_round=0, is_best=True)

    # Setup step results trace
    step_results_over_time = []
    step_results_over_time.append(
        FineTuneStepResults(
            step_index=0,
            best_validation_loss=best_validation_loss,
            best_validation_score=validation_metric.convert_error_to_score(
                best_validation_loss,
            ),
            training_loss=0.0,
            validation_loss=best_validation_loss,
            patience_left=adaptive_es.remaining_patience(cur_round=0),
            time_left=time_limit,
            device_utilization=torch.cuda.utilization(device=device)
            if device == SupportedDevice.GPU
            else 0.0,
            step_with_update=False,
            optimizer_step_skipped=False,
            grad_norm_before_clip=-1,
        ),
    )
    torch.save(
        dict(
            state_dict=model.module.state_dict() if is_data_parallel else model.state_dict(),
            config=checkpoint_config),
        str(save_path_to_fine_tuned_model),
    )
    logger.debug(f"Initial validation loss: {best_validation_loss}")

    # Setup data loader
    data_loader = get_data_loader(
        X_train=X_train,
        y_train=y_train,
        batch_size=fts.batch_size,
        max_steps=fts.max_steps,
        torch_rng=torch_rng,
        is_classification=is_classification,
        num_workers=fts.data_loader_workers,
    )
    # Setup progress bar
    iter_steps_pbar = tqdm(
        enumerate(data_loader, start=1),
        desc="Fine-tuning Steps",
        total=fts.max_steps,
        initial=1,
        disable=disable_progress_bar,
    )

    # Fine-Tuning Loop
    early_stop_no_imp = False
    early_stop_no_time = False
    gradient_accumulation_steps = (
        fts.update_every_n_steps if fts.update_every_n_steps > 1 else None
    )
    optimizer.zero_grad()
    skipped_steps = 0
    for step_i, batch_data in iter_steps_pbar:
        # Check for updating
        update_now = (step_i + 1) % fts.update_every_n_steps == 0
        validate_now = (step_i + 1) % fts.validate_every_n_steps == 0

        model.train()
        optimizer.train()
        step_results = _fine_tune_step(
            batch_X_train=batch_data["X_train"],
            batch_X_test=batch_data["X_test"],
            batch_y_train=batch_data["y_train"],
            batch_y_test=batch_data["y_test"],
            device=device,
            optimizer=optimizer,
            model_forward_fn=model_forward_fn,
            loss_fn=fts.loss_fn,
            gradient_accumulation_steps=gradient_accumulation_steps,
            # Updated by the loop
            model=model,
            scaler=scaler,
            step_with_update=update_now,
        )

        if step_results.optimizer_step_skipped:
            logger.info("\nOptimizer step skipped due to NaNs/infs in grad scaling.")
            validate_now = False
            skipped_steps += 1

        # -- Validate & save model
        if validate_now:
            model.eval()
            optimizer.eval()
            with torch.no_grad():
                validation_loss = validate_tabpfn_fn(model=model)

            # -- Check tuning state
            is_best = validation_loss < best_validation_loss
            early_stop_no_imp = adaptive_es.update(
                cur_round=(step_i - skipped_steps) // fts.update_every_n_steps,
                is_best=is_best,
            )
            if is_best:
                best_validation_loss = validation_loss
                torch.save(
                    dict(
                        state_dict=model.module.state_dict() if is_data_parallel else model.state_dict(),
                        config=checkpoint_config),
                    str(save_path_to_fine_tuned_model),
                )
        else:
            validation_loss = step_results_over_time[-1].validation_loss
            early_stop_no_imp = False

        time_spent = time.time() - st_time
        time_left = time_limit - time_spent
        early_stop_no_time = (time_left <= 0) or (
            time_left <= ((time_spent / step_i) * 1.1)
        )

        # -- Track Progress
        step_results = step_results.register_meta_state(
            step_index=step_i,
            validation_loss=validation_loss,
            best_validation_loss=best_validation_loss,
            best_validation_score=validation_metric.convert_error_to_score(
                best_validation_loss,
            ),
            patience_left=adaptive_es.remaining_patience(
                cur_round=(step_i - skipped_steps) // fts.update_every_n_steps,
            ),
            time_left=time_left,
        )
        if use_wandb:
            wandb.log(
                {
                    "train_loss": step_results.training_loss,
                    "val_loss": step_results.validation_loss,
                    "grad_norm": step_results.grad_norm_before_clip,
                },
            )
        iter_steps_pbar.set_postfix(step_results.to_results_dict())
        step_results_over_time.append(step_results)

        # Fix Initial training loss
        if step_i == 1:
            step_results_over_time[0].training_loss = step_results.training_loss

        # -- Early Stopping
        # Break from adaptive early stopping
        # Break if not enough time for another epoch
        if early_stop_no_imp or early_stop_no_time:
            break

    _tore_down_tuning(
        task_type=task_type,
        step_results_over_time=step_results_over_time,
        fts=fts,
        early_stop_no_imp=early_stop_no_imp,
        early_stop_no_time=early_stop_no_time,
        show_training_curve=show_training_curve,
        st_time=st_time,
    )


def _model_forward(
    *,
    model: PerFeatureTransformer,
    X_train: torch.Tensor,  # (n_samples, batch_size, n_features)
    y_train: torch.Tensor,  # (n_samples, batch_size, 1)
    X_test: torch.Tensor,  # (n_samples, batch_size, n_features)
    n_classes: int | None,
    softmax_temperature: torch.Tensor | None = None,
    categorical_features_index: list[torch.Tensor] | None,
    use_autocast: bool = True,
    forward_for_validation: bool = False,
    device: SupportedDevice,
    outer_loop_autocast: bool = False,
    is_data_parallel: bool,
) -> torch.Tensor:
    """Wrapper function to perform a forward pass with a TabPFN model.

    Arguments:
    ----------
    model: PerFeatureTransformer
        The model to use for the forward pass.
    X_train: torch.Tensor
        The training features.
    y_train: torch.Tensor
        The training target.
    X_test: torch.Tensor
        The test features.
    n_classes: int | None
        The number of classes for classification tasks, otherwise None.
    softmax_temperature: torch.Tensor | None
        The softmax temperature for the model, used to scale the logits.
        If None, no scaling is applied.
    categorical_features_index: list[int] | None
        The indices of the categorical features.
    use_autocast: bool
        Whether to use FP16 precision for the forward pass.
        This is required for flash attention!
    forward_for_validation: boo
        If True, this indicates that this is a forward pass for a validation score.
        This means that a regression model will return predictions instead of logits for the bar distribution.
    device: SupportedDevice
        The device to use for autocasting in the forward pass.

    Returns:
    --------
    pred_logits: torch.Tensor
        The predicted logits of the model. Logits are softmax scaled and selected down to:
            - classification: (n_samples, batch_size, n_classes)
            - regression: (n_samples, batch_size)
    """
    is_classification = n_classes is not None
    if not is_classification:
        # TabPFN model assumes z-normalized inputs.
        mean = y_train.mean(dim=0)
        std = y_train.std(dim=0)
        y_train = (y_train - mean) / std

    forward_kwargs = dict(
        train_x=X_train,
        train_y=y_train,
        test_x=X_test,
        categorical_inds=categorical_features_index,
    )

    if outer_loop_autocast:
        pred_logits = model(**forward_kwargs)
    else:
        with autocast(device_type=device, enabled=use_autocast):
            pred_logits = model(**forward_kwargs)

    if is_classification:
        pred_logits = pred_logits[:, :, :n_classes].float()

        if softmax_temperature is not None:
            pred_logits = pred_logits / softmax_temperature
    else:
        # Need to go step-wise over batch size as bar_dist.mean() does not support batched output.
        pred_logits = pred_logits.float()

        if softmax_temperature is not None:
            pred_logits = pred_logits / softmax_temperature

        if forward_for_validation:
            new_pred_logits = []
            for batch_i in range(pred_logits.shape[1]):
                bar_dist = deepcopy(model.module.criterion if is_data_parallel else model.criterion)
                bar_dist.borders = (
                    bar_dist.borders * std[batch_i] + mean[batch_i]
                ).float()
                new_pred_logits.append(bar_dist.mean(pred_logits[:, batch_i, :]))
            pred_logits = torch.stack(new_pred_logits, dim=-1)

    return pred_logits


def _fine_tune_step(
    *,
    batch_X_train: torch.Tensor,
    batch_X_test: torch.Tensor,
    batch_y_train: torch.Tensor,
    batch_y_test: torch.Tensor,
    device: SupportedDevice,
    model: PerFeatureTransformer,
    optimizer: Optimizer,
    model_forward_fn: Callable,
    loss_fn: _Loss,
    scaler: GradScaler,
    step_with_update: bool,
    gradient_accumulation_steps: int | None = None,
) -> FineTuneStepResults:
    """Perform one fine-tuning step for a TabPFN model.

    Arguments:
    ----------
    batch_X_train: torch.Tensor
        The training features.
    batch_X_test: torch.Tensor
        The test features.
    batch_y_train: torch.Tensor
        The training target.
    batch_y_test: torch.Tensor
        The test target.
    device: SupportedDevice
        The device to use for fine-tuning.
    model: PerFeatureTransformer
        The model to fine-tune.
    optimizer: torch.optim.Optimizer
        The optimizer to use for fine-tuning.
    model_forward_fn: Callable
        The forward pass function for the model.
    loss_fn: _Loss
        The loss function to use.
    scaler: GradScaler
        The gradient scaler to use for FP16 precision.
    step_with_update: bool
        Whether the optimizer, lr scheduler, and grad scaler shall be updated in this step.
    gradient_accumulation_steps: int
        The number of steps to accumulate gradients before updating the model.

    Returns:
    --------
    model: PerFeatureTransformer
        The fine-tuned model.
    step_results: FineTuneStepResults
        The results of the fine-tuning step.
    """
    # Move batch dimensions
    batch_X_train = torch.movedim(batch_X_train, 0, 1).to(device)
    batch_X_test = torch.movedim(batch_X_test, 0, 1).to(device)
    batch_y_train = torch.movedim(batch_y_train, 0, 1).to(device)
    batch_y_test = torch.movedim(batch_y_test, 0, 1).to(device)

    # Forward Mixed Precision
    with autocast(device_type=device, enabled=scaler.is_enabled()):
        pred_logits = model_forward_fn(  # autocast in model_forward_fn
            model=model,
            X_train=batch_X_train,
            y_train=batch_y_train,
            X_test=batch_X_test,
            outer_loop_autocast=True,
        )
        loss = compute_loss(loss_fn=loss_fn, logits=pred_logits, target=batch_y_test)

        if gradient_accumulation_steps is not None:
            loss = loss / gradient_accumulation_steps

    # Backward, Scaled for Mixed Precision
    scaler.scale(loss).backward()

    # Update
    optimizer_step_skipped = False
    grad_norm = -1
    if step_with_update:
        # Clip grads
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
            error_if_nonfinite=False,
        ).item()

        # Step optimizer and scaler
        org_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer_step_skipped = org_scale > scaler.get_scale()

        # Zero grad here due to gradient accumulation
        optimizer.zero_grad()

    return FineTuneStepResults(
        training_loss=loss.item()
        if gradient_accumulation_steps is None
        else loss.item() * gradient_accumulation_steps,
        device_utilization=torch.cuda.utilization(device=device)
        if device == SupportedDevice.GPU
        else 0.0,
        step_with_update=step_with_update,
        optimizer_step_skipped=optimizer_step_skipped,
        grad_norm_before_clip=grad_norm,
    )


def _setup_tuning(
    *,
    # Learning HPs
    learning_rate: float = 1e-8,
    batch_size: int = 1,
    update_every_n_steps: int = 1,
    validate_every_n_steps: int = 1,
    max_steps: int = 10000,
    adaptive_rate: float = 0.2,
    adaptive_offset: int = 5,
    min_patience: int = 20,
    max_patience: int = 100,
    data_loader_workers: int = 1,
    # Metadata
    model: PerFeatureTransformer,
    task_type: TaskType,
    is_classification: bool,
    is_data_parallel: bool,
) -> FineTuneSetup:
    return FineTuneSetup(
        optimizer=AdamWScheduleFree(model.parameters(), lr=learning_rate),
        max_steps=max_steps,
        adaptive_es=AdaptiveES(
            adaptive_rate=adaptive_rate,
            adaptive_offset=adaptive_offset,
            min_patience=min_patience,
            max_patience=max_patience,
        ),
        update_every_n_steps=update_every_n_steps,
        batch_size=batch_size,
        validate_every_n_steps=validate_every_n_steps,
        data_loader_workers=data_loader_workers,
        loss_fn=get_loss(
            task_type=task_type,
            borders=None if is_classification else
                    (model.module.criterion.borders if is_data_parallel else model.criterion.borders),
        ),
    )


def _tore_down_tuning(
    *,
    early_stop_no_imp: bool,
    early_stop_no_time: bool,
    show_training_curve: bool,
    st_time: float,
    step_results_over_time: list[FineTuneStepResults],
    fts: FineTuneSetup,
    task_type: TaskType,
) -> None:
    # -- Early Stopping reason (after tqdm finished)
    es_reason = None
    if early_stop_no_imp:
        es_reason = "Early stopping due to no improvement (AdaptiveES)."
    if early_stop_no_time:
        es_reason = "Early stopping due no time."
    if es_reason is not None:
        logger.log(10, es_reason)

    # -- Final Report
    best_step = np.argmin([x.validation_loss for x in step_results_over_time])
    fine_tuning_report = f"""=== Fine-Tuning Report for TabPFN ===
        \tTotal Time Spent: {time.time() - st_time}
        \tInitial Validation Loss: \t {step_results_over_time[0].validation_loss}
        \tBest Validation Loss: \t {step_results_over_time[-1].best_validation_loss}
        \tTotal Steps: {len(step_results_over_time)}
        \tBest Step: {best_step}
        \tEarly Stopping Reason: {es_reason}
        \tAvg. Time per Step: {(time.time() - st_time) / len(step_results_over_time)}
        \tAvg. Device Utilization: {np.mean([step.device_utilization for step in step_results_over_time])}
        """
    logger.info(fine_tuning_report)

    if show_training_curve:
        # --- Short Plot Hack
        import matplotlib.pyplot as plt
        import seaborn as sns

        train_loss_over_time = [step.training_loss for step in step_results_over_time]
        raw_train_loss_over_time = train_loss_over_time[:]
        for i in range(1, len(train_loss_over_time) + 1):
            train_loss_over_time[i - 1] = np.mean(
                raw_train_loss_over_time[max(0, i - fts.update_every_n_steps) : i],
            )
        validation_loss_over_time = [
            step.validation_loss for step in step_results_over_time
        ]
        plot_df = pd.DataFrame(
            {
                "train_loss": train_loss_over_time,
                "raw_train_loss": raw_train_loss_over_time,
                "validation_loss": validation_loss_over_time,
                "step": range(len(train_loss_over_time)),
            },
        )
        sns_plot_df = plot_df.melt(
            id_vars="step",
            value_vars=["train_loss", "validation_loss"],
            var_name="loss_type",
            value_name="loss",
        )
        fig, ax = plt.subplots(figsize=(8, 8))
        ax = sns.lineplot(
            data=sns_plot_df,
            x="step",
            y="loss",
            hue="loss_type",
            ax=ax,
            linewidth=3,
        )
        ax.axvline(
            x=best_step,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Best Step",
        )
        ax.legend(title="Legend")

        if fts.update_every_n_steps > 1:
            sns_plot_df = plot_df.melt(
                id_vars="step",
                value_vars=["raw_train_loss"],
                var_name="loss_type",
                value_name="loss",
            )
            sns.lineplot(
                data=sns_plot_df,
                x="step",
                y="loss",
                hue="loss_type",
                ax=ax,
                c="blue",
                alpha=0.5,
                linewidth=3,
            )

        plt.savefig(f"fine_tuning_loss_plot_{task_type}.png")
        plt.show()
