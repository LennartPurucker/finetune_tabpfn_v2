from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import torch
if TYPE_CHECKING:
    from tabpfn.model.transformer import PerFeatureTransformer
    from torch.serialization import FILE_LIKE


def save_model(
    *,
    model: PerFeatureTransformer,
    save_path_to_fine_tuned_model: FILE_LIKE,
    checkpoint_config: dict,
    is_data_parallel: bool = False,
) -> None:
    """Save the fine-tuned model to disk in a TabPFN-readable checkpoint format."""
    # -- Save fine-tuned model
    torch.save(
        dict(
            state_dict=model.module.state_dict() if is_data_parallel else model.state_dict(),
            config=checkpoint_config),
        f=save_path_to_fine_tuned_model,
    )
