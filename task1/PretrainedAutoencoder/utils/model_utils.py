"""Model and checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def resolve_device(requested_device: str | None = None) -> torch.device:
    """Resolve a device string with CUDA fallback."""
    if requested_device and requested_device != "auto":
        return torch.device(requested_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def count_total_parameters(model: torch.nn.Module) -> int:
    """Count all parameters."""
    return sum(parameter.numel() for parameter in model.parameters())


def freeze_module(module: torch.nn.Module) -> None:
    """Freeze all parameters in a module."""
    for parameter in module.parameters():
        parameter.requires_grad = False


def unfreeze_module(module: torch.nn.Module) -> None:
    """Unfreeze all parameters in a module."""
    for parameter in module.parameters():
        parameter.requires_grad = True


def save_checkpoint(path: str | Path, payload: Dict[str, Any]) -> None:
    """Save a PyTorch checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    """Load a checkpoint dictionary."""
    return torch.load(Path(path), map_location=map_location)


def extract_state_dict(checkpoint: Dict[str, Any], preferred_key: str = "model_state") -> Dict[str, Any]:
    """Extract a state_dict from either a raw state_dict or a full checkpoint."""
    if preferred_key in checkpoint and isinstance(checkpoint[preferred_key], dict):
        return checkpoint[preferred_key]
    return checkpoint


def load_encoder_weights(
    encoder: torch.nn.Module,
    checkpoint_path: str | Path,
    strict: bool = True,
) -> None:
    """Load encoder weights from a dedicated encoder file or a full checkpoint."""
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint, preferred_key="encoder_state")
    encoder.load_state_dict(state_dict, strict=strict)


def _build_optimizer_parameter_groups(
    model: torch.nn.Module,
    lr: float,
    encoder_lr_scale: float,
) -> List[Dict[str, Any]] | List[torch.nn.Parameter]:
    """Build optimizer parameter groups, optionally using a smaller encoder LR."""
    if encoder_lr_scale == 1.0 or not hasattr(model, "encoder"):
        return [parameter for parameter in model.parameters() if parameter.requires_grad]

    encoder_params = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
    encoder_param_ids = {id(parameter) for parameter in encoder_params}
    other_params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in encoder_param_ids
    ]

    parameter_groups: List[Dict[str, Any]] = []
    if encoder_params:
        parameter_groups.append({"params": encoder_params, "lr": lr * encoder_lr_scale})
    if other_params:
        parameter_groups.append({"params": other_params, "lr": lr})

    if not parameter_groups:
        raise ValueError("No trainable parameters found for optimizer construction.")
    return parameter_groups


def build_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Build an optimizer from a small config dictionary."""
    name = str(config.get("name", "adam")).lower()
    lr = float(config.get("lr", 1e-3))
    weight_decay = float(config.get("weight_decay", 0.0))
    encoder_lr_scale = float(config.get("encoder_lr_scale", 1.0))
    parameters = _build_optimizer_parameter_groups(model, lr=lr, encoder_lr_scale=encoder_lr_scale)

    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            momentum=float(config.get("momentum", 0.9)),
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """Build a scheduler from config or return None."""
    name = str(config.get("name", "none")).lower()
    if name in ("none", "", "null"):
        return None
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(config.get("step_size", 10)),
            gamma=float(config.get("gamma", 0.1)),
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config.get("t_max", 10)),
            eta_min=float(config.get("eta_min", 0.0)),
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(config.get("mode", "max")),
            factor=float(config.get("factor", 0.5)),
            patience=int(config.get("patience", 3)),
            min_lr=float(config.get("min_lr", 0.0)),
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def step_scheduler(
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    monitor_value: float | None = None,
) -> None:
    """Advance a scheduler if one is configured."""
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        if monitor_value is None:
            raise ValueError("ReduceLROnPlateau requires a monitor value.")
        scheduler.step(monitor_value)
    else:
        scheduler.step()
