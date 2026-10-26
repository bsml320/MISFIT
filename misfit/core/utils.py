"""Seeding, logging, config loading, and result saving utilities."""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
import random
from typing import TypeVar

import numpy as np
import torch
import yaml

T = TypeVar("T")


def seed_everything(seed: int) -> None:
    """Set random seed for torch, numpy, and Python random.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict:
    """Load YAML config file and return as dict.

    Args:
        path: Path to YAML file.

    Returns:
        Config as nested dict.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def save_config(config: dict, path: str | Path) -> None:
    """Save config dict to YAML file.

    Args:
        config: Config dict to save.
        path: Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def config_to_dataclass(config_dict: dict, cls: type[T]) -> T:
    """Convert dict to dataclass instance.

    Handles nested dataclass fields by recursively converting sub-dicts.

    Args:
        config_dict: Dict with keys matching dataclass fields.
        cls: Target dataclass type.

    Returns:
        Instance of cls.
    """
    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}
    for key, value in config_dict.items():
        if key not in field_types:
            continue
        field_type = field_types[key]
        # Handle nested dataclass fields
        if isinstance(value, dict) and dataclasses.is_dataclass(field_type):
            value = config_to_dataclass(value, field_type)
        kwargs[key] = value
    return cls(**kwargs)


def dataclass_to_dict(obj: object) -> dict:
    """Convert a dataclass instance to a serializable dict.

    Args:
        obj: Dataclass instance.

    Returns:
        Dict with all fields.
    """
    if not dataclasses.is_dataclass(obj):
        raise TypeError(f"{type(obj)} is not a dataclass.")
    result = {}
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name)
        if dataclasses.is_dataclass(value):
            value = dataclass_to_dict(value)
        result[f.name] = value
    return result


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configure structured logging.

    Args:
        level: Logging level string.
        log_file: Optional file to write logs to.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger("fire_mechanist")
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)

        if log_file is not None:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(str(log_file))
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger


class _NumpyTorchEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays and torch tensors."""

    def default(self, obj: object) -> object:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        return super().default(obj)


def save_results(results: dict, path: str | Path) -> None:
    """Save experiment results as JSON with numpy/tensor serialization.

    Args:
        results: Dict of results (may contain numpy arrays and torch tensors).
        path: Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, cls=_NumpyTorchEncoder, indent=2)


def load_results(path: str | Path) -> dict:
    """Load experiment results from JSON.

    Args:
        path: Path to JSON file.

    Returns:
        Results dict.
    """
    path = Path(path)
    with open(path) as f:
        return json.load(f)
