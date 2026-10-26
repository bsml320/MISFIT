"""Optimization utilities for FIRE-MECHANIST training.

Provides EarlyStopping, a stateful monitor over a validation metric that
signals when training should halt.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("fire_mechanist")


class EarlyStopping:
    """Monitor validation metric for early stopping.

    Args:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum improvement to count as progress.
        mode: "min" (loss) or "max" (accuracy).
    """

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value: float | None = None
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Update with new metric value.

        Args:
            value: Current metric value.

        Returns:
            True if training should stop.
        """
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == "min":
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping triggered after %d epochs without improvement.",
                    self.counter,
                )

        return self.should_stop
