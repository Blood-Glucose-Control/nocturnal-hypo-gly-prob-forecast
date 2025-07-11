"""Checkpoint management for model training."""

from pathlib import Path
from typing import Any, Dict, Optional, Callable


class CheckpointManager:
    """Manages model checkpoints during training."""

    def __init__(
        self,
        output_dir: str = "models/checkpoints",
        checkpoint_frequency: int = 10,
        max_checkpoints: Optional[int] = None,
    ):
        """Initialize checkpoint manager.

        Args:
            output_dir: Directory to save checkpoints
            checkpoint_frequency: Save checkpoint every N epochs
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.output_dir = Path(output_dir)
        self.checkpoint_frequency = checkpoint_frequency
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []

    def should_checkpoint(self, epoch: int) -> bool:
        """Determine if we should save a checkpoint at this epoch."""
        return epoch % self.checkpoint_frequency == 0

    def save_checkpoint(
        self,
        model: Any,
        epoch: int,
        metrics: Dict[str, float],
        save_func: Callable[[Any, Path], None],
    ) -> Path:
        """Save a model checkpoint.

        Args:
            model: Model to checkpoint
            epoch: Current epoch number
            metrics: Training metrics to record
            save_func: Function to save the model

        Returns:
            Path to saved checkpoint
        """
        checkpoint_dir = self.output_dir / f"checkpoint_epoch_{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save the model
        save_func(model, checkpoint_dir)

        # Save metrics
        import json

        with open(checkpoint_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        self.checkpoints.append(checkpoint_dir)
        self._manage_checkpoints()

        return checkpoint_dir

    def _manage_checkpoints(self) -> None:
        """Remove old checkpoints if max_checkpoints is set."""
        if self.max_checkpoints is None:
            return

        if len(self.checkpoints) > self.max_checkpoints:
            checkpoints_to_remove = self.checkpoints[: -self.max_checkpoints]
            for checkpoint in checkpoints_to_remove:
                self._remove_checkpoint(checkpoint)
            self.checkpoints = self.checkpoints[-self.max_checkpoints :]

    def _remove_checkpoint(self, checkpoint_dir: Path) -> None:
        """Remove a checkpoint directory."""
        import shutil

        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
