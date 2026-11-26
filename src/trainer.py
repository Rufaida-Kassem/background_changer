"""Training and TorchScript export for the person-segmentation model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 15
    patience: int = 3
    learning_rate: float = 1e-4


class PersonSegmentationTrainer:
    """Fine-tune a segmentation model and export its best validation weights."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader[Any],
        validation_loader: DataLoader[Any],
        device: torch.device,
        config: TrainingConfig,
    ) -> None:
        if config.epochs < 1 or config.patience < 1:
            raise ValueError("epochs and patience must be positive.")
        if config.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if len(train_loader) == 0 or len(validation_loader) == 0:
            raise ValueError("Training and validation data loaders must not be empty.")

        self.model = model.to(device)
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.device = device
        self.config = config
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )

    def fit(self, output_path: Path) -> None:
        best_validation_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0

        for epoch in range(self.config.epochs):
            training_loss = self._run_epoch(self.train_loader, training=True)
            validation_loss = self._run_epoch(
                self.validation_loader, training=False
            )
            print(
                f"Epoch [{epoch + 1}/{self.config.epochs}] "
                f"Train Loss: {training_loss:.4f} | "
                f"Val Loss: {validation_loss:.4f}"
            )

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in self.model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                print(
                    "Validation loss did not improve. "
                    f"Early stopping counter: "
                    f"{epochs_without_improvement}/{self.config.patience}"
                )
                if epochs_without_improvement >= self.config.patience:
                    print("Early stopping triggered.")
                    break

        if best_state is None:
            raise RuntimeError("Training completed without a valid model checkpoint.")
        self.model.load_state_dict(best_state)
        self.model.eval()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            scripted_model = torch.jit.script(self.model)
            scripted_model.save(str(output_path))
        except Exception as error:
            raise RuntimeError(
                f"Could not export the trained model as TorchScript: {error}"
            ) from error
        print(
            f"Training complete. Best validation loss: {best_validation_loss:.4f}. "
            f"Saved model to {output_path}."
        )

    def _run_epoch(self, loader: DataLoader[Any], training: bool) -> float:
        self.model.train(training)
        total_loss = 0.0
        context = torch.enable_grad() if training else torch.inference_mode()
        with context:
            for images, masks in loader:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                outputs = self.model(images)["out"]
                loss = self.criterion(outputs, masks)
                if training:
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item()

        return total_loss / len(loader)
