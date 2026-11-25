from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "model"
    / "deeplabv3_background_segmentation.pt"
)
INPUT_SIZE = (256, 256)
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def load_model(
    model_path: Optional[Path] = None,
) -> Tuple[torch.nn.Module, torch.device]:
    """Load the exported TorchScript person-segmentation model."""
    path = model_path if model_path is not None else MODEL_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Segmentation model not found: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(str(path), map_location=device)
    model.eval()
    return model, device


def predict_person_mask(
    frame: np.ndarray, model: torch.nn.Module, device: torch.device
) -> np.ndarray:
    """Return a smoothed person mask with the same dimensions as the BGR frame."""
    height, width = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_frame, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    normalized = (resized.astype(np.float32) / 255.0 - IMAGE_MEAN) / IMAGE_STD
    input_tensor = (
        torch.from_numpy(normalized.transpose(2, 0, 1))
        .unsqueeze(0)
        .to(device)
    )

    with torch.inference_mode():
        output = model(input_tensor)

    if not isinstance(output, dict) or "out" not in output:
        raise RuntimeError("The segmentation model did not return an 'out' mask.")

    mask = torch.sigmoid(output["out"]).squeeze().cpu().numpy()
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(mask, (15, 15), 0)


def replace_background(
    frame: np.ndarray,
    background: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    """Composite the person in a BGR frame over a replacement BGR background."""
    height, width = frame.shape[:2]
    mask = predict_person_mask(frame, model, device)
    background = cv2.resize(
        background, (width, height), interpolation=cv2.INTER_LINEAR
    )
    alpha = mask[:, :, np.newaxis]
    composite = frame.astype(np.float32) * alpha + background.astype(np.float32) * (
        1.0 - alpha
    )
    return np.clip(composite, 0, 255).astype(np.uint8)
