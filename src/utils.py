from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch


MODEL_FILENAME = "deeplabv3_background_segmentation.pt"
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "model"
    / MODEL_FILENAME
)
INPUT_SIZE = (256, 256)
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def load_model(
    model_path: Optional[Path] = None,
    repo_id: Optional[str] = None,
) -> Tuple[torch.nn.Module, torch.device]:
    """Load the TorchScript model locally or from a public Hugging Face repo."""
    if model_path is not None:
        path = model_path
    elif repo_id:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError

        try:
            path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=MODEL_FILENAME,
                    repo_type="model",
                )
            )
        except (EntryNotFoundError, HfHubHTTPError, ValueError) as error:
            raise RuntimeError(
                f"Could not download '{MODEL_FILENAME}' from Hugging Face "
                f"model repo '{repo_id}'. Check the repo ID and uploaded filename."
            ) from error
    else:
        path = MODEL_PATH

    if not path.is_file():
        if model_path is None and not repo_id:
            raise FileNotFoundError(
                f"Segmentation model not found at {path}. Set MODEL_REPO_ID in "
                "Streamlit Secrets or place the model at this path."
            )
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
