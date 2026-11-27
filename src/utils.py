from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as transforms_functional


MODEL_FILENAME = "deeplabv3_background_segmentation.pt"
UNET_MODEL_FILENAME = "unet_background_segmentation.pt"
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME
UNET_MODEL_PATH = MODEL_DIR / UNET_MODEL_FILENAME
INPUT_SIZE = (256, 256)
UNET_INPUT_SIZE = (384, 384)


def load_model(
    model_path: Optional[Path] = None,
    repo_id: Optional[str] = None,
    filename: str = MODEL_FILENAME,
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
                    filename=filename,
                    repo_type="model",
                )
            )
        except (EntryNotFoundError, HfHubHTTPError, ValueError) as error:
            raise RuntimeError(
                f"Could not download '{filename}' from Hugging Face "
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
    frame: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    input_size: tuple[int, int] = INPUT_SIZE,
) -> np.ndarray:
    """Return the raw person-probability mask at the frame's dimensions."""
    height, width = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = transforms_functional.resize(
        Image.fromarray(rgb_frame),
        (input_size[1], input_size[0]),
    )
    input_tensor = transforms_functional.to_tensor(resized)
    input_tensor = transforms_functional.normalize(
        input_tensor,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    input_tensor = input_tensor.unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model(input_tensor)

    if isinstance(output, dict):
        if "out" not in output:
            raise RuntimeError("The segmentation model did not return an 'out' mask.")
        output = output["out"]
    if not isinstance(output, torch.Tensor):
        raise RuntimeError("The segmentation model did not return a tensor mask.")

    mask = torch.sigmoid(output).squeeze().cpu().numpy()
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)


def guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    r: int = 8,
    eps: float = 1e-3,
) -> np.ndarray:
    """Refine a mask against image edges using the notebook's guided filter."""
    kernel_size = (2 * r + 1, 2 * r + 1)

    def mean(image: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(image, -1, kernel_size)

    mean_guide = mean(guide)
    mean_src = mean(src)
    covariance = mean(guide * src) - mean_guide * mean_src
    variance = mean(guide * guide) - mean_guide * mean_guide
    coefficient = covariance / (variance + eps)
    intercept = mean_src - coefficient * mean_guide
    return mean(coefficient) * guide + mean(intercept)


def replace_background(
    frame: np.ndarray,
    background: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    input_size: tuple[int, int] = INPUT_SIZE,
) -> np.ndarray:
    """Composite a BGR frame using the notebook's refined person mask."""
    height, width = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    mask = predict_person_mask(frame, model, device, input_size)
    erosion_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * 3 + 1, 2 * 3 + 1)
    )
    alpha = cv2.erode(mask, erosion_kernel)

    low, high = 0.35, 0.65
    alpha = np.clip((alpha - low) / (high - low), 0, 1)
    alpha = alpha * alpha * (3 - 2 * alpha)

    guide = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    alpha = np.clip(
        guided_filter(guide, alpha.astype(np.float32), r=8, eps=1e-3),
        0,
        1,
    )

    resized_background = cv2.resize(rgb_background, (width, height))
    alpha = alpha[:, :, np.newaxis]
    composite = (
        rgb_frame.astype(np.float32) * alpha
        + resized_background.astype(np.float32) * (1.0 - alpha)
    )
    composite = np.clip(composite, 0, 255).astype(np.uint8)
    return cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
