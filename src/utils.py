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

# Default edge-refinement settings (these reproduce the previous fixed behaviour).
DEFAULT_EROSION = 3
DEFAULT_SOFTNESS = 0.30
DEFAULT_RADIUS = 8


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
    """Refine a mask against image edges using a guided filter."""
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


def refine_alpha(
    mask: np.ndarray,
    frame: np.ndarray,
    erosion: int = DEFAULT_EROSION,
    softness: float = DEFAULT_SOFTNESS,
    radius: int = DEFAULT_RADIUS,
    eps: float = 1e-3,
) -> np.ndarray:
    """Turn a raw probability mask into a clean alpha matte.

    1. Erode slightly to remove the halo of the old background.
    2. Sharpen the transition with a smoothstep around 0.5 (``softness`` is the
       width of that transition).
    3. Snap the edge to real image edges (hair) with a guided filter.

    ``frame`` is the BGR image the mask was predicted from.
    """
    alpha = mask.astype(np.float32)
    if erosion > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * erosion + 1, 2 * erosion + 1)
        )
        alpha = cv2.erode(alpha, kernel)

    softness = float(np.clip(softness, 0.02, 1.0))
    low, high = 0.5 - softness / 2, 0.5 + softness / 2
    alpha = np.clip((alpha - low) / (high - low), 0, 1)
    alpha = alpha * alpha * (3 - 2 * alpha)

    guide = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    alpha = guided_filter(guide, alpha.astype(np.float32), r=max(int(radius), 1), eps=eps)
    return np.clip(alpha, 0, 1)


def composite(frame: np.ndarray, background: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Alpha-blend a BGR frame over a BGR background (resized to the frame)."""
    height, width = frame.shape[:2]
    resized_background = cv2.resize(background, (width, height))
    alpha = alpha[:, :, np.newaxis]
    blended = (
        frame.astype(np.float32) * alpha
        + resized_background.astype(np.float32) * (1.0 - alpha)
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def replace_background(
    frame: np.ndarray,
    background: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    input_size: tuple[int, int] = INPUT_SIZE,
    *,
    erosion: int = DEFAULT_EROSION,
    softness: float = DEFAULT_SOFTNESS,
    radius: int = DEFAULT_RADIUS,
) -> np.ndarray:
    """Composite a BGR frame over ``background`` using the refined person mask."""
    mask = predict_person_mask(frame, model, device, input_size)
    alpha = refine_alpha(mask, frame, erosion, softness, radius)
    return composite(frame, background, alpha)