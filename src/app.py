from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import streamlit as st
import torch
from streamlit.errors import StreamlitSecretNotFoundError

from utils import (
    DEFAULT_EROSION,
    DEFAULT_RADIUS,
    DEFAULT_SOFTNESS,
    INPUT_SIZE,
    MODEL_FILENAME,
    UNET_INPUT_SIZE,
    UNET_MODEL_FILENAME,
    UNET_MODEL_PATH,
    composite,
    load_model,
    predict_person_mask,
    refine_alpha,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DEFAULT_MODEL_TYPE = "unet"
DEFAULT_UNET_REPO_ID = "rufaidakassem/unet_background_segmentation"
SUPPORTED_MODEL_TYPES = {"unet", "deeplabv3"}

GITHUB_URL = "https://github.com/Rufaida-Kassem/background_changer"
HF_UNET_URL = "https://huggingface.co/rufaidakassem/unet_background_segmentation"
HF_DEEPLAB_URL = "https://huggingface.co/rufaidakassem/deeplabv3_background_segmentation"

# Optional demo content. Put images in these folders and they appear as samples.
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
SAMPLE_PHOTOS_DIR = ASSETS_DIR / "samples"
SAMPLE_BACKGROUNDS_DIR = ASSETS_DIR / "backgrounds"

IMAGE_TYPES = ("png", "jpg", "jpeg", "webp", "bmp")
VIDEO_TYPES = ("mp4", "mov", "avi", "mkv", "webm")

MAX_PHOTO_SIDE = 1600      # larger photos are downscaled to keep the demo fast
MAX_VIDEO_SECONDS = 20     # the free CPU host is slow, so keep videos short

# Set this to your measured validation IoU (for example 0.953) to show it in the
# sidebar. Leave it as None until you have computed it.
VALIDATION_IOU: Optional[float] = None


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RefineSettings:
    erosion: int = DEFAULT_EROSION
    softness: float = DEFAULT_SOFTNESS
    radius: int = DEFAULT_RADIUS


@dataclass(frozen=True)
class BackgroundChoice:
    kind: str                       # "image", "blur", "solid" or "gradient"
    image: Optional[np.ndarray] = None
    color_a: str = "#1E8C5A"
    color_b: str = "#4B6CB7"


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def _get_setting(name: str) -> Optional[str]:
    value = os.environ.get(name)
    if value:
        return value
    try:
        return st.secrets[name]
    except (KeyError, StreamlitSecretNotFoundError):
        return None


def get_model_type() -> str:
    model_type = _get_setting("MODEL_TYPE")
    if model_type is None:
        model_type = DEFAULT_MODEL_TYPE
    model_type = model_type.strip().lower()
    if model_type not in SUPPORTED_MODEL_TYPES:
        supported_types = ", ".join(sorted(SUPPORTED_MODEL_TYPES))
        raise ValueError(
            f"Unsupported MODEL_TYPE '{model_type}'. Choose from: {supported_types}."
        )
    return model_type


@st.cache_resource(show_spinner="Loading the background-removal model...")
def get_model() -> tuple[torch.nn.Module, torch.device, tuple[int, int]]:
    repo_id = _get_setting("MODEL_REPO_ID")
    model_type = get_model_type()

    if model_type == "unet":
        if repo_id:
            model, device = load_model(repo_id=repo_id, filename=UNET_MODEL_FILENAME)
        elif UNET_MODEL_PATH.is_file():
            model, device = load_model(model_path=UNET_MODEL_PATH)
        else:
            model, device = load_model(
                repo_id=DEFAULT_UNET_REPO_ID, filename=UNET_MODEL_FILENAME
            )
        return model, device, UNET_INPUT_SIZE

    model, device = load_model(repo_id=repo_id, filename=MODEL_FILENAME)
    return model, device, INPUT_SIZE


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #
def decode_image(data: bytes) -> Optional[np.ndarray]:
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def limit_size(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = max_side / max(height, width)
    if scale >= 1:
        return image
    size = (max(int(width * scale), 1), max(int(height * scale), 1))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def hex_to_bgr(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    red, green, blue = (int(color[i : i + 2], 16) for i in (0, 2, 4))
    return blue, green, red


def make_gradient(height: int, width: int, color_a: str, color_b: str) -> np.ndarray:
    start = np.array(hex_to_bgr(color_a), dtype=np.float32)
    end = np.array(hex_to_bgr(color_b), dtype=np.float32)
    xs = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    ys = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    t = ((xs + ys) / 2.0)[:, :, None]
    return (start * (1.0 - t) + end * t).astype(np.uint8)


def resolve_background(choice: BackgroundChoice, frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    if choice.kind == "image" and choice.image is not None:
        return choice.image
    if choice.kind == "blur":
        return cv2.GaussianBlur(frame, (0, 0), max(height, width) / 35)
    if choice.kind == "solid":
        return np.full((height, width, 3), hex_to_bgr(choice.color_a), dtype=np.uint8)
    return make_gradient(height, width, choice.color_a, choice.color_b)


def background_fingerprint(choice: BackgroundChoice) -> str:
    digest = hashlib.sha256(choice.kind.encode())
    if choice.image is not None:
        digest.update(choice.image.tobytes())
    digest.update((choice.color_a + choice.color_b).encode())
    return digest.hexdigest()


def list_assets(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        return {}
    extensions = {f".{ext}" for ext in IMAGE_TYPES}
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in extensions)
    return {p.stem.replace("_", " ").replace("-", " ").title(): p for p in files}


# --------------------------------------------------------------------------- #
# Photo pipeline (the mask is cached, so the sliders update instantly)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, max_entries=8)
def segment_photo(photo_bytes: bytes) -> tuple[np.ndarray, np.ndarray, float, str]:
    photo = decode_image(photo_bytes)
    if photo is None:
        raise ValueError("The photo could not be decoded as an image.")
    photo = limit_size(photo, MAX_PHOTO_SIDE)

    model, device, input_size = get_model()
    start = time.perf_counter()
    mask = predict_person_mask(photo, model, device, input_size)
    return photo, mask, time.perf_counter() - start, str(device)


def encode_png(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Could not encode the processed photo as PNG.")
    return encoded.tobytes()


# --------------------------------------------------------------------------- #
# Video pipeline
# --------------------------------------------------------------------------- #
def to_browser_mp4(path: Path) -> bytes:
    """Re-encode to H.264 when ffmpeg exists, so every browser can play it."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        output = path.with_name("processed_h264.mp4")
        command = [
            ffmpeg, "-y", "-loglevel", "error", "-i", str(path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(output),
        ]
        try:
            subprocess.run(command, check=True, timeout=300)
            if output.is_file() and output.stat().st_size > 0:
                return output.read_bytes()
        except (subprocess.SubprocessError, OSError):
            pass
    return path.read_bytes()


def process_video(
    video_bytes: bytes,
    video_name: str,
    choice: BackgroundChoice,
    model: torch.nn.Module,
    device: torch.device,
    settings: RefineSettings,
    input_size: tuple[int, int],
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> bytes:
    suffix = Path(video_name).suffix or ".mp4"

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / f"input{suffix}"
        output_path = Path(temp_dir) / "processed.mp4"
        input_path.write_bytes(video_bytes)

        capture = cv2.VideoCapture(str(input_path))
        writer = None
        processed_frames = 0
        try:
            if not capture.isOpened():
                raise ValueError(
                    "Could not open this video. Try an MP4 encoded with H.264."
                )

            fps = capture.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError("Could not read a valid frame rate from this video.")

            frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            total_frames = (
                int(frame_count) if np.isfinite(frame_count) and frame_count > 0 else 0
            )
            if total_frames and total_frames / fps > MAX_VIDEO_SECONDS:
                raise ValueError(
                    f"This video is {total_frames / fps:.0f} s long. The demo is "
                    f"limited to {MAX_VIDEO_SECONDS} s, so please upload a shorter clip."
                )

            success, frame = capture.read()
            if not success or frame is None:
                raise ValueError("The uploaded video contains no readable frames.")

            height, width = frame.shape[:2]
            output_width = width + width % 2
            output_height = height + height % 2
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (output_width, output_height),
            )
            if not writer.isOpened():
                raise RuntimeError("Could not create the processed MP4 video.")

            # Image, solid and gradient backgrounds are built once; blur follows the frame.
            fixed_background = (
                None if choice.kind == "blur" else resolve_background(choice, frame)
            )
            started = time.perf_counter()

            while success and frame is not None:
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                background = (
                    fixed_background
                    if fixed_background is not None
                    else resolve_background(choice, frame)
                )
                mask = predict_person_mask(frame, model, device, input_size)
                alpha = refine_alpha(
                    mask, frame, settings.erosion, settings.softness, settings.radius
                )
                result = composite(frame, background, alpha)

                if (output_height, output_width) != (height, width):
                    result = cv2.copyMakeBorder(
                        result, 0, output_height - height, 0, output_width - width,
                        cv2.BORDER_REPLICATE,
                    )
                writer.write(result)
                processed_frames += 1

                if progress_callback is not None and total_frames:
                    progress_callback(
                        processed_frames, total_frames, time.perf_counter() - started
                    )
                success, frame = capture.read()
        finally:
            capture.release()
            if writer is not None:
                writer.release()

        if processed_frames == 0:
            raise RuntimeError("No video frames were processed.")
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError("The processed video file is empty.")

        return to_browser_mp4(output_path)


# --------------------------------------------------------------------------- #
# UI: sidebar
# --------------------------------------------------------------------------- #
def render_sidebar() -> RefineSettings:
    with st.sidebar:
        st.header("About the model")
        try:
            model_type = get_model_type()
        except ValueError:
            model_type = DEFAULT_MODEL_TYPE

        if model_type == "unet":
            st.markdown(
                "**U-Net** with an ImageNet-pretrained **ResNet-34** encoder, "
                "fine-tuned to separate a person from the background.\n\n"
                "- Loss: BCE + Dice\n"
                "- Input size: 384 × 384\n"
                "- Data: Supervisely person dataset (2,667 images)\n"
                "- Exported with TorchScript"
            )
            model_url = HF_UNET_URL
        else:
            st.markdown(
                "**DeepLabV3** (ResNet-50) fine-tuned to separate a person "
                "from the background.\n\n"
                "- Loss: BCE\n"
                "- Input size: 256 × 256\n"
                "- Data: Supervisely person dataset (2,667 images)\n"
                "- Exported with TorchScript"
            )
            model_url = HF_DEEPLAB_URL

        if VALIDATION_IOU is not None:
            st.metric("Validation IoU", f"{VALIDATION_IOU:.3f}")

        st.divider()
        st.header("Edge refinement")
        st.caption("These controls only change how the mask is blended. Try them on hair.")
        erosion = st.slider(
            "Edge tightening (px)", 0, 8, DEFAULT_EROSION,
            help="Shrinks the mask slightly to remove the halo of the old background.",
        )
        softness = st.slider(
            "Edge softness", 0.05, 0.90, DEFAULT_SOFTNESS, 0.05,
            help="Width of the transition between person and background. "
                 "Low is sharp, high is feathered.",
        )
        radius = st.slider(
            "Hair detail radius", 1, 16, DEFAULT_RADIUS,
            help="How far the guided filter looks when snapping the edge to the photo. "
                 "Small keeps fine strands, large is smoother.",
        )

        st.divider()
        st.markdown(f"[Source code on GitHub]({GITHUB_URL})  \n[Model on Hugging Face]({model_url})")

    return RefineSettings(erosion=erosion, softness=softness, radius=radius)


# --------------------------------------------------------------------------- #
# UI: pickers
# --------------------------------------------------------------------------- #
def pick_photo() -> tuple[Optional[bytes], str]:
    samples = list_assets(SAMPLE_PHOTOS_DIR)
    options = (["Sample photo"] if samples else []) + ["Upload a photo"]
    source = st.radio("Photo", options, horizontal=True, key="photo_source")

    if source == "Sample photo":
        name = st.selectbox("Sample", list(samples), key="photo_sample")
        return samples[name].read_bytes(), name

    file = st.file_uploader("Choose a photo", type=IMAGE_TYPES, key="photo_upload")
    if file is None:
        return None, ""
    return file.getvalue(), file.name


def pick_background(prefix: str) -> Optional[BackgroundChoice]:
    samples = list_assets(SAMPLE_BACKGROUNDS_DIR)
    kinds = (["Sample image"] if samples else []) + [
        "Upload image", "Blur original", "Solid color", "Gradient",
    ]
    kind = st.selectbox("Background", kinds, key=f"{prefix}_bg_kind")

    if kind == "Sample image":
        name = st.selectbox("Sample background", list(samples), key=f"{prefix}_bg_sample")
        image = decode_image(samples[name].read_bytes())
        if image is None:
            st.error("That sample background could not be read.")
            return None
        return BackgroundChoice("image", image)

    if kind == "Upload image":
        file = st.file_uploader(
            "Choose a background image", type=IMAGE_TYPES, key=f"{prefix}_bg_upload"
        )
        if file is None:
            return None
        image = decode_image(file.getvalue())
        if image is None:
            st.error("The uploaded background could not be decoded as an image.")
            return None
        return BackgroundChoice("image", image)

    if kind == "Blur original":
        return BackgroundChoice("blur")

    if kind == "Solid color":
        color = st.color_picker("Color", "#1E8C5A", key=f"{prefix}_bg_color")
        return BackgroundChoice("solid", color_a=color)

    first, second = st.columns(2)
    color_a = first.color_picker("Start color", "#1E8C5A", key=f"{prefix}_bg_a")
    color_b = second.color_picker("End color", "#4B6CB7", key=f"{prefix}_bg_b")
    return BackgroundChoice("gradient", color_a=color_a, color_b=color_b)


# --------------------------------------------------------------------------- #
# UI: tabs
# --------------------------------------------------------------------------- #
def photo_tab(settings: RefineSettings) -> None:
    controls_photo, controls_background = st.columns(2)
    with controls_photo:
        photo_bytes, photo_name = pick_photo()
    with controls_background:
        choice = pick_background("photo")

    if photo_bytes is None or choice is None:
        st.info("Pick a sample or upload a photo and a background to see the result.")
        return

    try:
        with st.spinner("Segmenting the person..."):
            photo, mask, seconds, device = segment_photo(photo_bytes)
        alpha = refine_alpha(
            mask, photo, settings.erosion, settings.softness, settings.radius
        )
        result = composite(photo, resolve_background(choice, photo), alpha)
        png = encode_png(result)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        st.error(f"Could not process the photo: {error}")
        return

    original_column, result_column = st.columns(2)
    original_column.image(photo, channels="BGR", caption="Original")
    result_column.image(result, channels="BGR", caption="Background replaced")

    st.caption(f"Segmentation took {seconds:.2f} s on {device}.")
    st.download_button(
        "Download result (PNG)",
        data=png,
        file_name="background_replaced.png",
        mime="image/png",
    )

    st.subheader("Under the hood")
    mask_column, alpha_column, final_column = st.columns(3)
    mask_column.image(
        (np.clip(mask, 0, 1) * 255).astype(np.uint8),
        caption="1. Raw person mask from the network",
    )
    alpha_column.image(
        (alpha * 255).astype(np.uint8),
        caption="2. Refined alpha (tighten, sharpen, snap to edges)",
    )
    final_column.image(result, channels="BGR", caption="3. Alpha blend over the new background")
    st.caption(
        "The network predicts a low-resolution probability map. Refinement erodes it "
        "slightly, sharpens the transition, then uses a guided filter so the edge "
        "follows real strands of hair in the photo instead of a blurry outline."
    )


def video_tab(settings: RefineSettings) -> None:
    choice = pick_background("video")
    video_file = st.file_uploader(
        "Choose a recorded video", type=VIDEO_TYPES, key="video_upload"
    )
    st.caption(
        f"Videos are limited to {MAX_VIDEO_SECONDS} seconds on this demo. "
        "The processed video does not include the original audio."
    )
    if choice is None or video_file is None:
        st.info("Upload a background and a video to continue.")
        return

    video_bytes = video_file.getvalue()
    fingerprint = hashlib.sha256(
        video_bytes
        + background_fingerprint(choice).encode()
        + repr(settings).encode()
    ).hexdigest()
    if st.session_state.get("processed_video_fingerprint") != fingerprint:
        st.session_state.pop("processed_video", None)
        st.session_state.pop("processed_video_fingerprint", None)

    if st.button("Replace background", type="primary", key="replace_video"):
        progress = st.progress(0.0, text="Starting...")

        def report(done: int, total: int, elapsed: float) -> None:
            remaining = elapsed / done * (total - done)
            progress.progress(
                min(done / total, 1.0),
                text=f"Frame {done} of {total}, about {remaining:.0f} s left",
            )

        try:
            model, device, input_size = get_model()
            processed_video = process_video(
                video_bytes, video_file.name, choice, model, device,
                settings, input_size, report,
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
            st.error(f"Could not process the uploaded video: {error}")
        else:
            st.session_state["processed_video"] = processed_video
            st.session_state["processed_video_fingerprint"] = fingerprint
            st.success(f"Video processed on {device}.")
        finally:
            progress.empty()

    if st.session_state.get("processed_video_fingerprint") == fingerprint:
        processed_video = st.session_state["processed_video"]
        original_column, result_column = st.columns(2)
        original_column.subheader("Original")
        original_column.video(video_bytes)
        result_column.subheader("Background replaced")
        result_column.video(processed_video, format="video/mp4")
        st.download_button(
            "Download result (MP4)",
            data=processed_video,
            file_name="background_replaced.mp4",
            mime="video/mp4",
        )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(
        page_title="Photo & Video Background Changer",
        page_icon="🎬",
        layout="wide",
    )
    st.title("Photo & Video Background Changer")
    st.caption("Person segmentation with a fine-tuned U-Net · edge-aware alpha blending · PyTorch and OpenCV")

    settings = render_sidebar()

    photo_view, video_view = st.tabs(["📷 Photo", "🎥 Video"])
    with photo_view:
        photo_tab(settings)
    with video_view:
        video_tab(settings)

    st.divider()
    st.caption(
        f"Built by Rufaida Kassem · [GitHub]({GITHUB_URL}) · "
        f"[Model on Hugging Face]({HF_UNET_URL})"
    )


if __name__ == "__main__":
    main()