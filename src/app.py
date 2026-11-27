from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import streamlit as st
import torch
from streamlit.errors import StreamlitSecretNotFoundError

from utils import (
    INPUT_SIZE,
    MODEL_FILENAME,
    UNET_INPUT_SIZE,
    UNET_MODEL_FILENAME,
    UNET_MODEL_PATH,
    load_model,
    replace_background,
)

DEFAULT_MODEL_TYPE = "unet"
DEFAULT_UNET_REPO_ID = "rufaidakassem/unet_background_segmentation"
SUPPORTED_MODEL_TYPES = {"unet", "deeplabv3"}


def _get_setting(name: str) -> Optional[str]:
    value = os.environ.get(name)
    if value:
        return value
    try:
        return st.secrets[name]
    except (KeyError, StreamlitSecretNotFoundError):
        return None


@st.cache_resource(show_spinner="Loading the background-removal model...")
def get_model() -> tuple[torch.nn.Module, torch.device, tuple[int, int]]:
    repo_id = _get_setting("MODEL_REPO_ID")
    model_type = _get_setting("MODEL_TYPE")
    if model_type is None:
        model_type = DEFAULT_MODEL_TYPE
    model_type = model_type.strip().lower()
    if model_type not in SUPPORTED_MODEL_TYPES:
        supported_types = ", ".join(sorted(SUPPORTED_MODEL_TYPES))
        raise ValueError(
            f"Unsupported MODEL_TYPE '{model_type}'. Choose from: {supported_types}."
        )

    if model_type == "unet":
        if repo_id:
            model, device = load_model(
                repo_id=repo_id,
                filename=UNET_MODEL_FILENAME,
            )
        elif UNET_MODEL_PATH.is_file():
            model, device = load_model(model_path=UNET_MODEL_PATH)
        else:
            model, device = load_model(
                repo_id=DEFAULT_UNET_REPO_ID,
                filename=UNET_MODEL_FILENAME,
            )
        return model, device, UNET_INPUT_SIZE

    model, device = load_model(repo_id=repo_id, filename=MODEL_FILENAME)
    return model, device, INPUT_SIZE


def process_video(
    video_bytes: bytes,
    video_name: str,
    background: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    progress_callback: Optional[Callable[[float], None]] = None,
    *,
    input_size: tuple[int, int] = INPUT_SIZE,
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
                int(frame_count)
                if np.isfinite(frame_count) and frame_count > 0
                else 0
            )

            success, frame = capture.read()
            if not success or frame is None:
                raise ValueError("The uploaded video contains no readable frames.")

            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_width = width + width % 2
            output_height = height + height % 2
            writer = cv2.VideoWriter(
                str(output_path), fourcc, fps, (output_width, output_height)
            )
            if not writer.isOpened():
                raise RuntimeError("Could not create the processed MP4 video.")

            while success and frame is not None:
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                result = replace_background(
                    frame, background, model, device, input_size
                )
                if (output_height, output_width) != (height, width):
                    result = cv2.copyMakeBorder(
                        result,
                        0,
                        output_height - height,
                        0,
                        output_width - width,
                        cv2.BORDER_REPLICATE,
                    )
                writer.write(result)
                processed_frames += 1

                if progress_callback is not None and total_frames:
                    progress_callback(min(processed_frames / total_frames, 1.0))

                success, frame = capture.read()
        finally:
            capture.release()
            if writer is not None:
                writer.release()

        if processed_frames == 0:
            raise RuntimeError("No video frames were processed.")
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError("The processed video file is empty.")

        return output_path.read_bytes()


def main() -> None:
    st.set_page_config(page_title="Video Background Changer")
    st.title("Video Background Changer")
    st.write("Upload a background image and a recorded video to replace its background.")

    background_file = st.file_uploader(
        "Choose a background image",
        type=("png", "jpg", "jpeg", "webp", "bmp"),
    )
    video_file = st.file_uploader(
        "Choose a recorded video",
        type=("mp4", "mov", "avi", "mkv", "webm"),
    )
    if background_file is None or video_file is None:
        st.info("Upload both files to process the video.")
        return

    background_data = np.frombuffer(background_file.getvalue(), dtype=np.uint8)
    background = cv2.imdecode(background_data, cv2.IMREAD_COLOR)
    if background is None:
        st.error("The uploaded background file could not be decoded as an image.")
        return

    video_bytes = video_file.getvalue()
    fingerprint = hashlib.sha256(video_bytes + background.tobytes()).hexdigest()
    if st.session_state.get("processed_video_fingerprint") != fingerprint:
        st.session_state.pop("processed_video", None)
        st.session_state.pop("processed_video_fingerprint", None)

    st.image(background, channels="BGR", caption="Selected background")

    if st.button("Replace background", type="primary"):
        progress = st.progress(0.0, text="Processing video frames...")
        try:
            model, device, input_size = get_model()
            processed_video = process_video(
                video_bytes,
                video_file.name,
                background,
                model,
                device,
                progress_callback=lambda value: progress.progress(
                    value, text="Processing video frames..."
                ),
                input_size=input_size,
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
        st.subheader("Processed video")
        st.video(processed_video, format="video/mp4")
        st.download_button(
            "Download processed video",
            data=processed_video,
            file_name="background_replaced.mp4",
            mime="video/mp4",
        )
        st.caption("The processed video does not include the original audio.")


if __name__ == "__main__":
    main()
