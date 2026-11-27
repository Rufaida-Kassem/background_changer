<div align="center">

# 🎬 Photo & Video Background Changer

**Replace the background in a photo or recorded video with any image, powered by a fine-tuned person-segmentation model.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://background-changer.streamlit.app/)
[![Model on Hugging Face](https://img.shields.io/badge/Model-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/rufaidakassem/deeplabv3_background_segmentation)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[**Try the app**](https://background-changer.streamlit.app/) · [**Get the model**](https://huggingface.co/rufaidakassem/deeplabv3_background_segmentation) · [**Train your own**](#-train-the-segmentation-model)

</div>

<!-- Add a demo GIF or before/after screenshot here, for example:
<p align="center"><img src="assets/demo.gif" width="700" alt="Demo"></p>
-->

---

## ✨ Features

- 🖼️ **Any background image**: PNG, JPG/JPEG, WEBP, and BMP.
- 📷 **Photo background replacement**: process a single photo and download the result as a PNG.
- 🎥 **Common video formats**: MP4, MOV, AVI, MKV, and WEBM.
- 🧍 **Person segmentation**: a fine-tuned U-Net (ResNet-34) binary model separates the person from the original background by default. DeepLabV3 (ResNet-50) is also supported.
- 🪄 **Edge-aware mask refinement**: the mask is eroded, sharpened with a smooth transition, and refined against image edges with a guided filter.
- 📊 **Progress tracking**: see processing progress while your video is rendered.
- ⬇️ **Preview and download**: watch the result in the app and save it as an MP4.
- ⚡ **GPU when available**: uses CUDA if present, otherwise falls back to CPU.
- 🧪 **Reproducible training**: a full pipeline to fine-tune DeepLabV3 and export it as TorchScript.

## 🧠 How it works

```mermaid
flowchart LR
    B[Input video] --> C[Read frame]
    C --> E[Resize and normalize]
    E --> M[Segmentation model<br/>person mask]
    M --> S[Resize and refine mask<br/>with guided filter]
    A[Background image<br/>resized to frame] --> D
    S --> D[Alpha composite<br/>person over background]
    C --> D
    D --> F[Write MP4 frame]
    F --> G[Preview and download]
```

For every frame the app:

1. Resizes it to the selected model's input size (384×384 for U-Net or 256×256 for DeepLabV3) and normalizes it with ImageNet statistics.
2. Runs the TorchScript model and applies a sigmoid to get a per-pixel person probability.
3. Resizes the mask to the frame, erodes it slightly, sharpens its transition, and refines it with the notebook's guided filter.
4. Blends the frame and the background: `output = frame × mask + background × (1 − mask)`.

### Model details

| Item           | Value                                                                                                                      |
| -------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Architecture   | U-Net with a ResNet-34 encoder by default; DeepLabV3 with a ResNet-50 backbone is also supported                           |
| Task           | Binary segmentation (person vs. background)                                                                                |
| Input size     | U-Net: 384 × 384; DeepLabV3: 256 × 256; both use ImageNet mean/std normalization                                           |
| Loss           | U-Net: BCE + Dice; DeepLabV3: `BCEWithLogitsLoss`                                                                          |
| Optimizer      | Adam, learning rate `1e-4`                                                                                                 |
| Training setup | 85/15 train/validation split, up to 15 epochs, early stopping (patience 3)                                                |
| Export         | Best validation checkpoint saved as a TorchScript file                                                                    |
| Dataset        | [Supervisely filtered person segmentation](https://www.kaggle.com/datasets/tapakah68/supervisely-filtered-segmentation-person-dataset) |

---

## ⚠️ Limitations

- The processed video **does not include the original audio**.
- **CPU processing can be slow** for longer videos. Use a CUDA-enabled PyTorch build if you have a GPU.
- Each frame is segmented **independently** (no temporal smoothing), so edges can shimmer slightly on fast motion or busy scenes.
- The model is trained for **people only**; other foreground objects are treated as background.
- If a video can't be decoded, try re-encoding it as an **MP4 (H.264)**.

---

## 🚀 Quick start

### 1. Clone and install

```bash
git clone https://github.com/Rufaida-Kassem/background_changer.git
cd background_changer
```

Install PyTorch for your platform first (choose a CUDA-enabled build if your machine supports it, and install a compatible `torchvision` alongside it), then the remaining requirements:

```bash
python -m pip install torch torchvision
python -m pip install -r requirements.txt
```

### 2. Configure the model

The app uses U-Net by default. It loads `model/unet_background_segmentation.pt` when that local file exists; otherwise it downloads `unet_background_segmentation.pt` from [rufaidakassem/unet_background_segmentation](https://huggingface.co/rufaidakassem/unet_background_segmentation). Both model files are excluded from Git. Set `MODEL_REPO_ID` to use a different Hugging Face repository containing the selected model's filename. The existing DeepLabV3 model is hosted at [rufaidakassem/deeplabv3_background_segmentation](https://huggingface.co/rufaidakassem/deeplabv3_background_segmentation).

<details open>
<summary>Windows (PowerShell)</summary>

```powershell
python -m streamlit run src/app.py
```

</details>

<details>
<summary>macOS / Linux</summary>

```bash
python -m streamlit run src/app.py
```

</details>

To select DeepLabV3 explicitly, set `MODEL_TYPE=deeplabv3` before launching (in PowerShell: `$env:MODEL_TYPE = "deeplabv3"`). If `MODEL_REPO_ID` is set, it must contain the filename for the selected model.

### 3. Run the app

```bash
python -m streamlit run src/app.py
```

Open the local URL printed by Streamlit, choose Photo or Video, upload the media and replacement background, then click **Replace background** to preview or download the result.

### ☁️ Deploying on Streamlit Community Cloud

No secrets are needed for the default U-Net deployment. To select the hosted DeepLabV3 model, add:

```toml
MODEL_TYPE = "deeplabv3"
MODEL_REPO_ID = "rufaidakassem/deeplabv3_background_segmentation"
```

---

## 🏋️ Train the segmentation model

Training uses the [Supervisely filtered person segmentation dataset](https://www.kaggle.com/datasets/tapakah68/supervisely-filtered-segmentation-person-dataset). Download and extract it locally; it is not included in this repository.

```bash
python src/pipeline.py --dataset-root path/to/dataset
```

By default, the pipeline discovers matching image and mask files by normalized filename. Files with `mask` in their path or name are treated as masks, and paths containing `collage` are ignored when discovering images. If automatic discovery doesn't match your layout, pass the directories explicitly:

```bash
python src/pipeline.py \
  --dataset-root path/to/dataset \
  --images-dir path/to/dataset/images \
  --masks-dir path/to/dataset/masks
```

The pipeline:

1. Checks sampled masks and image/mask pairs.
2. Splits the data into training and validation sets.
3. Fine-tunes DeepLabV3 (CUDA is used when available).
4. Exports the best validation model as TorchScript to `model/deeplabv3_background_segmentation.pt`.

<details>
<summary><b>Training arguments</b></summary>

| Argument          | Default                                      | Description                                                 |
| ----------------- | -------------------------------------------- | ----------------------------------------------------------- |
| `--output`        | `model/deeplabv3_background_segmentation.pt` | Output TorchScript model path                               |
| `--epochs`        | `15`                                         | Maximum number of training epochs                           |
| `--batch-size`    | `16`                                         | Batch size                                                  |
| `--learning-rate` | `0.0001`                                     | Adam optimizer learning rate                                |
| `--patience`      | `3`                                          | Validation epochs without improvement before early stopping |
| `--num-workers`   | `2`                                          | Data-loader worker processes                                |
| `--sample-size`   | `100`                                        | Number of image/mask pairs sampled for mask checks          |
| `--seed`          | `42`                                         | Random seed                                                 |
| `--no-pretrained` | Off                                          | Initialize without pretrained DeepLabV3 weights             |

Run `python src/pipeline.py --help` for the full command-line help.

</details>

### Sharing your trained model

To host a model on Hugging Face, upload `deeplabv3_background_segmentation.pt` to a model repository and point `MODEL_REPO_ID` at it.

---

## 📁 Repository layout

| Path                        | Purpose                                                           |
| --------------------------- | ----------------------------------------------------------------- |
| `src/app.py`                | Streamlit user interface and video processing                     |
| `src/utils.py`              | Model loading, person-mask prediction, and background compositing |
| `src/pipeline.py`           | Training command-line entry point                                 |
| `src/segmentation_model.py` | DeepLabV3 model definition                                        |
| `src/training_data.py`      | Dataset discovery, mask checks, and preprocessing                 |
| `src/trainer.py`            | Training loop, validation, and TorchScript export                 |
| `notebook/`                 | Colab notebook with the original training experiments             |
| `model/`                    | Locally generated segmentation model (not tracked by Git)         |
| `data/`                     | Local input data (not tracked by Git)                             |
| `outputs/`                  | Local processed videos and results (not tracked by Git)           |

## 🛠️ Built with

[PyTorch](https://pytorch.org/) · [torchvision (DeepLabV3)](https://pytorch.org/vision/) · [OpenCV](https://opencv.org/) · [NumPy](https://numpy.org/) · [Streamlit](https://streamlit.io/) · [Hugging Face Hub](https://huggingface.co/)

## 📄 License

Distributed under the [MIT License](LICENSE).

## 👩‍💻 Author

**Rufaida Kassem**: [GitHub](https://github.com/Rufaida-Kassem)