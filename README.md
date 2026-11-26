# Video Background Changer

Replace the background in a recorded video with an image using a person
segmentation model.

**Try the app:** [background-changer.streamlit.app](https://background-changer.streamlit.app/)

## What it does

- Accepts a background image and a recorded video.
- Uses a DeepLabV3-based binary segmentation model to separate the person from
  the original background, then composites the person over the selected image.
- Shows processing progress and lets you preview and download the result as an
  MP4.
- Uses CUDA when available and otherwise processes on the CPU.

Supported background image formats: PNG, JPG/JPEG, WEBP, and BMP. Supported
video upload formats: MP4, MOV, AVI, MKV, and WEBM. If a video cannot be
decoded, try an MP4 encoded with H.264. The processed video does not include
the original audio, and CPU processing may be slow for longer videos.

## Run locally

Use Python 3 and run these commands from the repository root. Install PyTorch
for your platform first; choose a CUDA-enabled build if your machine supports
it. Install a compatible `torchvision` version alongside PyTorch.

```powershell
python -m pip install torch torchvision
python -m pip install -r requirements.txt
python -m streamlit run src\app.py
```

Open the local URL printed by Streamlit. Upload a background image and a
recorded video, select **Replace background**, then preview or download the
processed MP4.

### Model setup

The app loads `model/deeplabv3_background_segmentation.pt` by default. The
trained model is excluded from Git, so obtain or train the TorchScript model
before running the app.

To load the model from a Hugging Face model repository instead, upload the file
`deeplabv3_background_segmentation.pt` to that repository and set `MODEL_REPO_ID`
to its repository ID. Locally, you can set the environment variable before
launching Streamlit:

```powershell
$env:MODEL_REPO_ID = "your-account/your-model-repo"
python -m streamlit run src\app.py
```

For Streamlit Community Cloud, add the following in the app's **Settings >
Secrets**, replacing the example with your public Hugging Face repository ID:

```toml
MODEL_REPO_ID = "your-account/your-model-repo"
```

The model is downloaded and cached on first use. A public Hugging Face
repository does not require a token. If `MODEL_REPO_ID` is not set, the app
looks for the model file in the local `model/` directory.

## Train the segmentation model

Training uses the [Supervisely filtered person segmentation
dataset](https://www.kaggle.com/datasets/tapakah68/supervisely-filtered-segmentation-person-dataset).
Download and extract the dataset locally; it is not included in this
repository.

From the repository root, run:

```powershell
python src\pipeline.py --dataset-root path\to\dataset
```

By default, the training pipeline discovers matching image and mask files by
normalized filename. It treats files with `mask` in their path or filename as
masks, and ignores paths containing `collage` when discovering images. If
automatic discovery does not match the dataset layout, provide both directories
explicitly:

```powershell
python src\pipeline.py `
  --dataset-root path\to\dataset `
  --images-dir path\to\dataset\images `
  --masks-dir path\to\dataset\masks
```

The pipeline checks sampled masks and image/mask pairs, splits the data into
training and validation sets, fine-tunes DeepLabV3, and exports the best
validation model as TorchScript. By default, the model is saved to
`model\deeplabv3_background_segmentation.pt`. Training uses CUDA when available.

Optional training arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `--output` | `model\deeplabv3_background_segmentation.pt` | Output TorchScript model path |
| `--epochs` | `15` | Maximum number of training epochs |
| `--batch-size` | `16` | Batch size |
| `--learning-rate` | `0.0001` | Adam optimizer learning rate |
| `--patience` | `3` | Validation epochs without improvement before early stopping |
| `--num-workers` | `2` | Data-loader worker processes |
| `--sample-size` | `100` | Number of image/mask pairs sampled for mask checks |
| `--seed` | `42` | Random seed |
| `--no-pretrained` | Off | Initialize without pretrained DeepLabV3 weights |

Run `python src\pipeline.py --help` for the full command-line help.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/app.py` | Streamlit user interface and video processing |
| `src/utils.py` | Model loading, person-mask prediction, and background compositing |
| `src/pipeline.py` | Training command-line entry point |
| `src/segmentation_model.py` | DeepLabV3 model definition |
| `src/training_data.py` | Dataset discovery, mask checks, and preprocessing |
| `src/trainer.py` | Training loop, validation, and TorchScript export |
| `model/` | Locally generated segmentation model (not tracked by Git) |
| `data/` | Local input data (not tracked by Git) |
| `outputs/` | Local processed videos and results (not tracked by Git) |

## License

This project is distributed under the [MIT License](LICENSE).
