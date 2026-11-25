# background_changer

## Recorded video background replacement

The app loads the TorchScript model from the local `model/` directory when
available. For Streamlit Community Cloud, keep the large model out of GitHub and
host it in a public [Hugging Face model repository](https://huggingface.co/new).

1. Create a public model repository on Hugging Face.
2. Upload `deeplabv3_background_segmentation.pt` to the repository root.
3. In the Streamlit app settings, add this under **Secrets**, replacing the
   example with your Hugging Face repo ID:

   ```toml
   MODEL_REPO_ID = "your-account/your-model-repo"
   ```

The app downloads and caches the model on first use. Public repositories need no
Hugging Face token. The `.pt` file is ignored by Git so it won't be committed.

Install PyTorch for your platform (choose a CUDA-enabled build if available),
then install the application dependencies:

```powershell
python -m pip install torch
python -m pip install -r requirements.txt
```

Run the app from the repository root using the same Python environment that
installed the dependencies:

```powershell
python -m streamlit run src\app.py
```

Upload a background image and a recorded video, then click **Replace background**.
Preview and download the processed MP4 from the page. The exported video does not
include audio. CUDA is used when available; CPU processing may be slower.
