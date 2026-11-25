# background_changer

## Recorded video background replacement

The Streamlit app uses the TorchScript person-segmentation model in `model/`.
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
