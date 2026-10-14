# Streamlit deployment package

This folder is a self-contained app root for local use or Streamlit Community Cloud. It contains the inference app, its trained model bundle, and only the runtime dependencies needed to score transactions.

## Run locally

From this folder, create/activate your Python 3.12 environment, install `requirements.txt`, then run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Upload a CSV with `Time`, `V1` through `V28`, and `Amount`. The app returns each transaction with a fraud probability and threshold-based flag.

## Deploy to Streamlit Community Cloud

Push this folder's contents to a GitHub repository (or use this folder as the repository root). In Streamlit Community Cloud, choose **Create app**, select the repository and branch, and set the app file path to `streamlit_app.py`. The app requires the model file at `artifacts/best_model.joblib`; keep that file in the repository. Community Cloud reads dependencies from this folder's `requirements.txt`.
