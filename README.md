# AirScope — Air Quality Downscaling UI

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app starts in demo mode. The overview heatmap and placeholder model images are illustrative and are not measured NO₂ or real model predictions. On the Explore satellite data page, upload a CSV with columns `longitude`, `latitude`, and `no2` to display exported Earth Engine samples.

## Deploy
Push `streamlit_app.py` and `requirements.txt` to a GitHub repository, then create an app at https://share.streamlit.io/ and select `streamlit_app.py` as the entrypoint.

## Scientific note
Sentinel-5P NO₂ observations alone do not provide independent fine-resolution ground truth. Report actual downscaling performance only after defining a valid target/reference dataset and evaluation protocol.
