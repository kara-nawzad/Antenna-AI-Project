# Dockerfile for deploying the SPU Neural Antenna Synthesis app on fly.io
# Uses an official Python slim base; installs TensorFlow CPU and Streamlit.
FROM python:3.11-slim

# System deps needed by TF/scipy/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    # Streamlit config
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8080 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install Python deps first (layer cached unless requirements change)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app (code, small scalers, assets)
COPY . .

# Large model files (forward_model_v2_curve.keras, s11_dip_specialist.pkl,
# antenna_data_cleaned.csv) are NOT baked into the image. They are downloaded
# from Hugging Face on first boot (engine_v2._download_hf / 10_web_app._download_hf)
# and cached in the container filesystem for the lifetime of the VM.

EXPOSE 8080

# Healthcheck (the app itself does a startup healthcheck, but fly likes one too).
# Start-period is generous: first cold boot downloads ~125 MB of models/CSV from
# Hugging Face AND loads TensorFlow, which can take 2-4 minutes.
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
    CMD curl -fsS http://localhost:8080/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "10_web_app.py", \
            "--server.port=8080", \
            "--server.address=0.0.0.0", \
            "--server.enableCORS=false", \
            "--server.enableXsrfProtection=false", \
            "--server.headless=true"]
