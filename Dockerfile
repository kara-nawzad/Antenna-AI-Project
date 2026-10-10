FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8080 \
    # TensorFlow on a single shared vCPU: cap the thread pools. The defaults spawn
    # one worker per host core and then thrash, which costs both startup time and
    # steady-state latency.
    TF_NUM_INTRAOP_THREADS=1 \
    TF_NUM_INTEROP_THREADS=1 \
    OMP_NUM_THREADS=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ---------------------------------------------------------------------------
# Pre-fetch the model + dataset at BUILD time.
#
# This is the single largest fix for the 20-30 s cold start: forward_model_v2_curve.keras
# is ~115 MB and lives in a US-region Hub repo, while the app runs in fra. Downloading
# it on the first request put a transatlantic 115 MB transfer on the critical path of
# every machine start, because the container filesystem is ephemeral.
#
# Baking it into the image means _download_hf() finds the file already present and
# returns immediately, with no network call at all.
#
# The `|| echo` keeps the build green if the Hub is unreachable -- the app then falls
# back to downloading at runtime, exactly as before.
# ---------------------------------------------------------------------------
ARG HF_REPO=kara-nawzad/antenna-models
ENV HF_REPO=${HF_REPO}
RUN python -c "import os; from huggingface_hub import hf_hub_download; \
repo=os.environ['HF_REPO']; \
[print(f'[build] fetching {f} from {repo}') or hf_hub_download(repo_id=repo, filename=f, local_dir='.') \
 for f in ('forward_model_v2_curve.keras', 'antenna_data_cleaned.csv') \
 if not os.path.exists(f)]" \
    || echo "[build] WARNING: model pre-fetch failed; the app will download at runtime instead"

# Fail the build early (not at first request) if the scalers are unresolved LFS
# pointers. They are ~1.6 KB and must be real files inside the image.
RUN python -c "import sys; \
bad=[p for p in ('scaler_geo.pkl','scaler_perf.pkl') \
     if open(p,'rb').read(32).startswith(b'version https://git-lfs')]; \
sys.exit(f'ERROR: unresolved Git LFS pointer(s): {bad} -- run: git lfs pull') if bad else None"

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
    CMD curl -fsS http://localhost:8080/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "10_web_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
