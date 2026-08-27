FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1 curl && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 STREAMLIT_SERVER_HEADLESS=true STREAMLIT_SERVER_ADDRESS=0.0.0.0 STREAMLIT_SERVER_PORT=8080
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 CMD curl -fsS http://localhost:8080/_stcore/health || exit 1
ENTRYPOINT ["streamlit", "run", "10_web_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
