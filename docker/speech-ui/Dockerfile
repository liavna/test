# Speech.AI Docker Image
# Powered by HPE Private Cloud AI

FROM python:3.11-slim

# Metadata
LABEL maintainer="liavna"
LABEL version="2.0.0"
LABEL description="Speech.AI - Live Transcription & Diarization"

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Streamlit Configuration - CRITICAL for Istio/EZUA
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_ENABLE_CORS=true
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
ENV STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION=false
ENV STREAMLIT_SERVER_BASE_URL_PATH=""

# Working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY speech_ai_app.py .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Expose port
EXPOSE 8501

# Run Streamlit with all necessary flags
CMD ["streamlit", "run", "speech_ai_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=true", \
     "--server.enableXsrfProtection=false", \
     "--server.enableWebsocketCompression=false", \
     "--browser.gatherUsageStats=false"]
