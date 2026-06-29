# Production image for the Streamlit dashboard.
# Works on Render, Railway, Fly.io, any VPS, or local Docker.
FROM python:3.12-slim

# System fonts (Hebrew glyphs for matplotlib) + curl for the healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-noto-core curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app.
COPY . .

# Honor a platform-provided $PORT when present (Render/Railway/Fly), else 8501.
ENV PORT=8501
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD curl --fail http://localhost:${PORT}/_stcore/health || exit 1

CMD streamlit run dashboard.py --server.port=${PORT} --server.address=0.0.0.0
