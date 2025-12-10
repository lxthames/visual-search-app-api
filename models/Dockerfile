# Use a slim Python base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies (for opencv, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app ./app
COPY static ./static

# Create data directory (will be mounted as volume)
RUN mkdir -p /app/data

# Optional: set env vars (can also be overridden at runtime)
ENV PYTHONUNBUFFERED=1 \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000

# Expose the port Uvicorn will run on
EXPOSE 8000

# Start FastAPI with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
