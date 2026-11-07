FROM python:3.10-slim

# Install system dependencies (ffmpeg, libGL, build tools for TensorFlow)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc g++ libopenblas-dev libstdc++6 ffmpeg libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip and install Python dependencies
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ENV XDG_CACHE_HOME=/app/.cache
RUN mkdir -p $XDG_CACHE_HOME
# Optional: pre‑download minimal Demucs model (demucs_ft)
# RUN python -c "from demucs import pretrained; pretrained.get_model('demucs_ft')"

# Copy the rest of the application code
COPY . ./

# Expose the port the app runs on
EXPOSE 7860

# Command to run the FastAPI app via uvicorn wrapper in app.py
CMD ["python", "app.py"]
