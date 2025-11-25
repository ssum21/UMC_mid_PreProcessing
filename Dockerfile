FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (ffmpeg)
# Clean up apt cache to reduce image size
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python dependencies
# Use --extra-index-url to ensure CPU version of PyTorch is installed (much smaller)
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the port
EXPOSE 8000

# Start the server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
