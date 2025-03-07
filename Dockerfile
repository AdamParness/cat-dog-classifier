FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Make sure the models directory exists
RUN mkdir -p models

# Create a non-root user and switch to it
RUN useradd -m -u 1000 user
USER user

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV TENSORFLOW_ENABLE_MKL=0

# Run the application
CMD uvicorn app:app --host 0.0.0.0 --port $PORT 