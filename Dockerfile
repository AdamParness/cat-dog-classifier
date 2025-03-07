FROM python:3.10-slim

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

# Run the application
CMD uvicorn app:app --host 0.0.0.0 --port $PORT 