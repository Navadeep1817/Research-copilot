FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch CPU first (separate step for caching)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Create data directories
RUN mkdir -p data/qdrant_storage data/raw

# Expose port
EXPOSE 8000

# Start command
CMD python startup.py
