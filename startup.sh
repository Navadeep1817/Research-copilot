#!/bin/bash
# startup.sh — runs on Railway boot
# Seeds the knowledge base if empty, then starts the API server

echo "Starting Research Copilot..."

# Create data directories
mkdir -p data/qdrant_storage data/raw

# Seed knowledge base on first boot (idempotent — safe to run multiple times)
echo "Seeding knowledge base..."
python scripts/seed_data.py

# Start FastAPI server
echo "Starting API server on port $PORT..."
exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
