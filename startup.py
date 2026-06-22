"""
startup.py — Railway entry point.
Seeds knowledge base then starts uvicorn.
"""
import os
import subprocess
import sys

# Seed knowledge base
print("Seeding knowledge base...")
subprocess.run([sys.executable, "scripts/seed_data.py"], check=False)

# Start server
port = os.environ.get("PORT", "8000")
print(f"Starting server on port {port}...")
os.execv(sys.executable, [
    sys.executable, "-m", "uvicorn",
    "backend.main:app",
    "--host", "0.0.0.0",
    "--port", port,
]) 