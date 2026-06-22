"""
test_api.py — Run this to diagnose what the backend returns.
Place in research_copilot/ and run: python test_api.py
"""
import httpx
import json

print("Testing Research Copilot API...")
print("="*60)

try:
    with httpx.Client(timeout=120) as client:
        with client.stream(
            "POST",
            "http://localhost:8000/api/research",
            json={
                "query": "What is RAG?",
                "session_id": "test123",
                "strategy": "hybrid",
                "max_sources": 3,
                "evaluate": False,
            },
        ) as response:
            print(f"HTTP Status: {response.status_code}")
            print("-"*60)
            for line in response.iter_lines():
                if line.startswith("data:"):
                    try:
                        payload = json.loads(line[5:].strip())
                        event = payload.get("event", "")
                        data  = payload.get("data", {})
                        print(f"[{event}] {json.dumps(data)[:200]}")
                    except Exception as e:
                        print(f"Parse error: {e} | raw: {line[:100]}")
                elif line.strip():
                    print(f"RAW: {line[:100]}")

except httpx.ConnectError:
    print("ERROR: Cannot connect to localhost:8000")
    print("Make sure uvicorn is running in another terminal.")
except Exception as e:
    print(f"ERROR: {e}")

print("="*60)
print("Done.")