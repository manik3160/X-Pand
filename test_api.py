import requests
import json
import time

payload = {
    "locations": [{"lat": 28.5, "lon": 77.1, "grid_id": f"cell_{i}"} for i in range(12000)]
}

print("Sending request to API...")
start = time.time()
try:
    resp = requests.post("http://localhost:8000/batch", json=payload, timeout=30)
    print(f"Status: {resp.status_code}")
    print(f"Time taken: {time.time() - start:.2f}s")
    if resp.status_code == 200:
        print(f"Predictions returned: {len(resp.json()['predictions'])}")
except Exception as e:
    print(f"Error: {e}")
