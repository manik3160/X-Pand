import requests
import time

payload = {"locations": [{"lat": 28.5, "lon": 77.1, "grid_id": f"cell_{i}"} for i in range(13000)]}

print("Sending request to /batch...")
start = time.time()
try:
    resp = requests.post("http://localhost:8000/batch", json=payload, timeout=60)
    print(f"Status: {resp.status_code}")
    print(f"Time taken: {time.time() - start:.2f}s")
    if resp.status_code == 200:
        print(f"Predictions returned: {len(resp.json()['predictions'])}")
    else:
        print(f"Error: {resp.text}")
except Exception as e:
    print(f"Failed: {e}")
