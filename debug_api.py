import json
from fastapi.testclient import TestClient
from api.main import app

payload = {
    "locations": [{"lat": 28.5, "lon": 77.1, "grid_id": "843901bffffffff"}]
}

with TestClient(app) as client:
    resp = client.post("/predict", json=payload)
    print("Status:", resp.status_code)
    try:
        print(json.dumps(resp.json(), indent=2))
    except Exception:
        print(resp.text)
