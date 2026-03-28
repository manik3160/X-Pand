import requests
payload = {
    "locations": [{"lat": 28.5, "lon": 77.1, "grid_id": "843901bffffffff"}] # Delhi center
}
resp = requests.post("http://localhost:8000/predict", json=payload)
print(f"Status: {resp.status_code}")
try:
    print("SHAP drivers found:", len(resp.json()["predictions"][0].get("shap_drivers", [])))
except Exception:
    print(resp.text)
