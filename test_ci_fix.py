import requests, json

# Test with actual hub cells (high probability)
print("=== Test CI on hub cells ===")
resp = requests.post("http://localhost:8000/batch", json={
    "locations": [
        {"lat": 28.7176, "lon": 77.1567, "grid_id": "cell_8259"},
        {"lat": 28.7266, "lon": 77.2799, "grid_id": "cell_8517"},
        {"lat": 28.7311, "lon": 77.0695, "grid_id": "cell_8593"},
        {"lat": 28.65, "lon": 77.23, "grid_id": "cell_4567"},
        {"lat": 28.60, "lon": 77.15, "grid_id": "cell_2345"},
    ],
    "city": "delhi"
})
data = resp.json()
for pred in data["predictions"]:
    gid = pred["grid_id"]
    p = pred["p_profit"]
    ci_lo = pred["ci_lower"]
    ci_hi = pred["ci_upper"]
    status = "OK" if ci_lo is not None and ci_hi is not None and ci_lo != ci_hi else "PROBLEM"
    print(f"  {gid}: p={p:.4f}, CI=[{ci_lo}, {ci_hi}] {status}")

# Test a batch of 20 cells to see variety
print("\n=== Test batch of 20 cells ===")
locs = []
for i in range(20):
    gid = f"cell_{1000 + i * 500}"
    locs.append({"lat": 28.5 + i * 0.02, "lon": 77.0 + i * 0.015, "grid_id": gid})
resp2 = requests.post("http://localhost:8000/batch", json={"locations": locs, "city": "delhi"})
data2 = resp2.json()
for pred in data2["predictions"]:
    p = pred["p_profit"]
    ci_lo = pred.get("ci_lower")
    ci_hi = pred.get("ci_upper")
    print(f"  {pred['grid_id']}: p={p:.4f}  CI=[{ci_lo}, {ci_hi}]")
