import requests
import json

# Test dynamic behavior: same city, different parameters
r1 = requests.post('http://localhost:8000/optimize', json={
    'max_hubs': 3, 'min_separation_km': 2.0,
    'min_prob_threshold': 0.7, 'city': 'delhi'
})
r2 = requests.post('http://localhost:8000/optimize', json={
    'max_hubs': 10, 'min_separation_km': 1.0,
    'min_prob_threshold': 0.5, 'city': 'delhi'
})
d1 = r1.json()
d2 = r2.json()

print(f"Test 1 (strict):  {len(d1['selected_hubs'])} hubs, {d1['eligible_cells']} eligible, score={d1['total_score']}")
print(f"Test 2 (relaxed): {len(d2['selected_hubs'])} hubs, {d2['eligible_cells']} eligible, score={d2['total_score']}")
print(f"Different results? {d1['selected_hubs'] != d2['selected_hubs']}")
print(f"Results change with params? {'YES - DYNAMIC' if d1 != d2 else 'NO - STATIC'}")
