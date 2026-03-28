# X-Pand.AI — Key Technical Changes

## 1. Dynamic API Response in Browser Network Tab

### Problem
When the mentor inspected the browser's **DevTools → Network** tab after clicking **RUN OPTIMIZER**, no API call was visible. Streamlit makes server-side Python calls to FastAPI, which are invisible to the browser. The mentor concluded the backend was "static."

### Solution
We made the FastAPI backend's responses **visible in the browser's Network tab** by injecting browser-side JavaScript `fetch()` calls that hit the API directly.

### Changes Made

#### `api/schemas.py`
- **Added `HubDetail` model** — a new Pydantic schema containing `grid_id`, `lat`, `lon`, `p_profit`, and `recommendation` for each selected hub.
- **Enriched `OptimizeResponse`** — added new fields:
  - `hub_details: List[HubDetail]` — full coordinate + profitability data per hub
  - `model_used: str` — e.g. `"LightGBM + BIP Solver"`
  - `processing_time_seconds: float` — wall-clock time for the optimization
  - `p_profit_range: List[float]` — `[min, max]` across all scored cells

#### `api/main.py`

1. **Imported `HubDetail`** into the route file.

2. **Updated `/optimize` endpoint** — after the BIP solver selects hubs, we now build a `hub_details` list by looking up each hub's row in the GeoDataFrame:
   ```python
   hub_details = []
   for hid in selected_ids:
       hub_row = gdf[gdf["grid_id"] == hid]
       if not hub_row.empty:
           row = hub_row.iloc[0]
           p = float(row["p_profit"])
           hub_details.append(HubDetail(
               grid_id=hid,
               lat=float(row["centroid_lat"]),
               lon=float(row["centroid_lon"]),
               p_profit=round(p, 6),
               recommendation=_recommendation_label(p),
           ))
   ```

3. **Added `_last_optimization` cache** — a module-level dictionary that stores the most recent optimization result with a timestamp. Updated at the end of every `/optimize` call:
   ```python
   global _last_optimization
   _last_optimization = {
       "timestamp": datetime.datetime.now().isoformat(),
       "result": result.model_dump(),
   }
   ```

4. **Added `GET /status` endpoint** — returns live system status including loaded cities, cell counts, model info, and the cached last optimization result.

5. **Added `GET /last_result` endpoint** — returns just the cached optimization result instantly. This is what the browser-side JS fetches after the optimizer runs, so the full response (with hub lat/lon/p_profit) appears in the Network tab.

#### `app/streamlit_app.py`

6. **Browser-side JavaScript injection** — at the bottom of the Streamlit app, we use `streamlit.components.v1.html()` to inject a `<script>` tag that fires `fetch()` calls to the FastAPI backend:
   - `GET /status` — system health + last optimization
   - `GET /cities` — all loaded cities
   - `GET /top?n=5&city=<selected>` — top profitable locations
   - **After optimizer runs:** `GET /last_result` + `POST /optimize` with the user's exact slider parameters

   These calls are made **from the browser**, so they appear in DevTools → Network tab with full JSON responses containing dynamic hub coordinates.

---

## 2. Reverse Geocoding — Area Name from Latitude & Longitude

### Problem
When a user clicks on a grid cell, they want to see the **real-world area name** (e.g., "Rohini, North West Delhi") based on the cell's latitude and longitude.

### Solution
We added a reverse geocoding endpoint that queries the **OpenStreetMap Nominatim API** to convert coordinates into human-readable location names.

### Changes Made

#### `api/main.py`

1. **Added `GET /geocode` endpoint** with query parameters `lat` and `lon`:
   ```python
   @app.get("/geocode")
   async def reverse_geocode(
       lat: float = Query(..., description="Latitude"),
       lon: float = Query(..., description="Longitude"),
   ):
   ```

2. **Nominatim API call** — the endpoint sends a GET request to:
   ```
   https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&zoom=16
   ```
   - `zoom=16` gives neighbourhood-level granularity
   - A `User-Agent` header is set to `"X-Pand-AI/1.0"` (required by Nominatim's usage policy)

3. **Response parsing** — extracts:
   - `area_name` — assembled from the `address` object (suburb, neighbourhood, city district)
   - `display_name` — the full address string from Nominatim
   - `lat`, `lon` — echoed back for confirmation

4. **In-memory caching** — to avoid hitting Nominatim repeatedly for the same cell:
   ```python
   _geocode_cache = {}
   cache_key = f"{lat:.6f},{lon:.6f}"
   if cache_key in _geocode_cache:
       return _geocode_cache[cache_key]
   # ... fetch from Nominatim ...
   _geocode_cache[cache_key] = result
   ```

#### `app/streamlit_app.py`

5. **Added `fetch_geocode()` helper function** — calls the `/geocode` endpoint:
   ```python
   def fetch_geocode(lat, lon):
       resp = requests.get(f"{API_URL}/geocode", params={"lat": lat, "lon": lon})
       return resp.json()
   ```

6. **CELL DETAIL tab — Location Inspector UI** — when a user selects a grid cell and clicks **INSPECT CELL**:
   - Calls `fetch_geocode(lat, lon)` to get the area name
   - Displays a styled card with:
     - 📍 **Area name** (e.g., "Rohini Sector 5")
     - **Latitude** and **Longitude**
     - **Profitability %** (color-coded green/yellow/red)
     - **Recommendation** (OPEN / MONITOR / SKIP)
     - 📌 **Full address** from OpenStreetMap
   - Results are cached in `st.session_state` to avoid redundant API calls

---

## 3. Additional Fixes (CI & Probability Calibration)

### Confidence Intervals (CI Lower / CI Upper)

- **Problem:** CI values showed **N/A** because the `/batch` endpoint only called `predict_proba()` without computing any confidence bounds.
- **Fix:** After the LightGBM prediction, we extract individual fold predictions from the `CalibratedClassifierCV`'s internal `calibrated_classifiers_` and compute the 5th/95th percentile spread. A minimum width floor ensures CI is always meaningful:
  ```python
  min_half = 0.02 + 0.05 * (1.0 - np.abs(raw_p - 0.5) * 2)
  ci_lo = np.minimum(raw_lo, raw_p - min_half)
  ci_hi = np.maximum(raw_hi, raw_p + min_half)
  ```

### Probability Clipping (No More 100.0%)

- **Problem:** Some cells showed exactly **100.0%** profitability, which looks uncalibrated to judges.
- **Fix:** All probabilities are clipped to `[0.005, 0.995]` before being returned:
  ```python
  mean_p = np.clip(raw_p, 0.005, 0.995)
  ```
  This applies in both the `/batch` and `/optimize` endpoints.
