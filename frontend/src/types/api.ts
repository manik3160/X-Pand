// ─── TypeScript types matching api/schemas.py ───

export interface LocationInput {
  lat: number
  lon: number
  grid_id?: string
}

export interface SHAPDriver {
  feature: string
  impact: number
}

export interface PredictionResult {
  grid_id: string
  lat: number
  lon: number
  p_profit: number
  ci_lower: number | null
  ci_upper: number | null
  estimated_cost: number | null
  recommendation: string
  shap_drivers: SHAPDriver[]
  is_cold_start: boolean
}

export interface PredictResponse {
  predictions: PredictionResult[]
  city: string
}

export interface HubDetail {
  grid_id: string
  lat: number
  lon: number
  p_profit: number
  recommendation: string
}

export interface OptimizeRequest {
  max_hubs: number
  min_separation_km: number
  min_prob_threshold: number
  city: string
}

export interface OptimizeResponse {
  selected_hubs: string[]
  hub_details: HubDetail[]
  total_score: number
  separation_constraint_met: boolean
  city: string
  eligible_cells: number
  total_cells: number
  model_used: string
  processing_time_seconds: number
  p_profit_range: [number, number]
}

export interface CityInfo {
  key: string
  name: string
  cell_count: number
  bbox: number[]
  map_center: { lat: number; lon: number }
  zoom: number
}

export interface CityListResponse {
  cities: CityInfo[]
}

export interface GridCell {
  grid_id: string
  lat: number
  lon: number
}

export interface TopLocation {
  rank: number
  grid_id: string
  lat: number
  lon: number
  p_profit: number
  recommendation: string
  top_drivers: SHAPDriver[]
}

export interface TopLocationsResponse {
  city: string
  top_locations: TopLocation[]
  total_found: number
}

export interface GeocodeResponse {
  lat: number
  lon: number
  area_name: string
  display_name: string
  address: Record<string, string>
}

export interface SystemStatus {
  status: string
  timestamp: string
  model: string
  cities_loaded: string[]
  cities_detail: Record<string, { cells: number; features: number }>
}

export interface SearchResult {
  lat: number
  lon: number
  display_name: string
  type: string
}
