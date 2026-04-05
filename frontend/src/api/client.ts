import axios from 'axios'
import type {
  CityInfo,
  GridCell,
  LocationInput,
  PredictResponse,
  PredictionResult,
  OptimizeRequest,
  OptimizeResponse,
  TopLocationsResponse,
  GeocodeResponse,
  SystemStatus,
  SearchResult,
} from '@/types/api'

// In dev, Vite proxy rewrites /api/* → http://localhost:8000/*
// In production, configure accordingly
const API_BASE = '/api'

const api = axios.create({
  baseURL: API_BASE,
  timeout: 300_000, // 5 min for large batch ops
  headers: { 'Content-Type': 'application/json' },
})

// ─── Cities ─────────────────────────────────────────
export async function fetchCities(): Promise<CityInfo[]> {
  const { data } = await api.get<{ cities: CityInfo[] }>('/cities')
  return data.cities
}

// ─── Grid ───────────────────────────────────────────
export async function fetchGrid(city: string): Promise<GridCell[]> {
  const { data } = await api.get<GridCell[]>('/grid', { params: { city } })
  return data
}

// ─── Batch Predict ──────────────────────────────────
export async function fetchBatchPredictions(
  locations: LocationInput[],
  city: string
): Promise<PredictionResult[]> {
  const { data } = await api.post<PredictResponse>('/batch', {
    locations,
    city,
  })
  return data.predictions
}

// ─── Single Predict (with SHAP) ─────────────────────
export async function fetchPrediction(
  location: LocationInput,
  city: string
): Promise<PredictionResult> {
  const { data } = await api.post<PredictResponse>('/predict', {
    locations: [location],
    city,
  })
  return data.predictions[0]
}

// ─── Optimize ───────────────────────────────────────
export async function runOptimizer(
  params: OptimizeRequest
): Promise<OptimizeResponse> {
  const { data } = await api.post<OptimizeResponse>('/optimize', params)
  return data
}

// ─── Top Locations ──────────────────────────────────
export async function fetchTopLocations(
  city: string,
  n: number = 10,
  minProb: number = 0.0
): Promise<TopLocationsResponse> {
  const { data } = await api.get<TopLocationsResponse>('/top', {
    params: { city, n, min_prob: minProb },
  })
  return data
}

// ─── Geocode ────────────────────────────────────────
export async function reverseGeocode(
  lat: number,
  lon: number
): Promise<GeocodeResponse> {
  const { data } = await api.get<GeocodeResponse>('/geocode', {
    params: { lat, lon },
  })
  return data
}

// ─── System Status ──────────────────────────────────
export async function fetchStatus(): Promise<SystemStatus> {
  const { data } = await api.get<SystemStatus>('/status')
  return data
}

// ─── Forward Geocoding Search ───────────────────────
export async function searchLocation(
  query: string,
  limit: number = 5
): Promise<SearchResult[]> {
  const { data } = await api.get<{ results: SearchResult[] }>('/search', {
    params: { q: query, limit },
  })
  return data.results
}
