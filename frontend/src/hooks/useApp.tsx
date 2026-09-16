import { createContext, useContext, useState, useCallback, useEffect, useRef, type ReactNode } from 'react'
import type { CityInfo, PredictionResult, OptimizeResponse, GridCell } from '@/types/api'
import { fetchCities, fetchGrid, fetchBatchPredictions, runOptimizer, fetchTopLocations } from '@/api/client'
import type { OptimizeRequest, TopLocation } from '@/types/api'

interface AppState {
  // Cities
  cities: CityInfo[]
  selectedCity: string
  citiesLoading: boolean
  citiesError: string | null

  // Grid & Predictions
  gridCells: GridCell[]
  predictions: PredictionResult[]
  predictionsLoading: boolean
  predictionsError: string | null

  // Optimizer
  optimizeResult: OptimizeResponse | null
  optimizing: boolean
  optimizeError: string | null

  // Top locations
  topLocations: TopLocation[]
  topLoading: boolean
  topError: string | null

  // Selected cell
  selectedCellId: string | null

  // Actions
  setSelectedCity: (city: string) => void
  setSelectedCellId: (id: string | null) => void
  loadCityData: (city: string) => Promise<void>
  runOptimize: (params: OptimizeRequest) => Promise<void>
  loadTopLocations: (city: string, n?: number) => Promise<void>
}

const AppContext = createContext<AppState | null>(null)

function errorMessage(err: unknown, fallback: string): string {
  if (err && typeof err === 'object' && 'message' in err && typeof (err as { message?: unknown }).message === 'string') {
    return (err as { message: string }).message || fallback
  }
  return fallback
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [cities, setCities] = useState<CityInfo[]>([])
  const [selectedCity, setSelectedCity] = useState('delhi')
  const [citiesLoading, setCitiesLoading] = useState(true)
  const [citiesError, setCitiesError] = useState<string | null>(null)

  const [gridCells, setGridCells] = useState<GridCell[]>([])
  const [predictions, setPredictions] = useState<PredictionResult[]>([])
  const [predictionsLoading, setPredictionsLoading] = useState(false)
  const [predictionsError, setPredictionsError] = useState<string | null>(null)

  const [optimizeResult, setOptimizeResult] = useState<OptimizeResponse | null>(null)
  const [optimizing, setOptimizing] = useState(false)
  const [optimizeError, setOptimizeError] = useState<string | null>(null)

  const [topLocations, setTopLocations] = useState<TopLocation[]>([])
  const [topLoading, setTopLoading] = useState(false)
  const [topError, setTopError] = useState<string | null>(null)

  const [selectedCellId, setSelectedCellId] = useState<string | null>(null)

  // Stale-response guards: client.ts has no AbortSignal support, so we
  // can't cancel in-flight requests — instead we tag each request and
  // drop the response if a newer request has since started.
  const cityReqId = useRef(0)
  const topReqId = useRef(0)

  // Load cities on mount
  useEffect(() => {
    setCitiesLoading(true)
    setCitiesError(null)
    fetchCities()
      .then(setCities)
      .catch(err => setCitiesError(errorMessage(err, 'Could not load cities.')))
      .finally(() => setCitiesLoading(false))
  }, [])

  const loadCityData = useCallback(async (city: string) => {
    const reqId = ++cityReqId.current
    setPredictionsLoading(true)
    setPredictionsError(null)
    setOptimizeResult(null)
    setOptimizeError(null)
    setSelectedCellId(null)
    try {
      const grid = await fetchGrid(city)
      if (cityReqId.current !== reqId) return

      const locations = grid.map(c => ({
        lat: c.lat,
        lon: c.lon,
        grid_id: c.grid_id,
      }))
      const preds = await fetchBatchPredictions(locations, city)
      if (cityReqId.current !== reqId) return

      setGridCells(grid)
      setPredictions(preds)
    } catch (err) {
      if (cityReqId.current !== reqId) return
      console.error('Failed to load city data:', err)
      setPredictionsError(errorMessage(err, 'Could not load this city’s predictions.'))
    } finally {
      if (cityReqId.current === reqId) setPredictionsLoading(false)
    }
  }, [])

  const runOptimize = useCallback(async (params: OptimizeRequest) => {
    setOptimizing(true)
    setOptimizeError(null)
    try {
      const result = await runOptimizer(params)
      setOptimizeResult(result)
    } catch (err) {
      console.error('Optimizer failed:', err)
      setOptimizeError(errorMessage(err, 'The optimizer failed to run.'))
    } finally {
      setOptimizing(false)
    }
  }, [])

  const loadTopLocations = useCallback(async (city: string, n: number = 10) => {
    const reqId = ++topReqId.current
    setTopLoading(true)
    setTopError(null)
    try {
      const data = await fetchTopLocations(city, n)
      if (topReqId.current !== reqId) return
      setTopLocations(data.top_locations)
    } catch (err) {
      if (topReqId.current !== reqId) return
      console.error('Failed to load top locations:', err)
      setTopError(errorMessage(err, 'Could not load top locations.'))
    } finally {
      if (topReqId.current === reqId) setTopLoading(false)
    }
  }, [])

  return (
    <AppContext.Provider
      value={{
        cities,
        selectedCity,
        citiesLoading,
        citiesError,
        gridCells,
        predictions,
        predictionsLoading,
        predictionsError,
        optimizeResult,
        optimizing,
        optimizeError,
        topLocations,
        topLoading,
        topError,
        selectedCellId,
        setSelectedCity,
        setSelectedCellId,
        loadCityData,
        runOptimize,
        loadTopLocations,
      }}
    >
      {children}
    </AppContext.Provider>
  )
}

export function useApp() {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error('useApp must be used within AppProvider')
  return ctx
}
