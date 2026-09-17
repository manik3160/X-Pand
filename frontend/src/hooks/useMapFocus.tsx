import { createContext, useContext, useState, useCallback, type ReactNode } from 'react'
import { useApp } from './useApp'

export interface FlyTarget {
  lat: number
  lon: number
  zoom: number
}

interface MapFocusState {
  flyTarget: FlyTarget | null
  /** Select a known grid cell and fly the map to it. */
  focusCell: (gridId: string) => void
  /** Fly to an arbitrary point (e.g. a search result) without changing selection. */
  focusPoint: (lat: number, lon: number, zoom?: number) => void
}

const MapFocusContext = createContext<MapFocusState | null>(null)

/**
 * Shared "fly the map here" state so the search bar, top-location rows and
 * optimizer hub rows all move the map the same way CityMap's search
 * fly-to already did — instead of only search doing it.
 */
export function MapFocusProvider({ children }: { children: ReactNode }) {
  const { predictions, setSelectedCellId } = useApp()
  const [flyTarget, setFlyTarget] = useState<FlyTarget | null>(null)

  const focusPoint = useCallback((lat: number, lon: number, zoom = 15) => {
    setFlyTarget({ lat, lon, zoom })
  }, [])

  const focusCell = useCallback((gridId: string) => {
    setSelectedCellId(gridId)
    const p = predictions.find(pr => pr.grid_id === gridId)
    if (p) setFlyTarget({ lat: p.lat, lon: p.lon, zoom: 15 })
  }, [predictions, setSelectedCellId])

  return (
    <MapFocusContext.Provider value={{ flyTarget, focusCell, focusPoint }}>
      {children}
    </MapFocusContext.Provider>
  )
}

export function useMapFocus() {
  const ctx = useContext(MapFocusContext)
  if (!ctx) throw new Error('useMapFocus must be used within MapFocusProvider')
  return ctx
}
