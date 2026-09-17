import type { PredictionResult } from '@/types/api'

/**
 * Nearest prediction cell to a lat/lon, within maxKm. Moved out of
 * CityMap.handleSearchSelect unchanged so search and the map-focus hook
 * can share it.
 */
export function findNearestCell(
  predictions: PredictionResult[],
  lat: number,
  lon: number,
  maxKm = 2
): string | null {
  if (predictions.length === 0) return null

  let bestDist = Infinity
  let bestId: string | null = null

  for (const p of predictions) {
    const dLat = (p.lat - lat) * 111.32 // ~km per degree lat
    const dLon = (p.lon - lon) * 111.32 * Math.cos((lat * Math.PI) / 180)
    const dist = Math.sqrt(dLat * dLat + dLon * dLon)
    if (dist < bestDist) {
      bestDist = dist
      bestId = p.grid_id
    }
  }

  return bestId && bestDist < maxKm ? bestId : null
}
