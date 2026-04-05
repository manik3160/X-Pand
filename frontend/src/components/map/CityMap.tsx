import { useEffect, useMemo, useRef, useState, useCallback } from 'react'
import { MapContainer, TileLayer, useMap, Polygon, CircleMarker, Popup, Tooltip } from 'react-leaflet'
import type { Map as LeafletMap } from 'leaflet'
import { useApp } from '@/hooks/useApp'
import { getProfitColor, formatPercent } from '@/lib/utils'
import SearchBar from './SearchBar'

function FlyToCity({ center, zoom }: { center: [number, number]; zoom: number }) {
  const map = useMap()
  useEffect(() => {
    map.flyTo(center, zoom, { duration: 1.5 })
  }, [center, zoom, map])
  return null
}

function FlyToPoint({ target }: { target: { lat: number; lon: number; zoom: number } | null }) {
  const map = useMap()
  useEffect(() => {
    if (target) {
      map.flyTo([target.lat, target.lon], target.zoom, { duration: 1.2 })
    }
  }, [target, map])
  return null
}

function GridLayer() {
  const { predictions, setSelectedCellId, selectedCellId } = useApp()

  const cellPolygons = useMemo(() => {
    const HALF_SIZE_LAT = 0.00225
    const HALF_SIZE_LON = 0.0025

    return predictions.map(p => {
      const corners: [number, number][] = [
        [p.lat - HALF_SIZE_LAT, p.lon - HALF_SIZE_LON],
        [p.lat - HALF_SIZE_LAT, p.lon + HALF_SIZE_LON],
        [p.lat + HALF_SIZE_LAT, p.lon + HALF_SIZE_LON],
        [p.lat + HALF_SIZE_LAT, p.lon - HALF_SIZE_LON],
      ]
      const [r, g, b] = getProfitColor(p.p_profit)
      return {
        ...p,
        corners,
        color: `rgb(${r},${g},${b})`,
        fillColor: `rgba(${r},${g},${b},0.75)`,
        isSelected: p.grid_id === selectedCellId,
      }
    })
  }, [predictions, selectedCellId])

  return (
    <>
      {cellPolygons.map(cell => (
        <Polygon
          key={cell.grid_id}
          positions={cell.corners}
          pathOptions={{
            color: cell.isSelected ? '#0099ff' : cell.color,
            fillColor: cell.fillColor,
            fillOpacity: cell.isSelected ? 0.95 : 0.75,
            weight: cell.isSelected ? 3 : 0.5,
          }}
          eventHandlers={{
            click: () => setSelectedCellId(cell.grid_id),
          }}
        >
          <Tooltip sticky>
            <div className="text-xs font-mono" style={{ color: '#ffffff' }}>
              <strong>{cell.grid_id}</strong><br />
              P(profit): {formatPercent(cell.p_profit)}<br />
              Status: <span style={{
                color: cell.recommendation === 'open' ? '#00ff88' :
                  cell.recommendation === 'monitor' ? '#ffaa00' : '#ff4444'
              }}>{cell.recommendation.toUpperCase()}</span>
            </div>
          </Tooltip>
        </Polygon>
      ))}
    </>
  )
}

function HubMarkers() {
  const { optimizeResult } = useApp()
  if (!optimizeResult) return null

  return (
    <>
      {optimizeResult.hub_details.map(hub => (
        <CircleMarker
          key={hub.grid_id}
          center={[hub.lat, hub.lon]}
          radius={12}
          pathOptions={{
            color: '#0099ff',
            fillColor: '#0099ff',
            fillOpacity: 0.3,
            weight: 3,
          }}
        >
          <Popup>
            <div className="text-xs" style={{ color: '#ffffff' }}>
              <strong style={{ color: '#0099ff' }}>🏢 Optimal Hub</strong><br />
              Grid: {hub.grid_id}<br />
              P(profit): {formatPercent(hub.p_profit)}<br />
              Status: {hub.recommendation.toUpperCase()}
            </div>
          </Popup>
        </CircleMarker>
      ))}
    </>
  )
}

function MapLegend() {
  const { optimizeResult } = useApp()

  return (
    <div
      className="absolute bottom-6 left-6 z-[1000] flex items-center gap-4 px-4 py-2.5 rounded-full"
      style={{
        background: 'rgba(255, 255, 255, 0.04)',
        border: '1px solid rgba(255, 255, 255, 0.08)',
        backdropFilter: 'blur(20px)',
      }}
    >
      <div className="flex items-center gap-1.5">
        <div className="w-2.5 h-2.5 rounded-full" style={{ background: '#00ff88' }} />
        <span className="text-[11px] text-text-secondary">High &gt;0.7</span>
      </div>
      <div className="flex items-center gap-1.5">
        <div className="w-2.5 h-2.5 rounded-full" style={{ background: '#ffaa00' }} />
        <span className="text-[11px] text-text-secondary">Monitor 0.4–0.7</span>
      </div>
      <div className="flex items-center gap-1.5">
        <div className="w-2.5 h-2.5 rounded-full" style={{ background: '#ff4444' }} />
        <span className="text-[11px] text-text-secondary">Skip &lt;0.4</span>
      </div>
      {optimizeResult && (
        <div className="flex items-center gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full" style={{ background: '#0099ff', border: '2px solid #0099ff' }} />
          <span className="text-[11px] text-text-secondary">Selected hub</span>
        </div>
      )}
    </div>
  )
}

export default function CityMap() {
  const { cities, selectedCity, predictions, predictionsLoading, setSelectedCellId } = useApp()
  const mapRef = useRef<LeafletMap | null>(null)
  const [flyTarget, setFlyTarget] = useState<{ lat: number; lon: number; zoom: number } | null>(null)

  const cityInfo = cities.find(c => c.key === selectedCity)
  const center: [number, number] = cityInfo
    ? [cityInfo.map_center.lat, cityInfo.map_center.lon]
    : [28.65, 77.10]
  const zoom = cityInfo?.zoom || 10.2

  const handleSearchSelect = useCallback((lat: number, lon: number, _displayName: string) => {
    // 1. Fly map to the searched location
    setFlyTarget({ lat, lon, zoom: 15 })

    // 2. Find the nearest grid cell within ~2km
    if (predictions.length === 0) return

    let bestDist = Infinity
    let bestId: string | null = null

    for (const p of predictions) {
      const dLat = (p.lat - lat) * 111.32 // ~km per degree lat
      const dLon = (p.lon - lon) * 111.32 * Math.cos(lat * Math.PI / 180)
      const dist = Math.sqrt(dLat * dLat + dLon * dLon)
      if (dist < bestDist) {
        bestDist = dist
        bestId = p.grid_id
      }
    }

    // Only auto-select if within 2km — otherwise it's out of grid bounds
    if (bestId && bestDist < 2) {
      setSelectedCellId(bestId)
    }
  }, [predictions, setSelectedCellId])

  return (
    <div
      className="relative w-full h-full overflow-hidden"
      style={{
        borderRadius: '16px',
        border: '1px solid rgba(255, 255, 255, 0.08)',
      }}
    >
      {/* Search Bar */}
      <SearchBar onSelect={handleSearchSelect} />

      {/* Loading overlay */}
      {predictionsLoading && (
        <div
          className="absolute inset-0 z-[1000] flex items-center justify-center"
          style={{ background: 'rgba(4, 4, 6, 0.85)', backdropFilter: 'blur(8px)' }}
        >
          <div className="flex flex-col items-center gap-3">
            <div
              className="w-5 h-5 rounded-full animate-spin-slow"
              style={{ border: '2px solid rgba(255,255,255,0.1)', borderTopColor: '#00ff88' }}
            />
            <span className="text-sm text-text-secondary">Scoring grid cells...</span>
          </div>
        </div>
      )}

      <MapContainer
        center={center}
        zoom={zoom}
        className="w-full h-full"
        zoomControl={false}
        ref={mapRef}
        style={{ background: '#040406' }}
      >
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          attribution='&copy; <a href="https://carto.com/">CARTO</a>'
          maxZoom={19}
        />
        <FlyToCity center={center} zoom={zoom} />
        <FlyToPoint target={flyTarget} />
        {predictions.length > 0 && <GridLayer />}
        <HubMarkers />
      </MapContainer>

      <MapLegend />

      {/* Cell count badge */}
      <div
        className="absolute top-4 right-4 z-[1000] px-3 py-1.5 rounded-full text-[11px] text-text-secondary"
        style={{
          background: 'rgba(255, 255, 255, 0.04)',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          backdropFilter: 'blur(12px)',
        }}
      >
        {predictions.length.toLocaleString()} cells loaded
      </div>
    </div>
  )
}

