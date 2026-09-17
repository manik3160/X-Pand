import { useEffect, useMemo, useRef } from 'react'
import { MapContainer, TileLayer, useMap, CircleMarker, Popup } from 'react-leaflet'
import L from 'leaflet'
import type { Map as LeafletMap } from 'leaflet'
import { useApp } from '@/hooks/useApp'
import { useMapFocus } from '@/hooks/useMapFocus'
import { formatPercent } from '@/lib/utils'
import { theme, profitColorHex, profitFillOpacity } from '@/lib/theme'
import { basemap } from '@/lib/basemap'
import Spinner from '@/components/ui/Spinner'

function FlyToCity({ lat, lon, zoom }: { lat: number; lon: number; zoom: number }) {
  const map = useMap()
  useEffect(() => {
    map.flyTo([lat, lon], zoom, { duration: 1.5 })
    // Only lat/lon/zoom (primitives) are in the dep array — passing a
    // freshly-built [lat, lon] array here instead would re-trigger the
    // fly-to on every render.
  }, [lat, lon, zoom, map])
  return null
}

function FlyToPoint() {
  const map = useMap()
  const { flyTarget } = useMapFocus()
  useEffect(() => {
    if (flyTarget) {
      map.flyTo([flyTarget.lat, flyTarget.lon], flyTarget.zoom, { duration: 1.2 })
    }
  }, [flyTarget, map])
  return null
}

const HALF_SIZE_LAT = 0.00225
const HALF_SIZE_LON = 0.0025

/**
 * Draws the prediction grid with plain Leaflet layers on a shared canvas
 * renderer instead of one react-leaflet <Polygon>/<Tooltip> per cell.
 * Delhi NCR alone is ~3,300 cells — mounting that many React component
 * instances (each with its own hooks and event handlers) is what made
 * panning janky; a canvas renderer draws them as pixels on one <canvas>.
 */
function GridLayer() {
  const map = useMap()
  const { predictions, setSelectedCellId, selectedCellId } = useApp()
  const renderer = useMemo(() => L.canvas({ padding: 0.5 }), [])
  const groupRef = useRef<L.LayerGroup | null>(null)
  const rectsRef = useRef<Map<string, L.Rectangle>>(new Map())
  const selectedIdRef = useRef<string | null>(null)

  useEffect(() => {
    const group = L.layerGroup().addTo(map)
    groupRef.current = group
    return () => {
      group.remove()
      groupRef.current = null
    }
  }, [map])

  useEffect(() => {
    const group = groupRef.current
    if (!group) return
    group.clearLayers()
    rectsRef.current.clear()
    selectedIdRef.current = null

    for (const p of predictions) {
      const bounds: L.LatLngBoundsExpression = [
        [p.lat - HALF_SIZE_LAT, p.lon - HALF_SIZE_LON],
        [p.lat + HALF_SIZE_LAT, p.lon + HALF_SIZE_LON],
      ]
      const color = profitColorHex(p.p_profit)
      const rect = L.rectangle(bounds, {
        renderer,
        color,
        fillColor: color,
        fillOpacity: profitFillOpacity(p.p_profit),
        weight: 0,
      })
      rect.bindTooltip(
        `<div style="font-family:'Space Grotesk',monospace;font-size:12px">
          <strong>${p.grid_id}</strong><br/>
          P(profit): ${formatPercent(p.p_profit)}<br/>
          Status: <span style="color:${
            p.recommendation === 'open' ? theme.profitHigh : p.recommendation === 'monitor' ? theme.profitMid : theme.profitLow
          }">${p.recommendation.toUpperCase()}</span>
        </div>`,
        { sticky: true }
      )
      rect.on('click', () => setSelectedCellId(p.grid_id))
      rect.addTo(group)
      rectsRef.current.set(p.grid_id, rect)
    }
    // predictions is the only thing that should rebuild the whole layer;
    // selection is handled by the effect below without a rebuild.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [predictions, renderer, map])

  // Restyle only the previously- and newly-selected cell, not the whole grid.
  useEffect(() => {
    const rects = rectsRef.current
    const prevId = selectedIdRef.current
    if (prevId && prevId !== selectedCellId) {
      const prevRect = rects.get(prevId)
      const prevPred = predictions.find(p => p.grid_id === prevId)
      if (prevRect && prevPred) {
        const color = profitColorHex(prevPred.p_profit)
        prevRect.setStyle({ color, fillColor: color, fillOpacity: profitFillOpacity(prevPred.p_profit), weight: 0 })
      }
    }
    if (selectedCellId) {
      const rect = rects.get(selectedCellId)
      const pred = predictions.find(p => p.grid_id === selectedCellId)
      if (rect && pred) {
        const color = profitColorHex(pred.p_profit)
        rect.setStyle({ color: theme.accent, fillColor: color, fillOpacity: 0.9, weight: 2 })
        rect.bringToFront()
      }
    }
    selectedIdRef.current = selectedCellId
  }, [selectedCellId, predictions])

  return null
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
            color: theme.accent,
            fillColor: theme.accent,
            fillOpacity: 0.3,
            weight: 3,
          }}
        >
          <Popup>
            <div className="text-xs" style={{ color: theme.text }}>
              <strong style={{ color: theme.accent }}>Optimal hub</strong><br />
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

export function MapLegend() {
  const { optimizeResult } = useApp()

  return (
    <div className="pointer-events-auto panel flex items-center gap-4 px-4 py-2.5 rounded-full">
      <div className="flex items-center gap-1.5">
        <div className="w-2.5 h-2.5 rounded-full" style={{ background: theme.profitHigh }} />
        <span className="text-[11px] text-text-secondary">High &gt;0.7</span>
      </div>
      <div className="flex items-center gap-1.5">
        <div className="w-2.5 h-2.5 rounded-full" style={{ background: theme.profitMid }} />
        <span className="text-[11px] text-text-secondary">Monitor 0.4–0.7</span>
      </div>
      <div className="flex items-center gap-1.5">
        <div className="w-2.5 h-2.5 rounded-full" style={{ background: theme.profitLow }} />
        <span className="text-[11px] text-text-secondary">Skip &lt;0.4</span>
      </div>
      {optimizeResult && (
        <div className="flex items-center gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full" style={{ background: theme.accent, border: `2px solid ${theme.accent}` }} />
          <span className="text-[11px] text-text-secondary">Selected hub</span>
        </div>
      )}
    </div>
  )
}

/** Full-bleed map — panels float over it (see DashboardPage), it is never resized by them. */
export default function CityMap() {
  const { cities, selectedCity, predictions, predictionsLoading, predictionsError } = useApp()
  const mapRef = useRef<LeafletMap | null>(null)

  const cityInfo = cities.find(c => c.key === selectedCity)
  const centerLat = cityInfo?.map_center.lat ?? 28.65
  const centerLon = cityInfo?.map_center.lon ?? 77.10
  const zoom = cityInfo?.zoom || 10.2

  return (
    <div className="fixed inset-0">
      {predictionsLoading && (
        <div className="absolute inset-0 z-[1000] flex items-center justify-center" style={{ background: 'rgb(var(--bg) / 0.85)', backdropFilter: 'blur(8px)' }}>
          <div className="flex flex-col items-center gap-3 text-center px-6">
            <Spinner />
            <span className="text-sm text-text-secondary">Scoring grid cells…</span>
            <span className="text-xs text-text-muted max-w-[220px]">
              First load can take up to a minute while the model server wakes up.
            </span>
          </div>
        </div>
      )}

      {!predictionsLoading && predictionsError && predictions.length === 0 && (
        <div className="absolute inset-0 z-[1000] flex items-center justify-center" style={{ background: 'rgb(var(--bg) / 0.85)', backdropFilter: 'blur(8px)' }}>
          <div className="flex flex-col items-center gap-2 text-center px-6 max-w-[280px]">
            <span className="text-sm" style={{ color: theme.profitLow }}>{predictionsError}</span>
          </div>
        </div>
      )}

      <MapContainer
        center={[centerLat, centerLon]}
        zoom={zoom}
        className="w-full h-full"
        zoomControl={false}
        preferCanvas
        ref={mapRef}
        style={{ background: theme.bg }}
      >
        <TileLayer
          url={basemap.url}
          attribution={basemap.attribution}
          maxNativeZoom={basemap.maxNativeZoom}
          maxZoom={basemap.maxZoom}
        />
        <TileLayer
          url={basemap.labelsUrl}
          maxNativeZoom={basemap.maxNativeZoom}
          maxZoom={basemap.maxZoom}
        />
        <FlyToCity lat={centerLat} lon={centerLon} zoom={zoom} />
        <FlyToPoint />
        {predictions.length > 0 && <GridLayer />}
        <HubMarkers />
      </MapContainer>
    </div>
  )
}
