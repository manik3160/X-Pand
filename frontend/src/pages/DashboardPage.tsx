import { useEffect } from 'react'
import { useApp } from '@/hooks/useApp'
import { MapFocusProvider } from '@/hooks/useMapFocus'
import CityMap, { MapLegend } from '@/components/map/CityMap'
import TopBar from '@/components/dashboard/TopBar'
import ControlsCard from '@/components/dashboard/ControlsCard'
import ResultsCard from '@/components/dashboard/ResultsCard'
import CellDetail from '@/components/dashboard/CellDetail'
import SideSheet from '@/components/ui/SideSheet'
import ErrorBanner from '@/components/ui/ErrorBanner'

/**
 * Page-level data effects. Two of these used to live inside components
 * that Radix Tabs would unmount (TopLocations) or that were about to be
 * deleted (Sidebar) — both need to survive regardless of which tab or
 * panel is currently visible.
 */
function useDashboardEffects() {
  const {
    cities, selectedCity, predictions, predictionsLoading, predictionsError, loadCityData,
    loadTopLocations,
  } = useApp()

  // Initial (and only) city load. Guarded by predictionsError so a failed
  // load shows a retry banner instead of retrying forever.
  useEffect(() => {
    if (cities.length > 0 && predictions.length === 0 && !predictionsLoading && !predictionsError) {
      loadCityData(selectedCity)
    }
  }, [cities, selectedCity, predictions.length, predictionsLoading, predictionsError, loadCityData])

  useEffect(() => {
    if (predictions.length > 0) {
      loadTopLocations(selectedCity, 8)
    }
  }, [selectedCity, predictions.length, loadTopLocations])
}

function DashboardContent() {
  const { selectedCellId, setSelectedCellId, predictionsError, loadCityData, selectedCity } = useApp()
  useDashboardEffects()

  return (
    <div className="h-screen w-screen overflow-hidden relative" style={{ background: 'rgb(var(--bg))' }}>
      <CityMap />

      {/* Floating chrome — pointer-events re-enabled per element so clicks pass through to the map elsewhere */}
      <div className="pointer-events-none absolute inset-0 flex flex-col">
        <div className="p-3 pointer-events-none">
          <TopBar />
        </div>

        {predictionsError && (
          <div className="px-3 pointer-events-auto max-w-[420px]">
            <ErrorBanner message={predictionsError} onRetry={() => loadCityData(selectedCity)} className="panel" />
          </div>
        )}

        {/* Desktop: floating side panels */}
        <div className="hidden md:flex flex-1 min-h-0 px-3 pb-3 justify-between items-start gap-3">
          <div className="pointer-events-auto">
            <ControlsCard />
          </div>
          <div className="pointer-events-auto">
            <ResultsCard />
          </div>
        </div>

        {/* Mobile fallback: stacked, scrollable — replaced by a bottom sheet in a later pass */}
        <div className="md:hidden flex-1 min-h-0 overflow-y-auto px-3 pb-3 space-y-3 pointer-events-auto">
          <ControlsCard />
          <ResultsCard />
        </div>

        <div className="hidden md:block absolute bottom-3 left-3 pointer-events-none">
          <MapLegend />
        </div>
      </div>

      <SideSheet
        open={!!selectedCellId}
        onClose={() => setSelectedCellId(null)}
        title={selectedCellId ?? ''}
      >
        {selectedCellId && <CellDetail key={selectedCellId} gridId={selectedCellId} />}
      </SideSheet>
    </div>
  )
}

export default function DashboardPage() {
  return (
    <MapFocusProvider>
      <DashboardContent />
    </MapFocusProvider>
  )
}
