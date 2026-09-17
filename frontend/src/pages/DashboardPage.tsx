import { useEffect, useState } from 'react'
import { useApp } from '@/hooks/useApp'
import { MapFocusProvider } from '@/hooks/useMapFocus'
import { useIsMobile } from '@/hooks/useMediaQuery'
import CityMap, { MapLegend } from '@/components/map/CityMap'
import TopBar from '@/components/dashboard/TopBar'
import ControlsCard, { OptimizerControlsContent } from '@/components/dashboard/ControlsCard'
import ResultsCard, { ResultsContent } from '@/components/dashboard/ResultsCard'
import CellDetail from '@/components/dashboard/CellDetail'
import SideSheet from '@/components/ui/SideSheet'
import BottomSheet, { type SheetSnap } from '@/components/ui/BottomSheet'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/Tabs'
import ErrorBanner from '@/components/ui/ErrorBanner'
import { X } from 'lucide-react'

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

type MobileTab = 'controls' | 'results' | 'detail'

/** Controls / Results / Detail in one draggable sheet, replacing the desktop floating panels + side sheet below md. */
function MobileSheet() {
  const { selectedCellId, setSelectedCellId } = useApp()
  const [snap, setSnap] = useState<SheetSnap>('peek')
  const [tab, setTab] = useState<MobileTab>('results')

  useEffect(() => {
    if (selectedCellId) {
      setTab('detail')
      setSnap(s => (s === 'peek' ? 'half' : s))
    } else if (tab === 'detail') {
      setTab('results')
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedCellId])

  return (
    <BottomSheet
      snap={snap}
      onSnapChange={setSnap}
      header={
        <Tabs value={tab} onValueChange={v => setTab(v as MobileTab)} className="w-full">
          <TabsList className="pt-0">
            <TabsTrigger value="controls">Controls</TabsTrigger>
            <TabsTrigger value="results">Results</TabsTrigger>
            {selectedCellId && <TabsTrigger value="detail">Detail</TabsTrigger>}
          </TabsList>
        </Tabs>
      }
    >
      <Tabs value={tab} onValueChange={v => setTab(v as MobileTab)}>
        <TabsContent value="controls" className="p-4">
          <OptimizerControlsContent />
        </TabsContent>
        <TabsContent value="results" className="flex flex-col">
          <ResultsContent />
        </TabsContent>
        {selectedCellId && (
          <TabsContent value="detail">
            <div className="flex items-center justify-between px-5 pt-1 pb-2">
              <span className="text-sm font-heading font-semibold text-text-primary">{selectedCellId}</span>
              <button
                onClick={() => setSelectedCellId(null)}
                aria-label="Close details"
                className="p-1.5 rounded-lg hover:bg-[rgb(var(--surface-raised)/0.6)] transition-colors"
              >
                <X className="w-4 h-4 text-text-muted" />
              </button>
            </div>
            <CellDetail key={selectedCellId} gridId={selectedCellId} />
          </TabsContent>
        )}
      </Tabs>
    </BottomSheet>
  )
}

function DashboardContent() {
  const { selectedCellId, setSelectedCellId, predictionsError, loadCityData, selectedCity } = useApp()
  const isMobile = useIsMobile()
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

        {!isMobile && (
          <div className="flex flex-1 min-h-0 px-3 pb-3 justify-between items-start gap-3">
            <div className="pointer-events-auto">
              <ControlsCard />
            </div>
            <div className="pointer-events-auto">
              <ResultsCard />
            </div>
          </div>
        )}

        {!isMobile && (
          <div className="absolute bottom-3 left-3 pointer-events-none">
            <MapLegend />
          </div>
        )}
      </div>

      {isMobile ? (
        <MobileSheet />
      ) : (
        <SideSheet
          open={!!selectedCellId}
          onClose={() => setSelectedCellId(null)}
          title={selectedCellId ?? ''}
        >
          {selectedCellId && <CellDetail key={selectedCellId} gridId={selectedCellId} />}
        </SideSheet>
      )}
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
