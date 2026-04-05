import Sidebar from '@/components/layout/Sidebar'
import CityMap from '@/components/map/CityMap'
import MetricCards from '@/components/dashboard/MetricCards'
import CellDetail from '@/components/dashboard/CellDetail'
import TopLocations from '@/components/dashboard/TopLocations'
import OptimizerPanel from '@/components/dashboard/OptimizerPanel'
import { useApp } from '@/hooks/useApp'

export default function DashboardPage() {
  const { predictions, selectedCellId, selectedCity, cities } = useApp()
  const cityInfo = cities.find(c => c.key === selectedCity)

  return (
    <div className="h-screen w-screen flex overflow-hidden" style={{ background: '#040406' }}>
      {/* Sidebar */}
      <Sidebar />

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header Bar */}
        <header
          className="h-14 flex items-center justify-between px-6 flex-shrink-0"
          style={{
            background: 'rgba(4, 4, 6, 0.8)',
            backdropFilter: 'blur(20px)',
            borderBottom: '1px solid rgba(255, 255, 255, 0.06)',
          }}
        >
          <div className="flex items-center gap-4">
            <h1 className="text-lg font-semibold text-white font-heading tracking-tight">
              Profitability Prediction Map
            </h1>
            <span className="text-[13px] text-text-muted uppercase tracking-wider">
              {cityInfo?.name || selectedCity} — 500m Grid
            </span>
          </div>
          <div className="flex items-center gap-4 text-xs">
            <div className="flex items-center gap-2 px-3 py-1 rounded-full" style={{ background: 'rgba(0,255,136,0.06)', border: '1px solid rgba(0,255,136,0.15)' }}>
              <div className="live-dot-sm" />
              <span className="text-accent font-semibold">LIVE SCORED</span>
            </div>
            <span className="text-text-muted font-heading tabular-nums">{predictions.length.toLocaleString()} cells</span>
          </div>
        </header>

        {/* Content Area */}
        <div className="flex-1 flex overflow-hidden">
          {/* Map + Dashboard */}
          <div className="flex-1 flex flex-col overflow-hidden px-3 pt-2 pb-3 gap-2">
            {/* Metric Cards */}
            {predictions.length > 0 && <MetricCards />}

            {/* Map + Right Panel */}
            <div className="flex-1 flex gap-3 min-h-0">
              {/* Map */}
              <div className="flex-1 min-w-0">
                <CityMap />
              </div>

              {/* Right info column */}
              <div className="w-64 flex flex-col gap-3 overflow-y-auto">
                <OptimizerPanel />
                <TopLocations />
              </div>
            </div>
          </div>

          {/* Cell Detail Panel */}
          {selectedCellId && <CellDetail />}
        </div>
      </div>
    </div>
  )
}
