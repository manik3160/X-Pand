import { MapPin, ChevronRight, Trophy } from 'lucide-react'
import { useApp } from '@/hooks/useApp'
import { useMapFocus } from '@/hooks/useMapFocus'
import { formatPercent, getProfitColorHex } from '@/lib/utils'
import { SkeletonRows } from '@/components/ui/Skeleton'
import EmptyState from '@/components/ui/EmptyState'
import ErrorBanner from '@/components/ui/ErrorBanner'

/**
 * List only — the GET /top fetch now lives in DashboardPage so that
 * Radix Tabs (which unmounts hidden tab content by default) doesn't
 * wipe this component's own effect every time the tab is hidden.
 */
export default function TopLocations() {
  const { topLocations, topLoading, topError, loadTopLocations, selectedCity } = useApp()
  const { focusCell } = useMapFocus()

  if (topLoading && topLocations.length === 0) {
    return <div className="p-4"><SkeletonRows count={4} /></div>
  }

  if (topError) {
    return <div className="p-4"><ErrorBanner message={topError} onRetry={() => loadTopLocations(selectedCity, 8)} /></div>
  }

  if (topLocations.length === 0) {
    return <EmptyState icon={Trophy} title="No locations yet" hint="Load a city to see its top-scoring cells." />
  }

  return (
    <div className="space-y-1 p-2">
      {topLocations.map((loc) => (
        <button
          key={loc.grid_id}
          onClick={() => focusCell(loc.grid_id)}
          className="w-full flex items-center justify-between p-2.5 rounded-xl hover:bg-[rgb(var(--surface-raised)/0.5)] transition-colors group text-left"
        >
          <div className="flex items-center gap-3 min-w-0">
            <div
              className="w-6 h-6 rounded-lg flex items-center justify-center text-xs font-bold text-accent font-heading flex-shrink-0"
              style={{ background: 'rgb(var(--accent) / 0.1)', border: '1px solid rgb(var(--accent) / 0.2)' }}
            >
              {loc.rank}
            </div>
            <div className="min-w-0">
              <div className="text-xs font-heading font-medium text-text-primary group-hover:text-accent transition-colors truncate">
                {loc.grid_id}
              </div>
              <div className="text-[10px] text-text-muted flex items-center gap-1">
                <MapPin className="w-2.5 h-2.5 flex-shrink-0" aria-hidden="true" />
                {loc.lat.toFixed(4)}, {loc.lon.toFixed(4)}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            <span className="text-sm font-bold font-heading tabular-nums" style={{ color: getProfitColorHex(loc.p_profit) }}>
              {formatPercent(loc.p_profit)}
            </span>
            <ChevronRight className="w-3.5 h-3.5 text-text-muted opacity-0 group-hover:opacity-100 transition-opacity" aria-hidden="true" />
          </div>
        </button>
      ))}
    </div>
  )
}
