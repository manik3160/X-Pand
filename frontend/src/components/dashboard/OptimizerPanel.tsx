import { MapPin, Target } from 'lucide-react'
import { useApp } from '@/hooks/useApp'
import { useMapFocus } from '@/hooks/useMapFocus'
import { formatPercent, getProfitColorHex } from '@/lib/utils'
import EmptyState from '@/components/ui/EmptyState'

/** Hub list only — the run summary (score / time / eligible) lives in ResultsCard so it's shown once, not duplicated. */
export default function OptimizerPanel() {
  const { optimizeResult } = useApp()
  const { focusCell } = useMapFocus()

  if (!optimizeResult || optimizeResult.hub_details.length === 0) {
    return <EmptyState icon={Target} title="No hubs yet" hint="Run the optimizer to see suggested hub locations here." />
  }

  return (
    <div className="space-y-1 p-2">
      {optimizeResult.hub_details.map((hub, i) => (
        <button
          key={hub.grid_id}
          onClick={() => focusCell(hub.grid_id)}
          className="w-full flex items-center justify-between p-2.5 rounded-xl transition-colors text-left hover:bg-[rgb(var(--accent)/0.08)]"
          style={{ background: 'rgb(var(--accent) / 0.04)', border: '1px solid rgb(var(--accent) / 0.12)' }}
        >
          <div className="flex items-center gap-2 min-w-0">
            <div
              className="w-5 h-5 rounded-lg flex items-center justify-center text-[10px] font-bold text-accent font-heading flex-shrink-0"
              style={{ background: 'rgb(var(--accent) / 0.12)' }}
            >
              {i + 1}
            </div>
            <div className="min-w-0">
              <div className="text-xs font-heading font-medium text-text-primary truncate">{hub.grid_id}</div>
              <div className="text-[10px] text-text-muted flex items-center gap-1">
                <MapPin className="w-2.5 h-2.5 flex-shrink-0" aria-hidden="true" />
                {hub.lat.toFixed(4)}, {hub.lon.toFixed(4)}
              </div>
            </div>
          </div>
          <span className="text-sm font-bold font-heading tabular-nums flex-shrink-0" style={{ color: getProfitColorHex(hub.p_profit) }}>
            {formatPercent(hub.p_profit)}
          </span>
        </button>
      ))}
    </div>
  )
}
