import { CheckCircle2, XCircle, ListChecks, Clock } from 'lucide-react'
import { useApp } from '@/hooks/useApp'
import Panel from '@/components/ui/Panel'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/Tabs'
import MetricChips from './MetricChips'
import TopLocations from './TopLocations'
import OptimizerPanel from './OptimizerPanel'

/** The only place optimizer results are shown (previously duplicated between the old Sidebar and this panel). */
function OptimizeSummary() {
  const { optimizeResult } = useApp()
  if (!optimizeResult) return null

  return (
    <div className="px-4 pt-3 grid grid-cols-3 gap-2 text-center">
      <div className="rounded-lg py-2" style={{ background: 'rgb(var(--surface-raised) / 0.4)' }}>
        <div className="text-base font-heading font-semibold text-accent tabular-nums">{optimizeResult.hub_details.length}</div>
        <div className="text-[10px] text-text-muted uppercase">Hubs</div>
      </div>
      <div className="rounded-lg py-2" style={{ background: 'rgb(var(--surface-raised) / 0.4)' }}>
        <div className="text-base font-heading font-semibold text-text-primary tabular-nums">{optimizeResult.total_score.toFixed(2)}</div>
        <div className="text-[10px] text-text-muted uppercase">Score</div>
      </div>
      <div className="rounded-lg py-2" style={{ background: 'rgb(var(--surface-raised) / 0.4)' }}>
        <div className="flex items-center justify-center gap-1 text-base font-heading font-semibold text-text-primary tabular-nums">
          <Clock className="w-3 h-3" aria-hidden="true" />
          {optimizeResult.processing_time_seconds.toFixed(1)}s
        </div>
        <div className="text-[10px] text-text-muted uppercase">Time</div>
      </div>
      <div className="col-span-3 flex items-center justify-between text-[11px] text-text-muted pt-1">
        {optimizeResult.separation_constraint_met ? (
          <span className="flex items-center gap-1 text-accent"><CheckCircle2 className="w-3 h-3" aria-hidden="true" /> Separation met</span>
        ) : (
          <span className="flex items-center gap-1" style={{ color: 'rgb(var(--profit-low))' }}><XCircle className="w-3 h-3" aria-hidden="true" /> Separation violated</span>
        )}
        <span>Eligible {optimizeResult.eligible_cells} / {optimizeResult.total_cells}</span>
      </div>
    </div>
  )
}

/** Chips + optimizer summary + tabbed lists, shared between the desktop floating card and the mobile bottom sheet. */
export function ResultsContent() {
  const { optimizeResult } = useApp()

  return (
    <>
      <div className="px-3 pt-3">
        <MetricChips />
      </div>
      <OptimizeSummary />
      <Tabs defaultValue="top" className="flex-1 flex flex-col min-h-0 mt-2">
        <TabsList>
          <TabsTrigger value="top">Top locations</TabsTrigger>
          <TabsTrigger value="hubs">
            Optimized hubs{optimizeResult ? ` (${optimizeResult.hub_details.length})` : ''}
          </TabsTrigger>
        </TabsList>
        <TabsContent value="top" className="flex-1 overflow-y-auto">
          <TopLocations />
        </TabsContent>
        <TabsContent value="hubs" className="flex-1 overflow-y-auto">
          <OptimizerPanel />
        </TabsContent>
      </Tabs>
    </>
  )
}

export default function ResultsCard() {
  return (
    <Panel
      title="Results"
      icon={<ListChecks className="w-3.5 h-3.5 text-text-muted" />}
      collapsible
      className="w-[300px] max-h-[70vh] flex flex-col"
      bodyClassName="flex-1 overflow-y-auto flex flex-col"
    >
      <ResultsContent />
    </Panel>
  )
}
