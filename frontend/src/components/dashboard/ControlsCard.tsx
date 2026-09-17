import { useState } from 'react'
import { Loader2, Zap, SlidersHorizontal } from 'lucide-react'
import { useApp } from '@/hooks/useApp'
import Panel from '@/components/ui/Panel'
import Slider from '@/components/ui/Slider'
import Button from '@/components/ui/Button'
import ErrorBanner from '@/components/ui/ErrorBanner'

/** The sliders + run button, shared between the desktop floating card and the mobile bottom sheet. */
export function OptimizerControlsContent() {
  const {
    selectedCity, predictions, predictionsLoading,
    optimizing, optimizeError, runOptimize,
  } = useApp()

  const [minProb, setMinProb] = useState(0.5)
  const [maxHubs, setMaxHubs] = useState(10)
  const [minSep, setMinSep] = useState(2.0)

  const handleOptimize = () => {
    runOptimize({
      max_hubs: maxHubs,
      min_separation_km: minSep,
      min_prob_threshold: minProb,
      city: selectedCity,
    })
  }

  return (
    <>
      <Slider id="min-probability" label="Min probability" value={minProb} onChange={setMinProb} min={0} max={1} step={0.05} formatValue={v => v.toFixed(2)} />
      <Slider id="max-hubs" label="Max hubs" value={maxHubs} onChange={setMaxHubs} min={1} max={50} step={1} formatValue={v => String(v)} />
      <Slider id="min-separation" label="Min separation (km)" value={minSep} onChange={setMinSep} min={0.5} max={10} step={0.5} formatValue={v => v.toFixed(1)} />

      <Button
        variant="primary"
        size="lg"
        onClick={handleOptimize}
        disabled={optimizing || predictionsLoading || predictions.length === 0}
        className="w-full mt-1"
      >
        {optimizing ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" aria-hidden="true" />
            Solving…
          </>
        ) : (
          <>
            <Zap className="w-4 h-4" aria-hidden="true" />
            Run optimizer
          </>
        )}
      </Button>

      {optimizeError && <ErrorBanner message={optimizeError} className="mt-3" />}
    </>
  )
}

export default function ControlsCard() {
  return (
    <Panel title="Optimizer controls" icon={<SlidersHorizontal className="w-3.5 h-3.5 text-text-muted" />} collapsible className="w-[260px]" bodyClassName="p-4">
      <OptimizerControlsContent />
    </Panel>
  )
}
