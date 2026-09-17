import { useMemo } from 'react'
import { useApp } from '@/hooks/useApp'
import Skeleton from '@/components/ui/Skeleton'

interface ChipProps {
  label: string
  value: number
  colorVar?: string
}

function Chip({ label, value, colorVar }: ChipProps) {
  return (
    <div className="flex flex-col gap-0.5 px-3 py-1.5 rounded-lg" style={{ background: 'rgb(var(--surface-raised) / 0.4)' }}>
      <span className="text-[10px] font-semibold text-text-muted uppercase tracking-[0.08em]">{label}</span>
      <span
        className="text-lg font-heading font-semibold tabular-nums"
        style={{ color: colorVar ? `rgb(var(${colorVar}))` : 'rgb(var(--text))' }}
      >
        {value.toLocaleString()}
      </span>
    </div>
  )
}

/** Total / high / monitor / skip counts — replaces MetricCards.tsx (dedupes the animated-counter logic that also lived in HomePage). */
export default function MetricChips() {
  const { predictions, predictionsLoading } = useApp()

  const counts = useMemo(() => {
    let high = 0, monitor = 0, skip = 0
    for (const p of predictions) {
      if (p.p_profit > 0.7) high++
      else if (p.p_profit >= 0.4) monitor++
      else skip++
    }
    return { total: predictions.length, high, monitor, skip }
  }, [predictions])

  if (predictionsLoading && predictions.length === 0) {
    return (
      <div className="flex items-center gap-2">
        {Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-11 w-20" />)}
      </div>
    )
  }

  if (predictions.length === 0) return null

  return (
    <div className="flex items-center gap-2 flex-wrap">
      <Chip label="Total" value={counts.total} />
      <Chip label="High" value={counts.high} colorVar="--profit-high" />
      <Chip label="Monitor" value={counts.monitor} colorVar="--profit-mid" />
      <Chip label="Skip" value={counts.skip} colorVar="--profit-low" />
    </div>
  )
}
