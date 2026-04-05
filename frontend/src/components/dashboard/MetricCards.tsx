import { useEffect, useState } from 'react'
import { useApp } from '@/hooks/useApp'

function AnimCounter({ end, duration = 1000 }: { end: number; duration?: number }) {
  const [val, setVal] = useState(0)
  useEffect(() => {
    if (end === 0) { setVal(0); return }
    let start = 0
    const step = end / (duration / 16)
    const id = setInterval(() => {
      start += step
      if (start >= end) { setVal(end); clearInterval(id) }
      else setVal(Math.floor(start))
    }, 16)
    return () => clearInterval(id)
  }, [end, duration])
  return <>{val.toLocaleString()}</>
}

interface CardProps {
  label: string
  value: number
  color: string
  borderColor: string
}

function MetricCard({ label, value, color, borderColor }: CardProps) {
  return (
    <div
      className="metric-card py-2.5 px-4"
      style={{ borderBottom: `2px solid ${borderColor}` }}
    >
      <span className="metric-label text-[10px]">{label}</span>
      <div className="metric-value font-heading tabular-nums text-xl" style={{ color }}>
        <AnimCounter end={value} />
      </div>
    </div>
  )
}

export default function MetricCards() {
  const { predictions } = useApp()

  const total = predictions.length
  const high = predictions.filter(p => p.p_profit > 0.7).length
  const monitor = predictions.filter(p => p.p_profit >= 0.4 && p.p_profit <= 0.7).length
  const skip = predictions.filter(p => p.p_profit < 0.4).length

  return (
    <div className="grid grid-cols-4 gap-3 animate-fade-up">
      <MetricCard label="Total Cells" value={total} color="#ffffff" borderColor="rgba(255,255,255,0.1)" />
      <MetricCard label="High Potential" value={high} color="#00ff88" borderColor="#00ff88" />
      <MetricCard label="Monitor" value={monitor} color="#ffaa00" borderColor="#ffaa00" />
      <MetricCard label="Skip" value={skip} color="#ff4444" borderColor="#ff4444" />
    </div>
  )
}
