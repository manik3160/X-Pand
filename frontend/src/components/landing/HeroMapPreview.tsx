import { useMemo } from 'react'
import { TrendingUp } from 'lucide-react'
import { theme } from '@/lib/theme'
import { mulberry32 } from '@/lib/seeded'

const COLS = 14
const ROWS = 10
const HUB_INDEX = 42 // a fixed "winning" cell the ring highlights

interface Cell {
  tier: 'high' | 'mid' | 'low'
}

function buildGrid(): Cell[] {
  const rand = mulberry32(20260226)
  return Array.from({ length: COLS * ROWS }, () => {
    const r = rand()
    const tier: Cell['tier'] = r > 0.82 ? 'high' : r > 0.55 ? 'mid' : 'low'
    return { tier }
  })
}

const TIER_COLOR: Record<Cell['tier'], string> = {
  high: theme.profitHigh,
  mid: theme.profitMid,
  low: theme.profitLow,
}
const TIER_OPACITY: Record<Cell['tier'], number> = {
  high: 0.55,
  mid: 0.32,
  low: 0.12,
}

/**
 * A static, seeded illustration of the dashboard's scored grid + a hub
 * pick — replaces the interactive globe (which showed world cities;
 * the product only covers India). No network calls, no live data.
 */
export default function HeroMapPreview() {
  const grid = useMemo(() => buildGrid(), [])

  return (
    <div className="relative w-full max-w-[480px] animate-fade-right">
      <div className="panel overflow-hidden p-3">
        <div
          className="grid gap-[2px] rounded-lg overflow-hidden relative"
          style={{ gridTemplateColumns: `repeat(${COLS}, 1fr)`, aspectRatio: `${COLS} / ${ROWS}` }}
        >
          {grid.map((cell, i) => (
            <div
              key={i}
              className="animate-fade-up motion-reduce:animate-none"
              style={{
                background: TIER_COLOR[cell.tier],
                opacity: TIER_OPACITY[cell.tier],
                animationDelay: `${0.4 + (i % COLS) * 0.008 + Math.floor(i / COLS) * 0.015}s`,
                animationDuration: '0.4s',
              }}
            />
          ))}
          {/* Hub ring over the fixed "winning" cell */}
          <div
            className="absolute rounded-full border-2 animate-fade-up motion-reduce:animate-none"
            style={{
              borderColor: theme.accent,
              width: `${(1 / COLS) * 100 * 1.6}%`,
              aspectRatio: '1 / 1',
              left: `${((HUB_INDEX % COLS) + 0.5) / COLS * 100}%`,
              top: `${(Math.floor(HUB_INDEX / COLS) + 0.5) / ROWS * 100}%`,
              transform: 'translate(-50%, -50%)',
              animationDelay: '0.9s',
              boxShadow: `0 0 0 4px ${theme.accent}22`,
            }}
          />
        </div>

        {/* Mock results chip, echoing the real dashboard's ResultsCard */}
        <div
          className="mt-3 flex items-center gap-2 px-3 py-2 rounded-lg animate-fade-up motion-reduce:animate-none"
          style={{ background: 'rgb(var(--surface-raised) / 0.5)', animationDelay: '1.05s' }}
        >
          <TrendingUp className="w-3.5 h-3.5 text-accent flex-shrink-0" aria-hidden="true" />
          <span className="text-xs text-text-secondary">Best cell scored </span>
          <span className="text-xs font-heading font-semibold text-accent tabular-nums">98.4%</span>
        </div>
      </div>
      <div className="absolute -bottom-6 left-0 text-[11px] text-text-muted">Illustrative preview — not live data</div>
    </div>
  )
}
