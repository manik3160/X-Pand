import { useMemo } from 'react'
import { theme } from '@/lib/theme'
import { mulberry32 } from '@/lib/seeded'

/* ─── 500m grid: a patch of cells with a scale bar over one of them ─── */

const GRID_COLS = 12
const GRID_ROWS = 5
const MARKED_COL = 4
const MARKED_ROW = 2

const TIER_COLOR = { high: theme.profitHigh, mid: theme.profitMid, low: theme.profitLow }
const TIER_OPACITY = { high: 0.55, mid: 0.32, low: 0.12 }

export function GridScaleVisual() {
  const cells = useMemo(() => {
    const rand = mulberry32(7301)
    const tiers = Array.from({ length: GRID_COLS * GRID_ROWS }, () => {
      const r = rand()
      return r > 0.8 ? 'high' : r > 0.5 ? 'mid' : 'low'
    }) as ('high' | 'mid' | 'low')[]
    // The cell the scale bar points at should be a good one, not a red one.
    tiers[MARKED_ROW * GRID_COLS + MARKED_COL] = 'high'
    return tiers
  }, [])

  return (
    <div className="mt-6" role="img" aria-label="A patch of grid cells, one marked as 500 metres wide">
      {/* scale bar spanning exactly one cell's column */}
      <div className="relative h-6" aria-hidden="true">
        <div
          className="absolute top-0 flex flex-col items-center"
          style={{ left: `${(MARKED_COL / GRID_COLS) * 100}%`, width: `${100 / GRID_COLS}%` }}
        >
          <span className="text-[11px] font-heading tabular-nums text-text-secondary leading-none mb-1">500 m</span>
          <div className="w-full h-[5px] border-x border-b" style={{ borderColor: 'rgb(var(--text-muted))' }} />
        </div>
      </div>
      <div className="grid gap-[2px]" style={{ gridTemplateColumns: `repeat(${GRID_COLS}, 1fr)` }} aria-hidden="true">
        {cells.map((tier, i) => {
          const marked = i === MARKED_ROW * GRID_COLS + MARKED_COL
          return (
            <div
              key={i}
              className="aspect-square rounded-[2px]"
              style={{
                background: TIER_COLOR[tier],
                opacity: marked ? 1 : TIER_OPACITY[tier],
                outline: marked ? `2px solid ${theme.accent}` : 'none',
                outlineOffset: marked ? '1px' : undefined,
              }}
            />
          )
        })}
      </div>
    </div>
  )
}

/* ─── pipeline: four stages as a vertical rail, each with what it does ─── */

const STAGES = [
  { name: 'GWR', note: 'local spatial context for every cell' },
  { name: 'LightGBM', note: 'profit probability per cell' },
  { name: 'Thompson Sampling', note: 'estimates for cold-start cells' },
  { name: 'BIP', note: 'picks hub sites under a separation rule' },
]

export function PipelineVisual() {
  return (
    <ol className="mt-6 relative">
      <div
        className="absolute left-[11px] top-3 bottom-3 w-px"
        style={{ background: 'rgb(var(--line) / 0.15)' }}
        aria-hidden="true"
      />
      {STAGES.map((s, i) => (
        <li key={s.name} className="relative flex items-start gap-4 pb-5 last:pb-0">
          <span
            className="relative z-10 w-6 h-6 rounded-full flex items-center justify-center text-[11px] font-heading font-semibold tabular-nums flex-shrink-0"
            style={{ background: 'rgb(var(--surface))', border: `1px solid ${theme.accent}`, color: theme.accent }}
          >
            {i + 1}
          </span>
          <div className="-mt-0.5">
            <div className="text-sm font-heading font-medium text-text-primary">{s.name}</div>
            <div className="text-xs text-text-muted">{s.note}</div>
          </div>
        </li>
      ))}
    </ol>
  )
}

/* ─── SHAP: diverging bars, like the real cell-detail chart ─── */

const DRIVERS = [
  { label: 'pop density', value: 2.1 },
  { label: 'income index', value: 1.2 },
  { label: 'road density', value: -1.6 },
]
const MAX_ABS = 2.5

export function ShapVisual() {
  return (
    <div className="w-full" role="img" aria-label="Example of feature impacts: population density and income index raise the score, road density lowers it">
      <div className="space-y-3" aria-hidden="true">
        {DRIVERS.map(d => {
          const pct = (Math.abs(d.value) / MAX_ABS) * 50
          const positive = d.value > 0
          return (
            <div key={d.label} className="grid items-center gap-3" style={{ gridTemplateColumns: '96px 1fr 44px' }}>
              <span className="text-xs text-text-secondary">{d.label}</span>
              <div className="relative h-3">
                <div className="absolute left-1/2 top-[-3px] bottom-[-3px] w-px" style={{ background: 'rgb(var(--line) / 0.2)' }} />
                <div
                  className="absolute top-0 h-full rounded-sm"
                  style={{
                    background: positive ? theme.profitHigh : theme.profitLow,
                    width: `${pct}%`,
                    left: positive ? '50%' : `${50 - pct}%`,
                  }}
                />
              </div>
              <span className="text-xs font-heading tabular-nums text-right" style={{ color: positive ? theme.profitHigh : theme.profitLow }}>
                {positive ? '+' : '−'}{Math.abs(d.value).toFixed(1)}
              </span>
            </div>
          )
        })}
      </div>
      <p className="mt-4 text-[11px] text-text-muted">Illustrative values — the real drivers differ for every cell.</p>
    </div>
  )
}
