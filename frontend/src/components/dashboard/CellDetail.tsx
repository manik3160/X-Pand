import { useEffect, useState, useMemo, useRef } from 'react'
import {
  TrendingUp, AlertTriangle, XCircle, Thermometer, Shield, Building,
} from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Cell, Tooltip } from 'recharts'
import { useApp } from '@/hooks/useApp'
import { fetchPrediction, reverseGeocode } from '@/api/client'
import { formatPercent, getProfitColorHex, cn } from '@/lib/utils'
import { theme } from '@/lib/theme'
import Spinner from '@/components/ui/Spinner'
import ErrorBanner from '@/components/ui/ErrorBanner'
import type { PredictionResult, SHAPDriver } from '@/types/api'

function RecommendationBadge({ rec }: { rec: string }) {
  const config = {
    open: { icon: TrendingUp, cls: 'badge-open', label: 'OPEN' },
    monitor: { icon: AlertTriangle, cls: 'badge-monitor', label: 'MONITOR' },
    skip: { icon: XCircle, cls: 'badge-skip', label: 'SKIP' },
  }
  const c = config[rec as keyof typeof config] || config.skip
  const Icon = c.icon
  return (
    <span className={c.cls}>
      <Icon className="w-3 h-3 mr-1" aria-hidden="true" />
      {c.label}
    </span>
  )
}

function ShapChart({ drivers }: { drivers: SHAPDriver[] }) {
  if (!drivers || drivers.length === 0) return null

  const data = drivers.map(d => ({
    name: d.feature.replace(/_/g, ' '),
    value: d.impact,
    fill: d.impact > 0 ? theme.profitHigh : theme.profitLow,
  }))

  return (
    <div className="h-40">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
          <XAxis type="number" tick={{ fill: theme.textSubtle, fontSize: 10 }} axisLine={false} tickLine={false} />
          <YAxis type="category" dataKey="name" tick={{ fill: theme.textMuted, fontSize: 11 }} axisLine={false} tickLine={false} width={130} />
          <Tooltip
            contentStyle={{
              background: theme.surface,
              border: '1px solid rgb(var(--line) / 0.08)',
              borderRadius: '10px',
              color: theme.text,
              fontSize: '12px',
            }}
            formatter={(val: unknown) => [Number(val).toFixed(4), 'Impact']}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={14}>
            {data.map((entry, index) => <Cell key={index} fill={entry.fill} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

interface CellDetailProps {
  gridId: string
}

/**
 * Body content only — CellDetail is wrapped in <SideSheet> by
 * DashboardPage, which owns the open/close chrome and the title.
 */
export default function CellDetail({ gridId }: CellDetailProps) {
  const { predictions, selectedCity } = useApp()

  const basicPred = useMemo(
    () => predictions.find(p => p.grid_id === gridId),
    [predictions, gridId]
  )

  // Lazy initializers, not an effect — this remounts per gridId (see the
  // `key={gridId}` in DashboardPage), so seeding state straight from the
  // already-known batch prediction is a render-time derivation, not a
  // side effect to synchronize. areaName/error correctly start blank.
  const [detail, setDetail] = useState<PredictionResult | null>(() => basicPred ?? null)
  const [areaName, setAreaName] = useState<string>('')
  // Seeded true whenever there's a basicPred to refine — the effect below
  // starts that fetch immediately on mount, so this is never wrong at
  // t=0; it only ever needs to be turned off, in the .finally() below.
  const [loading, setLoading] = useState(() => !!basicPred)
  const [error, setError] = useState<string | null>(null)
  const reqId = useRef(0)

  useEffect(() => {
    if (!basicPred) return
    const id = ++reqId.current

    fetchPrediction({ lat: basicPred.lat, lon: basicPred.lon, grid_id: gridId }, selectedCity)
      .then(result => { if (reqId.current === id) setDetail(result) })
      .catch(() => { if (reqId.current === id) setError('Could not load full details for this cell.') })
      .finally(() => { if (reqId.current === id) setLoading(false) })

    reverseGeocode(basicPred.lat, basicPred.lon)
      .then(geo => { if (reqId.current === id) setAreaName(geo.area_name) })
      .catch(() => { if (reqId.current === id) setAreaName('') })
  }, [gridId, basicPred, selectedCity])

  if (!detail) {
    return (
      <div className="p-5 flex items-center justify-center">
        <Spinner />
      </div>
    )
  }

  const profitColor = getProfitColorHex(detail.p_profit)

  return (
    <div>
      <div className="px-5 py-3 text-xs text-text-secondary">
        {areaName && <div className="truncate mb-0.5">{areaName}</div>}
        <div className="text-[10px] text-text-muted font-heading tabular-nums">
          {detail.lat.toFixed(5)}, {detail.lon.toFixed(5)}
        </div>
      </div>

      <div className="px-5 pb-5 space-y-4">
        {error && <ErrorBanner message={error} />}

        {/* Profit Gauge */}
        <div className="glass-card p-5">
          <div className="flex items-center justify-between mb-3">
            <span className="metric-label">Profitability score</span>
            <RecommendationBadge rec={detail.recommendation} />
          </div>
          <div className="text-4xl font-bold mb-3 font-heading tabular-nums" style={{ color: profitColor }}>
            {formatPercent(detail.p_profit)}
          </div>
          <div className="w-full h-1.5 rounded-full overflow-hidden" style={{ background: 'rgb(var(--line) / 0.08)' }}>
            <div
              className="h-full rounded-full transition-all duration-700 ease-out"
              style={{ backgroundColor: profitColor, width: `${Math.min(detail.p_profit * 100, 100)}%` }}
            />
          </div>
          {detail.ci_lower !== null && detail.ci_upper !== null && (
            <div className="mt-3 flex gap-4 text-xs text-text-secondary">
              <div><span className="text-text-muted">CI lower:</span> <span className="font-heading text-text-primary tabular-nums">{detail.ci_lower.toFixed(3)}</span></div>
              <div><span className="text-text-muted">CI upper:</span> <span className="font-heading text-text-primary tabular-nums">{detail.ci_upper.toFixed(3)}</span></div>
            </div>
          )}
        </div>

        {detail.is_cold_start && (
          <div className="flex items-center gap-2 p-3 rounded-xl" style={{ background: 'rgb(var(--profit-mid) / 0.06)', border: '1px solid rgb(var(--profit-mid) / 0.2)' }}>
            <AlertTriangle className="w-4 h-4 flex-shrink-0" style={{ color: 'rgb(var(--profit-mid))' }} aria-hidden="true" />
            <span className="text-xs" style={{ color: 'rgb(var(--profit-mid))' }}>Cold-start cell — using a Thompson Sampling estimate</span>
          </div>
        )}

        {/* SHAP Drivers */}
        <div className="glass-card p-5">
          <div className="flex items-center justify-between mb-3">
            <span className="metric-label">Top feature impacts (SHAP)</span>
            {loading && <Spinner className="w-3 h-3" />}
          </div>
          {detail.shap_drivers && detail.shap_drivers.length > 0 ? (
            <ShapChart drivers={detail.shap_drivers} />
          ) : detail.is_cold_start ? (
            <div className="text-xs text-text-muted py-4 text-center">No SHAP data — cold-start cells use Thompson Sampling</div>
          ) : loading ? (
            <div className="text-xs text-text-muted py-4 text-center">Loading SHAP analysis…</div>
          ) : (
            <div className="text-xs text-text-muted py-4 text-center">SHAP data unavailable</div>
          )}
        </div>

        {/* Quick Info Cards */}
        <div className="grid grid-cols-2 gap-3">
          <div className="glass-card p-4">
            <Thermometer className="w-4 h-4 text-accent mb-2" aria-hidden="true" />
            <div className="metric-label">Model</div>
            <div className="text-sm font-semibold text-text-primary mt-1 font-heading">
              {detail.is_cold_start ? 'Thompson' : 'LightGBM'}
            </div>
          </div>
          <div className="glass-card p-4">
            <Shield className="w-4 h-4 text-accent mb-2" aria-hidden="true" />
            <div className="metric-label">Interval width</div>
            <div className="text-sm font-semibold text-text-primary mt-1 font-heading tabular-nums">
              {detail.ci_lower !== null && detail.ci_upper !== null
                ? `±${((detail.ci_upper - detail.ci_lower) * 50).toFixed(1)}%`
                : 'N/A'}
            </div>
          </div>
        </div>

        {/* Investment Assessment */}
        <div className="glass-card p-5">
          <div className="flex items-center gap-2 mb-3">
            <Building className="w-4 h-4 text-accent" aria-hidden="true" />
            <span className="metric-label">Investment assessment</span>
          </div>
          <div className={cn(
            'text-lg font-bold mb-1 font-heading',
            detail.p_profit > 0.7 ? 'text-accent' : detail.p_profit >= 0.4 ? 'text-warning' : 'text-danger'
          )}>
            {detail.p_profit > 0.7 ? 'Strong opportunity' : detail.p_profit >= 0.4 ? 'Needs monitoring' : 'Not recommended'}
          </div>
          <p className="text-xs text-text-muted leading-relaxed">
            {detail.p_profit > 0.7
              ? 'High model score. Consider for expansion, alongside local due diligence.'
              : detail.p_profit >= 0.4
              ? 'Moderate potential. Monitor market conditions and competition before committing.'
              : 'Low model score. Explore alternative locations.'}
          </p>
          <div className="mt-4 pt-4 flex justify-between items-center" style={{ borderTop: '1px solid rgb(var(--line) / 0.08)' }}>
            <span className="text-xs text-text-muted font-heading tracking-wide">EST. SETUP COST</span>
            <span className="font-heading font-bold text-accent tabular-nums text-lg">
              {detail.estimated_cost != null
                ? new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(detail.estimated_cost)
                : 'N/A'}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
