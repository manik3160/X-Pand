import { useState, useCallback, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { MapPin, ChevronDown, Loader2, ArrowLeft, CheckCircle2, Zap, AlertTriangle } from 'lucide-react'
import { useApp } from '@/hooks/useApp'
import { useServerStatus } from '@/hooks/useServerStatus'
import { cn } from '@/lib/utils'

export default function Sidebar() {
  const navigate = useNavigate()
  const {
    cities, citiesLoading, selectedCity, setSelectedCity, loadCityData,
    predictions, predictionsLoading, predictionsError,
    optimizeResult, optimizing, optimizeError, runOptimize,
  } = useApp()
  const { status: serverStatus, refresh: refreshServerStatus } = useServerStatus()

  const [minProb, setMinProb] = useState(0.5)
  const [maxHubs, setMaxHubs] = useState(10)
  const [minSep, setMinSep] = useState(2.0)
  const [cityOpen, setCityOpen] = useState(false)

  const selectedCityInfo = cities.find(c => c.key === selectedCity)

  const handleCityChange = useCallback((cityKey: string) => {
    setSelectedCity(cityKey)
    loadCityData(cityKey)
    setCityOpen(false)
  }, [setSelectedCity, loadCityData])

  useEffect(() => {
    if (cities.length > 0 && predictions.length === 0 && !predictionsLoading && !predictionsError) {
      loadCityData(selectedCity)
    }
  }, [cities, selectedCity, predictions.length, predictionsLoading, predictionsError, loadCityData])

  const handleOptimize = () => {
    runOptimize({
      max_hubs: maxHubs,
      min_separation_km: minSep,
      min_prob_threshold: minProb,
      city: selectedCity,
    })
  }

  return (
    <aside
      className="w-[280px] h-full flex flex-col flex-shrink-0"
      style={{
        background: 'rgba(10, 10, 15, 0.95)',
        backdropFilter: 'blur(20px)',
        WebkitBackdropFilter: 'blur(20px)',
        borderRight: '1px solid rgba(255, 255, 255, 0.06)',
      }}
    >
      {/* ─── Header ─── */}
      <div className="px-5 py-5" style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
        <div className="flex items-center gap-3">
          <button
            onClick={() => navigate('/')}
            className="p-1.5 rounded-lg hover:bg-[rgba(255,255,255,0.05)] transition-all"
          >
            <ArrowLeft className="w-4 h-4 text-text-secondary" />
          </button>
          <div>
            <div className="font-heading font-semibold text-white text-base tracking-tight">X-PAND.AI</div>
            <div className="text-[11px] text-text-muted">BI-101 · Geospatial Intelligence</div>
          </div>
        </div>
        <div className="mt-3 inline-flex items-center gap-2 px-3 py-1 rounded-full" style={{ background: 'rgb(var(--accent) / 0.06)', border: '1px solid rgb(var(--accent) / 0.15)' }}>
          <div
            className="live-dot-sm"
            data-state={serverStatus === 'online' ? 'ok' : serverStatus === 'offline' ? 'off' : 'warn'}
          />
          <span className="text-[11px] font-semibold text-accent">
            {serverStatus === 'online' && 'LIVE ML'}
            {serverStatus === 'waking' && 'WAKING UP…'}
            {serverStatus === 'checking' && 'CONNECTING…'}
            {serverStatus === 'offline' && 'OFFLINE'}
          </span>
        </div>
      </div>

      {/* ─── City Selector ─── */}
      <div className="px-5 py-4" style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
        <div className="text-[11px] font-semibold text-text-muted uppercase tracking-[0.12em] mb-3">
          City Selector
        </div>
        <div className="relative">
          <button
            onClick={() => setCityOpen(!cityOpen)}
            className="w-full flex items-center justify-between p-3 rounded-[10px] transition-all"
            style={{
              background: 'rgba(255,255,255,0.05)',
              border: '1px solid rgba(255,255,255,0.1)',
            }}
          >
            <div className="flex items-center gap-2">
              <MapPin className="w-4 h-4 text-accent" />
              <span className="text-sm font-medium text-white">{selectedCityInfo?.name || selectedCity}</span>
            </div>
            <ChevronDown className={cn("w-4 h-4 text-text-muted transition-transform", cityOpen && "rotate-180")} />
          </button>

          {cityOpen && (
            <div
              className="absolute top-full left-0 right-0 mt-1 z-50 rounded-xl overflow-hidden"
              style={{
                background: 'rgba(10, 10, 15, 0.98)',
                border: '1px solid rgba(255,255,255,0.08)',
                backdropFilter: 'blur(20px)',
              }}
            >
              {cities.map(city => (
                <button
                  key={city.key}
                  onClick={() => handleCityChange(city.key)}
                  className={cn(
                    "w-full flex items-center justify-between p-3 text-sm transition-all",
                    city.key === selectedCity
                      ? "bg-[rgba(0,255,136,0.08)] text-accent"
                      : "text-white hover:bg-[rgba(255,255,255,0.05)]"
                  )}
                >
                  <span className="font-medium">{city.name}</span>
                  <span className="text-xs text-text-muted tabular-nums">{city.cell_count.toLocaleString()} cells</span>
                </button>
              ))}
            </div>
          )}
        </div>

        {selectedCityInfo && (
          <div className="mt-2 flex items-center gap-1.5 text-[13px] text-text-secondary">
            <MapPin className="w-3 h-3 text-accent" />
            {selectedCityInfo.cell_count.toLocaleString()} grid cells
          </div>
        )}
      </div>

      {/* ─── Optimizer Controls ─── */}
      <div className="px-5 py-4" style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
        <div className="text-[11px] font-semibold text-text-muted uppercase tracking-[0.12em] mb-4">
          Optimizer Controls
        </div>

        {/* Min Probability */}
        <div className="mb-4">
          <div className="flex justify-between items-center mb-2">
            <label className="text-xs text-text-secondary">MIN PROBABILITY</label>
            <span className="text-xs font-heading font-semibold text-accent tabular-nums">{minProb.toFixed(2)}</span>
          </div>
          <input type="range" min="0" max="1" step="0.05" value={minProb} onChange={e => setMinProb(parseFloat(e.target.value))} />
        </div>

        {/* Max Hubs */}
        <div className="mb-4">
          <div className="flex justify-between items-center mb-2">
            <label className="text-xs text-text-secondary">MAX HUBS</label>
            <span className="text-xs font-heading font-semibold text-accent tabular-nums">{maxHubs}</span>
          </div>
          <input type="range" min="1" max="50" step="1" value={maxHubs} onChange={e => setMaxHubs(parseInt(e.target.value))} />
        </div>

        {/* Min Separation */}
        <div className="mb-5">
          <div className="flex justify-between items-center mb-2">
            <label className="text-xs text-text-secondary">MIN SEPARATION KM</label>
            <span className="text-xs font-heading font-semibold text-accent tabular-nums">{minSep.toFixed(1)}</span>
          </div>
          <input type="range" min="0.5" max="10" step="0.5" value={minSep} onChange={e => setMinSep(parseFloat(e.target.value))} />
        </div>

        {/* Run Button */}
        <button
          onClick={handleOptimize}
          disabled={optimizing || predictionsLoading || predictions.length === 0}
          className={cn(
            "w-full flex items-center justify-center gap-2 h-12 rounded-xl font-bold text-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed",
            optimizing
              ? "bg-[rgb(var(--accent)/0.1)] text-accent cursor-wait"
              : "bg-accent text-accent-fg hover:brightness-105 hover:shadow-glow-green"
          )}
        >
          {optimizing ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Solving BIP...
            </>
          ) : (
            <>
              <Zap className="w-4 h-4" />
              RUN OPTIMIZER
            </>
          )}
        </button>

        {optimizeError && (
          <div
            className="mt-3 flex items-start gap-2 rounded-lg px-3 py-2.5 text-xs"
            style={{ background: 'rgb(var(--profit-low) / 0.08)', border: '1px solid rgb(var(--profit-low) / 0.25)', color: 'rgb(var(--profit-low))' }}
            role="alert"
          >
            <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
            <span>{optimizeError}</span>
          </div>
        )}
      </div>

      {/* ─── Prediction load error ─── */}
      {predictionsError && (
        <div className="px-5 py-4" style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
          <div
            className="rounded-xl p-4 flex items-start gap-2.5"
            style={{ background: 'rgb(var(--profit-low) / 0.06)', border: '1px solid rgb(var(--profit-low) / 0.25)' }}
            role="alert"
          >
            <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" style={{ color: 'rgb(var(--profit-low))' }} />
            <div className="flex-1">
              <div className="text-xs font-medium" style={{ color: 'rgb(var(--profit-low))' }}>{predictionsError}</div>
              <button
                onClick={() => loadCityData(selectedCity)}
                className="mt-2 text-xs font-semibold text-accent hover:underline"
              >
                Retry
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ─── Optimization Result ─── */}
      {optimizeResult && (
        <div className="px-5 py-4" style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
          <div
            className="rounded-xl p-4"
            style={{
              background: 'rgb(var(--surface-raised) / 0.4)',
              border: '1px solid rgb(var(--accent) / 0.3)',
            }}
          >
            <div className="text-2xl font-bold text-accent font-heading mb-2">
              {optimizeResult.hub_details.length} hubs selected
            </div>
            <div className="space-y-1.5 text-xs text-text-secondary">
              <div className="flex justify-between">
                <span>Score</span>
                <span className="font-heading text-white tabular-nums">{optimizeResult.total_score.toFixed(4)}</span>
              </div>
              <div className="flex justify-between items-center">
                <span>Separation</span>
                {optimizeResult.separation_constraint_met ? (
                  <span className="flex items-center gap-1 text-accent"><CheckCircle2 className="w-3 h-3" /> MET</span>
                ) : (
                  <span className="text-danger">VIOLATED</span>
                )}
              </div>
              <div className="flex justify-between">
                <span>Eligible</span>
                <span className="text-white tabular-nums">{optimizeResult.eligible_cells} / {optimizeResult.total_cells}</span>
              </div>
            </div>
            <div
              className="mt-3 pt-2 text-[11px] font-mono text-text-muted"
              style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}
            >
              LightGBM scored {optimizeResult.total_cells} cells → BIP solved in {optimizeResult.processing_time_seconds.toFixed(1)}s
            </div>
          </div>
        </div>
      )}

      {/* ─── System Status ─── */}
      <div className="px-5 py-4 mt-auto">
        <div className="space-y-2.5">
          <div className="flex items-center gap-2">
            <div className="live-dot-sm" data-state={serverStatus === 'online' ? 'ok' : serverStatus === 'offline' ? 'off' : 'warn'} />
            <span className="text-xs text-text-secondary">
              {serverStatus === 'online' && 'API connected'}
              {serverStatus === 'waking' && 'API waking up…'}
              {serverStatus === 'checking' && 'Connecting to API…'}
              {serverStatus === 'offline' && 'API unreachable'}
            </span>
            {serverStatus === 'offline' && (
              <button onClick={refreshServerStatus} className="text-xs font-semibold text-accent hover:underline">
                Retry
              </button>
            )}
          </div>
          <div className="flex items-center gap-2">
            <div className="live-dot-sm" data-state={predictionsError ? 'off' : predictions.length > 0 ? 'ok' : 'warn'} />
            <span className="text-xs text-text-secondary">
              {predictionsError ? 'Models unavailable' : predictions.length > 0 ? 'Models loaded' : 'Loading models…'}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div className="live-dot-sm" data-state={citiesLoading ? 'warn' : 'ok'} />
            <span className="text-xs text-text-secondary">
              {citiesLoading ? 'Loading cities…' : `${cities.length} cities active`}
            </span>
          </div>
        </div>
      </div>
    </aside>
  )
}
