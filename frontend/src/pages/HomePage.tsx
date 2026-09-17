import { useNavigate } from 'react-router-dom'
import { Grid3X3, Brain, BarChart3, ArrowRight } from 'lucide-react'
import { useApp } from '@/hooks/useApp'
import HeroMapPreview from '@/components/landing/HeroMapPreview'
import { theme } from '@/lib/theme'

/* ─── Feature Card ─── */
function FeatureCard({
  icon: Icon,
  title,
  description,
  iconColor,
  delay,
}: {
  icon: React.ElementType
  title: string
  description: string
  iconColor: string
  delay: string
}) {
  return (
    <div className="glass-card-hover p-6 cursor-default animate-fade-up" style={{ animationDelay: delay }}>
      <div
        className="w-10 h-10 rounded-xl flex items-center justify-center mb-4"
        style={{ background: `${iconColor}15`, border: `1px solid ${iconColor}30` }}
      >
        <Icon className="w-5 h-5" style={{ color: iconColor }} aria-hidden="true" />
      </div>
      <h3 className="text-base font-semibold text-text-primary mb-2 font-heading">{title}</h3>
      <p className="text-sm text-text-secondary leading-relaxed">{description}</p>
    </div>
  )
}

function Stat({ value, label }: { value: string; label: string }) {
  return (
    <div className="px-4 first:pl-0 last:pr-0">
      <div className="text-2xl font-semibold text-text-primary font-heading tabular-nums">{value}</div>
      <div className="text-xs text-text-muted mt-0.5">{label}</div>
    </div>
  )
}

/* ─── HOME PAGE ─── */
export default function HomePage() {
  const navigate = useNavigate()
  const { cities } = useApp()
  const cityCount = cities.length || 16

  return (
    <div className="min-h-screen relative overflow-x-hidden">
      {/* ═══ NAVBAR ═══ */}
      <nav
        className="fixed top-0 left-0 right-0 z-50 h-16 flex items-center justify-between px-4 sm:px-8"
        style={{
          background: 'rgb(var(--bg) / 0.8)',
          backdropFilter: 'blur(20px)',
          WebkitBackdropFilter: 'blur(20px)',
          borderBottom: '1px solid rgb(var(--line) / 0.06)',
        }}
      >
        <div className="flex items-center gap-2">
          <div className="live-dot-sm" />
          <span className="font-heading text-lg font-bold tracking-tight text-text-primary">X-PAND.AI</span>
        </div>

        <button onClick={() => navigate('/dashboard')} className="btn-nav text-sm">
          Launch dashboard <ArrowRight className="w-3.5 h-3.5" aria-hidden="true" />
        </button>
      </nav>

      {/* ═══ HERO ═══ */}
      <section className="min-h-screen flex items-center pt-24 pb-16 lg:pt-16 lg:pb-0">
        <div className="max-w-7xl mx-auto px-4 sm:px-8 w-full flex flex-col lg:flex-row items-center gap-12">

          <div className="flex-1 w-full lg:max-w-[55%]">
            <div
              className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full mb-8 animate-fade-up"
              style={{ background: 'rgb(var(--surface-raised) / 0.5)', border: '1px solid rgb(var(--line) / 0.08)' }}
            >
              <div className="live-dot-sm" />
              <span className="text-sm text-accent font-medium">ML-powered · 500m grid · explainable scoring</span>
            </div>

            <h1 className="text-[42px] sm:text-[56px] lg:text-[64px] font-bold font-heading leading-[1.05] tracking-[-0.03em] mb-6 animate-fade-up" style={{ animationDelay: '0.1s' }}>
              <span className="text-text-primary">Predict where</span>
              <br />
              <span className="text-accent">profit lives.</span>
            </h1>

            <p className="text-lg text-text-secondary font-normal mb-5 animate-fade-up leading-relaxed max-w-lg" style={{ animationDelay: '0.2s' }}>
              500-metre grid intelligence across {cityCount} Indian cities.
              <br />
              Satellite and OSM data feed a chained ML pipeline to pinpoint your next winning location.
            </p>

            <div className="flex flex-wrap items-center gap-y-4 mb-10 animate-fade-up" style={{ animationDelay: '0.25s' }}>
              <Stat value={String(cityCount)} label="Indian cities" />
              <div className="w-px h-10 bg-[rgb(var(--line)/0.08)]" />
              <Stat value="500m" label="grid precision" />
              <div className="w-px h-10 bg-[rgb(var(--line)/0.08)]" />
              <Stat value="4" label="chained models" />
              <div className="w-px h-10 bg-[rgb(var(--line)/0.08)]" />
              <Stat value="SHAP" label="explanations" />
            </div>

            <div className="flex items-center gap-5 animate-fade-up" style={{ animationDelay: '0.3s' }}>
              <button onClick={() => navigate('/dashboard')} className="btn-primary group">
                Launch dashboard
                <ArrowRight className="w-5 h-5 group-hover:translate-x-0.5 transition-transform" aria-hidden="true" />
              </button>
            </div>
          </div>

          <div className="flex-1 w-full flex justify-center lg:justify-end">
            <HeroMapPreview />
          </div>
        </div>
      </section>

      {/* ═══ FEATURE CARDS ═══ */}
      <section className="py-24 px-4 sm:px-8">
        <div className="max-w-7xl mx-auto grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
          <FeatureCard
            icon={Grid3X3}
            title="500m precision grid"
            description="Not pin codes. Not districts. Every 500-metre cell of a city scored independently using satellite and OSM data."
            iconColor={theme.profitHigh}
            delay="0.5s"
          />
          <FeatureCard
            icon={Brain}
            title="4-stage pipeline"
            description="GWR → LightGBM → Thompson Sampling → BIP. Chained spatial intelligence with confidence intervals."
            iconColor={theme.accent}
            delay="0.6s"
          />
          <FeatureCard
            icon={BarChart3}
            title="Explainable by default"
            description="Every prediction ships with SHAP drivers, confidence bounds, and a plain-language recommendation."
            iconColor={theme.profitMid}
            delay="0.7s"
          />
        </div>
      </section>

      {/* ═══ FOOTER ═══ */}
      <footer className="py-8 px-4 sm:px-8 text-center" style={{ borderTop: '1px solid rgb(var(--line) / 0.06)' }}>
        <p className="text-[13px] text-text-muted">
          Built with WorldPop · OpenStreetMap · LightGBM · PuLP · FastAPI
        </p>
      </footer>
    </div>
  )
}
