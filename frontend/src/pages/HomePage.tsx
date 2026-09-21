import { useNavigate } from 'react-router-dom'
import { ArrowRight } from 'lucide-react'
import { useApp } from '@/hooks/useApp'
import HeroMapPreview from '@/components/landing/HeroMapPreview'
import { GridScaleVisual, PipelineVisual, ShapVisual } from '@/components/landing/FeatureVisuals'

/* ─── Feature block: a heading + description with a fragment of the product beside/below it ─── */
function FeatureBlock({
  title,
  description,
  className = '',
  split = false,
  children,
}: {
  title: string
  description: string
  className?: string
  split?: boolean
  children: React.ReactNode
}) {
  return (
    <div className={`glass-card p-6 sm:p-8 ${split ? 'md:grid md:grid-cols-2 md:gap-10 md:items-center' : ''} ${className}`}>
      <div>
        <h3 className="text-lg font-semibold text-text-primary mb-2 font-heading">{title}</h3>
        <p className="text-sm text-text-secondary leading-relaxed max-w-md">{description}</p>
      </div>
      <div className={split ? 'mt-6 md:mt-0' : ''}>{children}</div>
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

      {/* ═══ HOW IT WORKS — each block shows a fragment of the product, not an icon ═══ */}
      <section className="py-24 px-4 sm:px-8">
        <div className="max-w-7xl mx-auto grid lg:grid-cols-5 gap-6">
          <FeatureBlock
            className="lg:col-span-3"
            title="Scored per 500-metre cell, not per pin code"
            description="Every cell of a city is scored on its own, from satellite and OpenStreetMap data — so a good block and a bad block next door don't average each other out."
          >
            <GridScaleVisual />
          </FeatureBlock>

          <FeatureBlock
            className="lg:col-span-2"
            title="Four models, chained"
            description="Each stage feeds the next, from local spatial context through to choosing the sites."
          >
            <PipelineVisual />
          </FeatureBlock>

          <FeatureBlock
            className="lg:col-span-5"
            split
            title="Every score comes with its reasons"
            description="Each prediction ships with the top feature drivers, a confidence interval, and a plain-language recommendation — so you can argue with it, not just accept it."
          >
            <ShapVisual />
          </FeatureBlock>
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
