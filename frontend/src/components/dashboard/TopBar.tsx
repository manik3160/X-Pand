import { useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowLeft } from 'lucide-react'
import { useApp } from '@/hooks/useApp'
import { useServerStatus } from '@/hooks/useServerStatus'
import Select from '@/components/ui/Select'
import StatusDot from '@/components/ui/StatusDot'
import SearchBar from '@/components/map/SearchBar'
import { useMapFocus } from '@/hooks/useMapFocus'

export default function TopBar() {
  const navigate = useNavigate()
  const { cities, selectedCity, setSelectedCity, loadCityData, citiesLoading } = useApp()
  const { status, refresh } = useServerStatus()
  const { focusPoint } = useMapFocus()

  const handleCityChange = useCallback((cityKey: string) => {
    setSelectedCity(cityKey)
    loadCityData(cityKey)
  }, [setSelectedCity, loadCityData])

  return (
    <div className="pointer-events-auto panel flex items-center gap-3 px-3 py-2 flex-wrap md:flex-nowrap">
      <button
        onClick={() => navigate('/')}
        aria-label="Back to home"
        className="p-1.5 rounded-lg hover:bg-[rgb(var(--surface-raised)/0.6)] transition-colors flex-shrink-0"
      >
        <ArrowLeft className="w-4 h-4 text-text-secondary" />
      </button>
      <div className="flex flex-col leading-tight flex-shrink-0">
        <span className="font-heading font-semibold text-text-primary text-sm tracking-tight">X-PAND.AI</span>
        <span className="text-[10px] text-text-muted">Geospatial Intelligence</span>
      </div>

      <div className="w-px h-6 bg-[rgb(var(--line)/0.1)] flex-shrink-0 hidden md:block" />

      <Select
        value={selectedCity}
        onValueChange={handleCityChange}
        aria-label="City"
        placeholder={citiesLoading ? 'Loading…' : 'Select a city'}
        options={cities.map(c => ({ value: c.key, label: c.name, hint: `${c.cell_count.toLocaleString()} cells` }))}
      />

      <div className="flex-1 min-w-[200px]">
        <SearchBar onSelect={(lat, lon) => focusPoint(lat, lon, 15)} />
      </div>

      <StatusDot status={status} compact onRetry={refresh} />
    </div>
  )
}
