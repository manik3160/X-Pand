import { useState, useRef, useEffect, useCallback, useId } from 'react'
import { Search, X, Loader2, MapPin } from 'lucide-react'
import { searchLocation } from '@/api/client'
import type { SearchResult } from '@/types/api'

interface SearchBarProps {
  onSelect: (lat: number, lon: number, displayName: string) => void
}

/**
 * ARIA 1.2 combobox: arrow keys move through results, Enter selects,
 * Escape closes. Sits inline in the dashboard's top bar (no longer
 * absolutely positioned over the map).
 */
export default function SearchBar({ onSelect }: SearchBarProps) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [open, setOpen] = useState(false)
  const [activeIndex, setActiveIndex] = useState(-1)
  const inputRef = useRef<HTMLInputElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const listId = useId()

  const doSearch = useCallback(async (q: string) => {
    if (q.trim().length < 2) {
      setResults([])
      setOpen(false)
      setError(null)
      return
    }
    setLoading(true)
    setError(null)
    try {
      const res = await searchLocation(q.trim(), 5)
      setResults(res)
      setOpen(true)
      setActiveIndex(-1)
    } catch (err) {
      console.error('Search failed:', err)
      setResults([])
      setOpen(true)
      setError(err instanceof Error ? err.message : 'Search failed.')
    } finally {
      setLoading(false)
    }
  }, [])

  const handleInputChange = (value: string) => {
    setQuery(value)
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => doSearch(value), 350)
  }

  const handleSelect = useCallback((result: SearchResult) => {
    setQuery(result.display_name.split(',').slice(0, 2).join(', '))
    setOpen(false)
    setResults([])
    onSelect(result.lat, result.lon, result.display_name)
  }, [onSelect])

  const handleClear = () => {
    setQuery('')
    setResults([])
    setOpen(false)
    setError(null)
    inputRef.current?.focus()
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (!open || results.length === 0) return
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setActiveIndex(i => (i + 1) % results.length)
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActiveIndex(i => (i <= 0 ? results.length - 1 : i - 1))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      if (activeIndex >= 0) handleSelect(results[activeIndex])
    } else if (e.key === 'Escape') {
      setOpen(false)
    }
  }

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  return (
    <div ref={containerRef} className="relative w-full">
      <div
        role="combobox"
        aria-expanded={open}
        aria-haspopup="listbox"
        aria-owns={listId}
        aria-controls={listId}
        className="flex items-center gap-2 h-9 px-3 rounded-lg transition-colors"
        style={{
          background: 'rgb(var(--surface-raised) / 0.5)',
          border: '1px solid rgb(var(--line) / 0.12)',
        }}
      >
        <Search className="w-4 h-4 text-text-muted flex-shrink-0" aria-hidden="true" />
        <input
          ref={inputRef}
          type="text"
          role="textbox"
          aria-autocomplete="list"
          aria-activedescendant={activeIndex >= 0 ? `${listId}-opt-${activeIndex}` : undefined}
          value={query}
          onChange={(e) => handleInputChange(e.target.value)}
          onKeyDown={handleKeyDown}
          onFocus={() => { if (results.length > 0 || error) setOpen(true) }}
          placeholder="Search any location in India…"
          className="flex-1 min-w-0 bg-transparent text-sm text-text-primary placeholder:text-text-muted outline-none font-sans"
          autoComplete="off"
          spellCheck={false}
        />
        {loading && <Loader2 className="w-4 h-4 text-accent animate-spin flex-shrink-0" aria-hidden="true" />}
        {query && !loading && (
          <button onClick={handleClear} aria-label="Clear search" className="p-0.5 rounded-md hover:bg-[rgb(var(--line)/0.08)] transition-colors">
            <X className="w-3.5 h-3.5 text-text-muted" />
          </button>
        )}
      </div>

      {open && (
        <div
          id={listId}
          role="listbox"
          className="absolute left-0 right-0 mt-1.5 rounded-xl overflow-hidden panel z-50"
        >
          {error && (
            <div className="px-4 py-3 text-xs" style={{ color: 'rgb(var(--profit-low))' }}>{error}</div>
          )}
          {!error && results.length === 0 && !loading && (
            <div className="px-4 py-3 text-xs text-text-muted">No places found.</div>
          )}
          {results.map((r, i) => {
            const shortName = r.display_name.split(',').slice(0, 3).join(',').trim()
            return (
              <button
                key={`${r.lat}-${r.lon}-${i}`}
                id={`${listId}-opt-${i}`}
                role="option"
                aria-selected={i === activeIndex}
                onClick={() => handleSelect(r)}
                onMouseEnter={() => setActiveIndex(i)}
                className="w-full flex items-start gap-3 px-4 py-3 text-left transition-colors"
                style={{
                  background: i === activeIndex ? 'rgb(var(--accent) / 0.08)' : 'transparent',
                  borderBottom: i < results.length - 1 ? '1px solid rgb(var(--line) / 0.04)' : 'none',
                }}
              >
                <MapPin className="w-4 h-4 text-accent mt-0.5 flex-shrink-0" aria-hidden="true" />
                <div className="min-w-0 flex-1">
                  <div className="text-sm text-text-primary truncate leading-tight">{shortName}</div>
                  <div className="text-[11px] text-text-muted mt-0.5 font-mono tabular-nums">
                    {r.lat.toFixed(4)}, {r.lon.toFixed(4)}
                  </div>
                </div>
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}
