import { useState, useRef, useEffect, useCallback } from 'react'
import { Search, X, Loader2, MapPin } from 'lucide-react'
import { searchLocation } from '@/api/client'
import type { SearchResult } from '@/types/api'

interface SearchBarProps {
  onSelect: (lat: number, lon: number, displayName: string) => void
}

export default function SearchBar({ onSelect }: SearchBarProps) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [open, setOpen] = useState(false)
  const [focused, setFocused] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Debounced search
  const doSearch = useCallback(async (q: string) => {
    if (q.trim().length < 2) {
      setResults([])
      setOpen(false)
      return
    }
    setLoading(true)
    try {
      const res = await searchLocation(q.trim(), 5)
      setResults(res)
      setOpen(res.length > 0)
    } catch (err) {
      console.error('Search failed:', err)
      setResults([])
    } finally {
      setLoading(false)
    }
  }, [])

  const handleInputChange = (value: string) => {
    setQuery(value)
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => doSearch(value), 350)
  }

  const handleSelect = (result: SearchResult) => {
    setQuery(result.display_name.split(',').slice(0, 2).join(', '))
    setOpen(false)
    setResults([])
    onSelect(result.lat, result.lon, result.display_name)
  }

  const handleClear = () => {
    setQuery('')
    setResults([])
    setOpen(false)
    inputRef.current?.focus()
  }

  // Close dropdown when clicking outside
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
    <div
      ref={containerRef}
      className="absolute top-4 left-1/2 -translate-x-1/2 z-[1001] w-[380px]"
    >
      {/* Search Input */}
      <div
        className="flex items-center gap-2 px-3.5 py-2.5 rounded-xl transition-all duration-200"
        style={{
          background: focused
            ? 'rgba(10, 10, 15, 0.95)'
            : 'rgba(10, 10, 15, 0.85)',
          border: focused
            ? '1px solid rgba(0, 255, 136, 0.3)'
            : '1px solid rgba(255, 255, 255, 0.1)',
          backdropFilter: 'blur(20px)',
          WebkitBackdropFilter: 'blur(20px)',
          boxShadow: focused
            ? '0 8px 32px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(0, 255, 136, 0.1)'
            : '0 4px 24px rgba(0, 0, 0, 0.4)',
        }}
      >
        <Search className="w-4 h-4 text-text-muted flex-shrink-0" />
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => handleInputChange(e.target.value)}
          onFocus={() => {
            setFocused(true)
            if (results.length > 0) setOpen(true)
          }}
          onBlur={() => setFocused(false)}
          placeholder="Search any location in India..."
          className="flex-1 bg-transparent text-sm text-white placeholder:text-text-muted outline-none font-sans"
          autoComplete="off"
          spellCheck={false}
        />
        {loading && (
          <Loader2 className="w-4 h-4 text-accent animate-spin flex-shrink-0" />
        )}
        {query && !loading && (
          <button
            onClick={handleClear}
            className="p-0.5 rounded-md hover:bg-[rgba(255,255,255,0.08)] transition-colors"
          >
            <X className="w-3.5 h-3.5 text-text-muted" />
          </button>
        )}
      </div>

      {/* Dropdown Results */}
      {open && results.length > 0 && (
        <div
          className="mt-1.5 rounded-xl overflow-hidden"
          style={{
            background: 'rgba(10, 10, 15, 0.96)',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            backdropFilter: 'blur(24px)',
            WebkitBackdropFilter: 'blur(24px)',
            boxShadow: '0 12px 48px rgba(0, 0, 0, 0.6)',
          }}
        >
          {results.map((r, i) => {
            // Truncate display name to first 3 segments
            const shortName = r.display_name.split(',').slice(0, 3).join(',').trim()
            return (
              <button
                key={`${r.lat}-${r.lon}-${i}`}
                onClick={() => handleSelect(r)}
                className="w-full flex items-start gap-3 px-4 py-3 text-left transition-all hover:bg-[rgba(0,255,136,0.04)]"
                style={{
                  borderBottom:
                    i < results.length - 1
                      ? '1px solid rgba(255, 255, 255, 0.04)'
                      : 'none',
                }}
              >
                <MapPin className="w-4 h-4 text-accent mt-0.5 flex-shrink-0" />
                <div className="min-w-0 flex-1">
                  <div className="text-sm text-white truncate leading-tight">
                    {shortName}
                  </div>
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
