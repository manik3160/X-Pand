import { useEffect, useRef, useState } from 'react'
import { fetchStatus } from '@/api/client'

export type ServerStatus = 'checking' | 'waking' | 'online' | 'offline'

interface ServerStatusInfo {
  status: ServerStatus
  citiesLoaded: number
  refresh: () => void
}

const WAKING_AFTER_MS = 3000
const POLL_INTERVAL_MS = 60_000

/**
 * Polls GET /status (unused until now) so the "API connected" / "Models
 * loaded" indicators reflect reality instead of always showing green.
 * Render's free tier can take up to ~60s to wake from sleep, so a slow
 * first response is reported as "waking", not "offline".
 */
export function useServerStatus(): ServerStatusInfo {
  const [status, setStatus] = useState<ServerStatus>('checking')
  const [citiesLoaded, setCitiesLoaded] = useState(0)
  const [tick, setTick] = useState(0)
  const reqId = useRef(0)

  useEffect(() => {
    const id = ++reqId.current

    // Don't flip straight to "checking" here — that would flicker the
    // status dot on every 60s re-poll even while still online. Only
    // change what's on screen once we know something (waking after a
    // delay, or the request actually resolves/rejects below).
    const wakingTimer = setTimeout(() => {
      if (reqId.current === id) setStatus(prev => (prev === 'online' ? prev : 'waking'))
    }, WAKING_AFTER_MS)

    fetchStatus()
      .then(data => {
        if (reqId.current !== id) return
        setStatus('online')
        setCitiesLoaded(data.cities_loaded?.length ?? 0)
      })
      .catch(() => {
        if (reqId.current !== id) return
        setStatus('offline')
      })
      .finally(() => clearTimeout(wakingTimer))

    return () => clearTimeout(wakingTimer)
  }, [tick])

  useEffect(() => {
    const interval = setInterval(() => {
      if (document.visibilityState === 'visible') setTick(t => t + 1)
    }, POLL_INTERVAL_MS)
    return () => clearInterval(interval)
  }, [])

  return { status, citiesLoaded, refresh: () => setTick(t => t + 1) }
}
