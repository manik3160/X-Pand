import type { ServerStatus } from '@/hooks/useServerStatus'

const LABEL: Record<ServerStatus, string> = {
  online: 'API connected',
  waking: 'API waking up…',
  checking: 'Connecting to API…',
  offline: 'API unreachable',
}

const DOT_STATE: Record<ServerStatus, 'ok' | 'warn' | 'off'> = {
  online: 'ok',
  waking: 'warn',
  checking: 'warn',
  offline: 'off',
}

interface StatusDotProps {
  status: ServerStatus
  compact?: boolean
  onRetry?: () => void
}

export default function StatusDot({ status, compact = false, onRetry }: StatusDotProps) {
  return (
    <div className="flex items-center gap-2">
      <div className="live-dot-sm" data-state={DOT_STATE[status]} />
      {!compact && <span className="text-xs text-text-secondary">{LABEL[status]}</span>}
      {status === 'offline' && onRetry && (
        <button onClick={onRetry} className="text-xs font-semibold text-accent hover:underline">
          Retry
        </button>
      )}
    </div>
  )
}
