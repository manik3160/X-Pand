import { AlertTriangle } from 'lucide-react'

interface ErrorBannerProps {
  message: string
  onRetry?: () => void
  className?: string
}

export default function ErrorBanner({ message, onRetry, className }: ErrorBannerProps) {
  return (
    <div
      role="alert"
      className={`flex items-start gap-2.5 rounded-xl p-3 ${className ?? ''}`}
      style={{ background: 'rgb(var(--profit-low) / 0.08)', border: '1px solid rgb(var(--profit-low) / 0.25)' }}
    >
      <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" style={{ color: 'rgb(var(--profit-low))' }} aria-hidden="true" />
      <div className="flex-1 min-w-0">
        <div className="text-xs font-medium" style={{ color: 'rgb(var(--profit-low))' }}>{message}</div>
        {onRetry && (
          <button onClick={onRetry} className="mt-1.5 text-xs font-semibold text-accent hover:underline">
            Retry
          </button>
        )}
      </div>
    </div>
  )
}
