import { cn } from '@/lib/utils'

export default function Spinner({ className, label = 'Loading' }: { className?: string; label?: string }) {
  return (
    <div
      role="status"
      className={cn('w-4 h-4 rounded-full animate-spin-slow flex-shrink-0', className)}
      style={{ border: '2px solid rgb(var(--line) / 0.1)', borderTopColor: 'rgb(var(--accent))' }}
    >
      <span className="sr-only">{label}</span>
    </div>
  )
}
