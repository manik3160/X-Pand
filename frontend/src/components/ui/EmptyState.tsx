import type { LucideIcon } from 'lucide-react'

interface EmptyStateProps {
  icon: LucideIcon
  title: string
  hint?: string
}

export default function EmptyState({ icon: Icon, title, hint }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center text-center gap-1.5 py-8 px-4">
      <Icon className="w-5 h-5 text-text-muted mb-1" aria-hidden="true" />
      <div className="text-xs font-medium text-text-secondary">{title}</div>
      {hint && <div className="text-[11px] text-text-muted max-w-[220px]">{hint}</div>}
    </div>
  )
}
