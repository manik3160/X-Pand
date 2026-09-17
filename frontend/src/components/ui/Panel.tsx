import { useState, type ReactNode } from 'react'
import { ChevronDown } from 'lucide-react'
import { cn } from '@/lib/utils'

interface PanelProps {
  title?: string
  icon?: ReactNode
  headerRight?: ReactNode
  collapsible?: boolean
  defaultCollapsed?: boolean
  className?: string
  bodyClassName?: string
  children: ReactNode
}

/**
 * The floating translucent surface every dashboard card sits on. Uses the
 * `.panel` token class from index.css (blur + border + shadow).
 */
export default function Panel({
  title,
  icon,
  headerRight,
  collapsible = false,
  defaultCollapsed = false,
  className,
  bodyClassName,
  children,
}: PanelProps) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed)
  const bodyId = title ? `panel-body-${title.replace(/\s+/g, '-').toLowerCase()}` : undefined

  return (
    <div className={cn('panel overflow-hidden pointer-events-auto', className)}>
      {title && (
        <div className="flex items-center justify-between px-4 py-3" style={{ borderBottom: collapsed ? 'none' : '1px solid rgb(var(--line) / 0.08)' }}>
          <div className="flex items-center gap-2 min-w-0">
            {icon}
            <span className="text-[11px] font-semibold text-text-muted uppercase tracking-[0.1em] truncate">{title}</span>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            {headerRight}
            {collapsible && (
              <button
                type="button"
                onClick={() => setCollapsed(c => !c)}
                aria-expanded={!collapsed}
                aria-controls={bodyId}
                aria-label={collapsed ? `Expand ${title}` : `Collapse ${title}`}
                className="p-1 rounded-md hover:bg-[rgb(var(--surface-raised)/0.6)] transition-colors"
              >
                <ChevronDown className={cn('w-3.5 h-3.5 text-text-muted transition-transform', collapsed && '-rotate-90')} />
              </button>
            )}
          </div>
        </div>
      )}
      {!collapsed && (
        <div id={bodyId} className={bodyClassName}>
          {children}
        </div>
      )}
    </div>
  )
}
