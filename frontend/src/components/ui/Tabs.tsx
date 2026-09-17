import * as RadixTabs from '@radix-ui/react-tabs'
import { cn } from '@/lib/utils'

export const Tabs = RadixTabs.Root

export function TabsList({ className, ...props }: RadixTabs.TabsListProps) {
  return (
    <RadixTabs.List
      className={cn('flex items-center gap-1 px-2 pt-2', className)}
      {...props}
    />
  )
}

export function TabsTrigger({ className, children, ...props }: RadixTabs.TabsTriggerProps) {
  return (
    <RadixTabs.Trigger
      className={cn(
        'flex-1 px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors outline-none',
        'text-text-muted hover:text-text-secondary',
        'data-[state=active]:bg-[rgb(var(--surface-raised)/0.7)] data-[state=active]:text-text-primary',
        className
      )}
      {...props}
    >
      {children}
    </RadixTabs.Trigger>
  )
}

export function TabsContent({ className, ...props }: RadixTabs.TabsContentProps) {
  return <RadixTabs.Content className={cn('outline-none', className)} {...props} />
}
