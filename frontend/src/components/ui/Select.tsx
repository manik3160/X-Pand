import * as RadixSelect from '@radix-ui/react-select'
import { Check, ChevronDown } from 'lucide-react'
import { cn } from '@/lib/utils'

export interface SelectOption {
  value: string
  label: string
  hint?: string
}

interface SelectProps {
  value: string
  onValueChange: (value: string) => void
  options: SelectOption[]
  placeholder?: string
  triggerClassName?: string
  'aria-label'?: string
}

/**
 * A keyboard-accessible, styled dropdown (arrow keys, typeahead, Escape)
 * replacing the hand-rolled div-based city selector.
 */
export default function Select({ value, onValueChange, options, placeholder, triggerClassName, ...aria }: SelectProps) {
  return (
    <RadixSelect.Root value={value} onValueChange={onValueChange}>
      <RadixSelect.Trigger
        className={cn(
          'inline-flex items-center justify-between gap-2 h-9 px-3 rounded-lg text-sm text-text-primary outline-none',
          'bg-[rgb(var(--surface-raised)/0.5)] border border-[rgb(var(--line)/0.12)] hover:border-[rgb(var(--line)/0.2)] transition-colors',
          'data-[state=open]:border-accent',
          triggerClassName
        )}
        aria-label={aria['aria-label']}
      >
        <RadixSelect.Value placeholder={placeholder} />
        <RadixSelect.Icon>
          <ChevronDown className="w-3.5 h-3.5 text-text-muted" />
        </RadixSelect.Icon>
      </RadixSelect.Trigger>
      <RadixSelect.Portal>
        <RadixSelect.Content
          className="panel z-50 overflow-hidden"
          position="popper"
          sideOffset={6}
        >
          <RadixSelect.Viewport className="p-1 max-h-[min(320px,var(--radix-select-content-available-height))]">
            {options.map(opt => (
              <RadixSelect.Item
                key={opt.value}
                value={opt.value}
                className={cn(
                  'flex items-center justify-between gap-3 px-3 py-2 rounded-lg text-sm text-text-primary outline-none cursor-pointer select-none',
                  'data-[highlighted]:bg-[rgb(var(--surface-raised)/0.7)] data-[state=checked]:text-accent'
                )}
              >
                <span className="flex items-center gap-2">
                  <span className="w-3.5 h-3.5 flex-shrink-0 text-accent">
                    <RadixSelect.ItemIndicator>
                      <Check className="w-3.5 h-3.5" />
                    </RadixSelect.ItemIndicator>
                  </span>
                  <RadixSelect.ItemText>{opt.label}</RadixSelect.ItemText>
                </span>
                {opt.hint && <span className="text-xs text-text-muted tabular-nums">{opt.hint}</span>}
              </RadixSelect.Item>
            ))}
          </RadixSelect.Viewport>
        </RadixSelect.Content>
      </RadixSelect.Portal>
    </RadixSelect.Root>
  )
}
