import { useEffect, useRef, type ReactNode } from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion'
import { X } from 'lucide-react'

interface SideSheetProps {
  open: boolean
  onClose: () => void
  title: ReactNode
  children: ReactNode
}

/**
 * A non-modal panel sliding in from the right, over the map (which stays
 * interactive underneath — a modal Dialog would trap focus and block map
 * panning, which we don't want here).
 */
export default function SideSheet({ open, onClose, title, children }: SideSheetProps) {
  const closeRef = useRef<HTMLButtonElement>(null)
  const reduceMotion = useReducedMotion()

  useEffect(() => {
    if (!open) return
    closeRef.current?.focus()
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [open, onClose])

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          role="dialog"
          aria-label={typeof title === 'string' ? title : 'Details'}
          className="pointer-events-auto fixed top-3 bottom-3 right-3 w-[380px] max-w-[calc(100vw-24px)] panel overflow-y-auto z-sheet"
          initial={reduceMotion ? { opacity: 0 } : { x: 32, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={reduceMotion ? { opacity: 0 } : { x: 32, opacity: 0 }}
          transition={{ duration: reduceMotion ? 0.1 : 0.22, ease: [0.2, 0.8, 0.2, 1] }}
        >
          <div className="sticky top-0 z-10 flex items-center justify-between px-4 py-3 panel" style={{ borderRadius: 0, borderTop: 'none', borderLeft: 'none', borderRight: 'none' }}>
            <div className="text-sm font-heading font-semibold text-text-primary truncate">{title}</div>
            <button
              ref={closeRef}
              onClick={onClose}
              aria-label="Close details"
              className="p-1.5 rounded-lg hover:bg-[rgb(var(--surface-raised)/0.6)] transition-colors flex-shrink-0"
            >
              <X className="w-4 h-4 text-text-muted" />
            </button>
          </div>
          {children}
        </motion.div>
      )}
    </AnimatePresence>
  )
}
