import { useRef, type ReactNode } from 'react'
import { motion, useReducedMotion, type PanInfo } from 'framer-motion'

export type SheetSnap = 'peek' | 'half' | 'full'

const SNAP_HEIGHT: Record<SheetSnap, string> = {
  peek: '64px',
  half: '48dvh',
  full: '88dvh',
}

const ORDER: SheetSnap[] = ['peek', 'half', 'full']

interface BottomSheetProps {
  snap: SheetSnap
  onSnapChange: (snap: SheetSnap) => void
  header?: ReactNode
  children: ReactNode
}

/**
 * A draggable bottom sheet with three snap points. The handle both
 * supports a drag gesture (flick up/down moves one snap step — this is
 * a direction gesture, not a 1:1 finger-follow drag, to keep the
 * implementation simple and predictable) and acts as a real,
 * keyboard-operable button that cycles snap points on click/Enter/Space.
 */
export default function BottomSheet({ snap, onSnapChange, header, children }: BottomSheetProps) {
  const reduceMotion = useReducedMotion()
  const draggingRef = useRef(false)

  const cycle = () => {
    if (draggingRef.current) return // the drag gesture already changed it — don't also toggle on the resulting click
    const idx = ORDER.indexOf(snap)
    onSnapChange(ORDER[(idx + 1) % ORDER.length])
  }

  const handleDragEnd = (_: unknown, info: PanInfo) => {
    const idx = ORDER.indexOf(snap)
    const draggedUp = info.offset.y < -40 || info.velocity.y < -300
    const draggedDown = info.offset.y > 40 || info.velocity.y > 300
    if (draggedUp && idx < ORDER.length - 1) onSnapChange(ORDER[idx + 1])
    else if (draggedDown && idx > 0) onSnapChange(ORDER[idx - 1])
    // Swallow the click framer-motion synthesizes right after a drag.
    requestAnimationFrame(() => { draggingRef.current = false })
  }

  return (
    <motion.div
      className="fixed left-0 right-0 bottom-0 z-sheet panel pointer-events-auto flex flex-col overflow-hidden"
      style={{ borderBottomLeftRadius: 0, borderBottomRightRadius: 0, borderLeft: 'none', borderRight: 'none', borderBottom: 'none' }}
      animate={{ height: SNAP_HEIGHT[snap] }}
      transition={{ duration: reduceMotion ? 0 : 0.25, ease: [0.2, 0.8, 0.2, 1] }}
    >
      <motion.div
        role="button"
        tabIndex={0}
        aria-label={snap === 'full' ? 'Collapse panel' : 'Expand panel'}
        onClick={cycle}
        onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); cycle() } }}
        drag="y"
        dragConstraints={{ top: 0, bottom: 0 }}
        dragElastic={0.2}
        onDragStart={() => { draggingRef.current = true }}
        onDragEnd={handleDragEnd}
        className="flex flex-col items-center gap-1.5 pt-2 pb-1.5 flex-shrink-0 cursor-grab active:cursor-grabbing touch-none outline-none"
      >
        <div className="w-9 h-1 rounded-full" style={{ background: 'rgb(var(--line) / 0.25)' }} />
        {header}
      </motion.div>
      <div className="flex-1 overflow-y-auto" style={{ overscrollBehavior: 'contain' }}>
        {children}
      </div>
    </motion.div>
  )
}
