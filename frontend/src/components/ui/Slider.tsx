import * as RadixSlider from '@radix-ui/react-slider'

interface SliderProps {
  id: string
  label: string
  value: number
  onChange: (value: number) => void
  min: number
  max: number
  step: number
  formatValue?: (value: number) => string
}

/** A labeled, keyboard-accessible slider (arrow keys move it; screen readers get aria-valuetext). */
export default function Slider({ id, label, value, onChange, min, max, step, formatValue }: SliderProps) {
  const display = formatValue ? formatValue(value) : String(value)
  return (
    <div className="mb-4 last:mb-0">
      <div className="flex justify-between items-center mb-2">
        <label htmlFor={id} className="text-xs text-text-secondary">{label}</label>
        <span className="text-xs font-heading font-semibold text-accent tabular-nums">{display}</span>
      </div>
      <RadixSlider.Root
        id={id}
        className="relative flex items-center select-none touch-none w-full h-4"
        value={[value]}
        onValueChange={([v]) => onChange(v)}
        min={min}
        max={max}
        step={step}
        aria-label={label}
        aria-valuetext={display}
      >
        <RadixSlider.Track className="relative grow rounded-full h-1" style={{ background: 'rgb(var(--line) / 0.1)' }}>
          <RadixSlider.Range className="absolute rounded-full h-full bg-accent" />
        </RadixSlider.Track>
        <RadixSlider.Thumb
          className="block w-4 h-4 rounded-full bg-text border-2 border-accent outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2"
          style={{ background: 'rgb(var(--text))' }}
        />
      </RadixSlider.Root>
    </div>
  )
}
