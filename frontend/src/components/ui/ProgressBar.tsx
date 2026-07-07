import { cn } from '@/utils/cn'

interface ProgressBarProps {
  value:      number   // 0-100 or 0-1
  max?:       number   // default 1 or 100 depending on value scale
  color?:     string   // tailwind bg- class
  className?: string
  height?:    'thin' | 'normal' | 'thick'
  animate?:   boolean
  label?:     string
  showValue?: boolean
}

export function ProgressBar({
  value,
  max,
  color,
  className,
  height = 'normal',
  animate = true,
  label,
  showValue,
}: ProgressBarProps) {
  const pct = max != null
    ? (value / max) * 100
    : value <= 1
      ? value * 100
      : value

  const clampedPct = Math.min(100, Math.max(0, pct))

  const autoColor =
    clampedPct >= 70 ? 'bg-buy' :
    clampedPct >= 40 ? 'bg-wait' :
    'bg-sell'

  const heights = { thin: 'h-0.5', normal: 'h-1.5', thick: 'h-2.5' }

  return (
    <div className={className}>
      {(label || showValue) && (
        <div className="flex items-center justify-between mb-1">
          {label    && <span className="label">{label}</span>}
          {showValue && <span className="num text-secondary text-xs">{Math.round(clampedPct)}%</span>}
        </div>
      )}
      <div className={cn('bg-navy-700 rounded-full overflow-hidden', heights[height])}>
        <div
          className={cn(
            'h-full rounded-full',
            color ?? autoColor,
            animate && 'transition-all duration-700 ease-out',
          )}
          style={{ width: `${clampedPct}%` }}
          role="progressbar"
          aria-valuenow={Math.round(clampedPct)}
          aria-valuemin={0}
          aria-valuemax={100}
        />
      </div>
    </div>
  )
}

export function RegimeBar({
  label,
  value,
  color,
  className,
}: {
  label:      string
  value:      number
  color:      string
  className?: string
}) {
  const pct = value <= 1 ? Math.round(value * 100) : Math.round(value)
  return (
    <div className={cn('flex items-center gap-2', className)}>
      <div className="h-1.5 flex-1 bg-navy-700 rounded-full overflow-hidden">
        <div
          className={cn('h-full rounded-full transition-all duration-700', color)}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className={cn('num text-xs w-8 text-right font-semibold', color.replace('bg-', 'text-'))}>
        {pct}%
      </span>
    </div>
  )
}
