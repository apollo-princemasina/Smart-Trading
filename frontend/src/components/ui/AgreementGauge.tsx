import { cn } from '@/utils/cn'
import { scoreToPercent } from '@/utils/format'

interface AgreementGaugeProps {
  agreement:  number  // 0-1
  conflict:   number  // 0-1
  className?: string
  compact?:   boolean
}

export function AgreementGauge({ agreement, conflict, className, compact }: AgreementGaugeProps) {
  const agrPct = scoreToPercent(agreement)
  const conPct = scoreToPercent(conflict)

  return (
    <div className={cn('space-y-2', className)}>
      {/* Agreement */}
      <div>
        <div className="flex items-center justify-between mb-1">
          <span className="label">Agreement</span>
          <span className="num text-buy text-xs font-semibold">{agrPct}%</span>
        </div>
        <div className="h-1.5 bg-navy-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-buy rounded-full transition-all duration-700"
            style={{ width: `${agrPct}%` }}
          />
        </div>
      </div>

      {/* Conflict */}
      {!compact && (
        <div>
          <div className="flex items-center justify-between mb-1">
            <span className="label">Conflict</span>
            <span className="num text-sell text-xs font-semibold">{conPct}%</span>
          </div>
          <div className="h-1.5 bg-navy-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-sell rounded-full transition-all duration-700"
              style={{ width: `${conPct}%` }}
            />
          </div>
        </div>
      )}
    </div>
  )
}

interface ScoreBarProps {
  label:      string
  value:      number  // 0-1
  color?:     string
  className?: string
}

export function ScoreBar({ label, value, color = 'bg-gold', className }: ScoreBarProps) {
  const pct = scoreToPercent(value)
  return (
    <div className={cn('flex items-center gap-2', className)}>
      <span className="label w-20 shrink-0">{label}</span>
      <div className="flex-1 h-1 bg-navy-700 rounded-full overflow-hidden">
        <div
          className={cn('h-full rounded-full transition-all duration-700', color)}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="num text-secondary text-xs w-8 text-right">{pct}%</span>
    </div>
  )
}
