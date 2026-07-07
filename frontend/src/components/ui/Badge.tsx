import { cn } from '@/utils/cn'
import type { Recommendation, Direction, ComponentStatus, SystemStatus } from '@/types/api'

type BadgeVariant =
  | 'buy' | 'sell' | 'wait'
  | 'ok' | 'warning' | 'danger' | 'neutral'
  | 'manipulation' | 'expansion' | 'consolidation'
  | 'gold' | 'blue' | 'muted'

type BadgeSize = 'xs' | 'sm' | 'md' | 'lg' | 'xl'

interface BadgeProps {
  variant?:   BadgeVariant
  size?:      BadgeSize
  children:   React.ReactNode
  className?: string
  dot?:       boolean
  pulse?:     boolean
}

const variantClasses: Record<BadgeVariant, string> = {
  buy:           'text-buy bg-buy-bg border-buy-dim',
  sell:          'text-sell bg-sell-bg border-sell-dim',
  wait:          'text-wait bg-wait-bg border-wait-dim',
  ok:            'text-buy bg-buy-bg border-buy-dim',
  warning:       'text-wait bg-wait-bg border-wait-dim',
  danger:        'text-sell bg-sell-bg border-sell-dim',
  neutral:       'text-secondary bg-navy-800 border-border',
  manipulation:  'text-manipulation bg-[#1E0A3C] border-[#5B21B6]',
  expansion:     'text-expansion bg-[#0A1E3C] border-[#1D4ED8]',
  consolidation: 'text-muted bg-navy-800 border-border',
  gold:          'text-gold bg-gold-glow border-gold/30',
  blue:          'text-signal bg-[#0A1E3C] border-[#1D4ED8]',
  muted:         'text-muted bg-navy-800 border-border/50',
}

const sizeClasses: Record<BadgeSize, string> = {
  xs: 'text-[10px] px-1.5 py-0.5 font-semibold tracking-wide',
  sm: 'text-[11px] px-2   py-0.5 font-semibold tracking-wide',
  md: 'text-xs     px-2.5 py-1   font-semibold tracking-wide',
  lg: 'text-sm     px-3   py-1   font-bold     tracking-wider',
  xl: 'text-base   px-4   py-1.5 font-bold     tracking-widest',
}

export function Badge({ variant = 'neutral', size = 'sm', children, className, dot, pulse }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded border uppercase',
        variantClasses[variant],
        sizeClasses[size],
        className,
      )}
    >
      {dot && (
        <span
          className={cn(
            'w-1.5 h-1.5 rounded-full shrink-0',
            {
              'bg-buy':           variant === 'buy' || variant === 'ok',
              'bg-sell':          variant === 'sell' || variant === 'danger',
              'bg-wait':          variant === 'wait' || variant === 'warning',
              'bg-manipulation':  variant === 'manipulation',
              'bg-expansion':     variant === 'expansion',
              'bg-muted':         variant === 'neutral' || variant === 'muted' || variant === 'consolidation',
              'bg-gold':          variant === 'gold',
              'animate-ping-slow': pulse,
            },
          )}
        />
      )}
      {children}
    </span>
  )
}

// ── Semantic convenience wrappers ─────────────────────────────────────────────

export function DirectionBadge({ direction, size }: { direction: Direction; size?: BadgeSize }) {
  const map: Record<Direction, BadgeVariant> = { BUY: 'buy', SELL: 'sell', HOLD: 'wait' }
  return (
    <Badge variant={map[direction] ?? 'neutral'} size={size ?? 'sm'} dot>
      {direction}
    </Badge>
  )
}

export function RecommendationBadge({ rec, size }: { rec: Recommendation; size?: BadgeSize }) {
  const map: Record<Recommendation, BadgeVariant> = { BUY: 'buy', SELL: 'sell', WAIT: 'wait' }
  return (
    <Badge variant={map[rec] ?? 'neutral'} size={size ?? 'md'} dot>
      {rec}
    </Badge>
  )
}

export function StatusBadge({ status, label }: { status: ComponentStatus | SystemStatus; label?: string }) {
  const map: Record<string, BadgeVariant> = {
    ok:          'ok',
    operational: 'ok',
    degraded:    'warning',
    error:       'danger',
    stopped:     'neutral',
  }
  return (
    <Badge variant={map[status] ?? 'neutral'} size="sm" dot>
      {label ?? status}
    </Badge>
  )
}
