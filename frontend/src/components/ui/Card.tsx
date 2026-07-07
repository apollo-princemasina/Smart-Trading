import { cn } from '@/utils/cn'

interface CardProps {
  children:   React.ReactNode
  className?: string
  title?:     string
  subtitle?:  string
  action?:    React.ReactNode
  noPadding?: boolean
  variant?:   'default' | 'elevated' | 'ghost' | 'bordered'
}

export function Card({
  children,
  className,
  title,
  subtitle,
  action,
  noPadding,
  variant = 'default',
}: CardProps) {
  const base = cn(
    'rounded-lg overflow-hidden',
    {
      'bg-surface border border-border':           variant === 'default',
      'bg-elevated border border-border/50':       variant === 'elevated',
      'bg-transparent':                            variant === 'ghost',
      'bg-surface border border-border-subtle':    variant === 'bordered',
    },
    className,
  )

  return (
    <div className={base}>
      {(title || action) && (
        <div className="flex items-center justify-between px-4 py-3 border-b border-border/60">
          <div>
            <p className="label">{title}</p>
            {subtitle && <p className="caption mt-0.5">{subtitle}</p>}
          </div>
          {action && <div className="shrink-0">{action}</div>}
        </div>
      )}
      <div className={noPadding ? '' : 'p-4'}>{children}</div>
    </div>
  )
}

export function CardSection({
  children,
  className,
  label,
}: {
  children: React.ReactNode
  className?: string
  label?: string
}) {
  return (
    <div className={cn('border-t border-border/60', className)}>
      {label && <p className="label px-4 pt-3 pb-1">{label}</p>}
      <div className="px-4 pb-3">{children}</div>
    </div>
  )
}
