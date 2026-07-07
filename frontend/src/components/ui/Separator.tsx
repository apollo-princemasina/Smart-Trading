import { cn } from '@/utils/cn'

interface SeparatorProps {
  className?: string
  vertical?:  boolean
  label?:     string
}

export function Separator({ className, vertical, label }: SeparatorProps) {
  if (label) {
    return (
      <div className={cn('flex items-center gap-3', className)}>
        <div className="flex-1 h-px bg-border" />
        <span className="label">{label}</span>
        <div className="flex-1 h-px bg-border" />
      </div>
    )
  }
  if (vertical) {
    return <div className={cn('w-px bg-border self-stretch', className)} />
  }
  return <div className={cn('h-px bg-border', className)} />
}
