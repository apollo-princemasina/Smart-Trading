import { cn } from '@/utils/cn'

interface SkeletonProps {
  className?: string
  variant?:  'line' | 'block' | 'circle'
  rows?:     number
}

function SkeletonBase({ className }: { className: string }) {
  return (
    <div
      className={cn(
        'bg-navy-700 rounded animate-pulse',
        className,
      )}
    />
  )
}

export function Skeleton({ className, variant = 'line', rows = 1 }: SkeletonProps) {
  if (variant === 'circle') {
    return <SkeletonBase className={cn('rounded-full', className)} />
  }
  if (variant === 'block') {
    return <SkeletonBase className={cn('h-24', className)} />
  }
  if (rows > 1) {
    return (
      <div className="space-y-2">
        {Array.from({ length: rows }).map((_, i) => (
          <SkeletonBase
            key={i}
            className={cn(
              'h-3',
              i === rows - 1 && 'w-3/4',
              className,
            )}
          />
        ))}
      </div>
    )
  }
  return <SkeletonBase className={cn('h-3', className)} />
}

export function SkeletonCard({ className }: { className?: string }) {
  return (
    <div className={cn('bg-surface border border-border rounded-lg p-4 space-y-3', className)}>
      <Skeleton className="w-1/3 h-3" />
      <Skeleton rows={3} />
    </div>
  )
}

export function SkeletonGauge({ size = 120 }: { size?: number }) {
  return (
    <div
      className="rounded-full bg-navy-700 animate-pulse mx-auto"
      style={{ width: size, height: size / 2 }}
    />
  )
}
