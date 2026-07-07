import { cn } from '@/utils/cn'

interface LivePulseProps {
  connected?: boolean
  className?: string
  showLabel?: boolean
  size?:      'sm' | 'md'
}

export function LivePulse({ connected = true, className, showLabel = false, size = 'sm' }: LivePulseProps) {
  const dot = size === 'md' ? 'w-2.5 h-2.5' : 'w-2 h-2'
  const ring = size === 'md' ? 'w-5 h-5' : 'w-4 h-4'

  return (
    <div className={cn('flex items-center gap-1.5', className)}>
      <div className="relative flex items-center justify-center">
        {connected && (
          <div
            className={cn(
              'absolute rounded-full bg-buy/30 animate-ping',
              ring,
            )}
            style={{ animationDuration: '1.8s' }}
          />
        )}
        <div
          className={cn(
            'rounded-full relative z-10',
            dot,
            connected ? 'bg-buy live-dot' : 'bg-muted',
          )}
        />
      </div>
      {showLabel && (
        <span className={cn(
          'text-[10px] font-bold tracking-widest uppercase',
          connected ? 'text-buy' : 'text-muted',
        )}>
          {connected ? 'LIVE' : 'DISCONNECTED'}
        </span>
      )}
    </div>
  )
}

export function ConnectionBanner({ connected }: { connected: boolean }) {
  if (connected) return null
  return (
    <div className="bg-sell-bg border-b border-sell-dim px-4 py-1.5 flex items-center gap-2">
      <div className="w-1.5 h-1.5 rounded-full bg-sell" />
      <span className="text-sell text-xs font-medium">
        Disconnected — reconnecting…
      </span>
    </div>
  )
}
