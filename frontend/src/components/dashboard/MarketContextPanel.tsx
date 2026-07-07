'use client'
import { useDashboard, useMarketRegime } from '@/hooks/useDashboard'
import { Badge } from '@/components/ui/Badge'
import { Skeleton } from '@/components/ui/Skeleton'
import { Separator } from '@/components/ui/Separator'
import { cn } from '@/utils/cn'
import { impactColor, impactLabel, formatPips } from '@/utils/format'
import type { RegimeData, RegimeSummary, EIEEvent } from '@/types/api'
import { useKillzone } from '@/hooks/useKillzone'

function regimeBadgeVariant(d: string): 'manipulation' | 'expansion' | 'consolidation' {
  if (d?.includes('MANIPULATION')) return 'manipulation'
  if (d?.includes('EXPANSION'))    return 'expansion'
  return 'consolidation'
}

function RegimeSection({ regime }: { regime: RegimeData | RegimeSummary | null }) {
  if (!regime) return <Skeleton rows={3} className="px-4 py-2" />

  const dominant = regime.dominant ?? '—'
  const scores   = (regime as RegimeData).scores ?? (regime as RegimeSummary).scores
  const atrPips  = (regime as RegimeData).atr_pips ?? (regime as RegimeSummary).atr_pips ?? null

  const scoreRows: [string, number, string][] = scores
    ? [
        ['MANIP',   scores.manipulation ?? 0, 'bg-manipulation'],
        ['EXPAND',  scores.expansion    ?? 0, 'bg-expansion'],
        ['CONSOL',  scores.consolidation ?? 0, 'bg-consolidation'],
      ]
    : []

  return (
    <div className="px-4 py-3 space-y-2.5">
      <div className="flex items-center justify-between">
        <span className="label">REGIME</span>
        <Badge variant={regimeBadgeVariant(dominant)} size="xs">
          {dominant.charAt(0) + dominant.slice(1).toLowerCase()}
        </Badge>
      </div>

      {scoreRows.map(([label, value, color]) => {
        const pct = Math.round(value <= 1 ? value * 100 : value)
        return (
          <div key={label} className="flex items-center gap-2">
            <span className="text-muted text-[10px] font-semibold w-14 shrink-0">{label}</span>
            <div className="flex-1 h-1 bg-navy-700 rounded-full overflow-hidden">
              <div
                className={cn('h-full rounded-full transition-all duration-700', color)}
                style={{ width: `${pct}%` }}
              />
            </div>
            <span className="num text-secondary text-[10px] w-7 text-right">{pct}%</span>
          </div>
        )
      })}

      {atrPips != null && (
        <div className="flex items-center justify-between pt-0.5">
          <span className="label">ATR</span>
          <span className="num text-secondary text-xs">{formatPips(atrPips)}</span>
        </div>
      )}
    </div>
  )
}

function ICTSection({ regime }: { regime: RegimeData | null }) {
  if (!regime || !('ict' in regime) || !regime.ict) return null
  const ict = regime.ict

  const flags: [string, boolean, string][] = [
    ['Liquidity Sweep',  ict.liquidity_sweep,  ict.sweep_direction ?? ''],
    ['ChoCH Detected',   ict.choch_detected,   ict.choch_direction ?? ''],
    ['BoS Detected',     ict.bos_detected,     ict.bos_direction ?? ''],
    ['FVG Active',       ict.fvg_active,        ict.fvg_direction ?? ''],
    ['Order Block',      ict.ob_active,         ict.ob_direction ?? ''],
  ]

  const active = flags.filter(([, v]) => v)
  if (!active.length) return null

  return (
    <div className="px-4 py-3">
      <span className="label block mb-2">ICT SIGNALS</span>
      <div className="space-y-1">
        {active.map(([label, , dir]) => (
          <div key={label} className="flex items-center justify-between">
            <span className="text-primary text-xs">{label}</span>
            {dir && <span className={cn('text-xs font-semibold', dir.includes('BUL') ? 'text-buy' : 'text-sell')}>{dir}</span>}
          </div>
        ))}
      </div>
    </div>
  )
}

function SessionSection() {
  const kz = useKillzone()
  return (
    <div className="px-4 py-3">
      <span className="label block mb-2">SESSION</span>
      <div className="flex items-center justify-between">
        <span className={cn('text-xs font-medium', kz.active ? 'text-primary' : 'text-secondary')}>
          {kz.name}
        </span>
        {kz.active && (
          <Badge variant="gold" size="xs" dot pulse>KILLZONE</Badge>
        )}
      </div>
      <p className="text-muted text-[10px] mt-1">{kz.utcRange}</p>
    </div>
  )
}

function EventsSection({ events }: { events: EIEEvent[] | undefined }) {
  if (!events?.length) {
    return (
      <div className="px-4 py-3">
        <span className="label block mb-2">UPCOMING EVENTS</span>
        <p className="text-muted text-xs">No events scheduled</p>
      </div>
    )
  }

  return (
    <div className="px-4 py-3">
      <span className="label block mb-2">UPCOMING EVENTS</span>
      <div className="space-y-2">
        {events.slice(0, 4).map((ev, i) => (
          <div key={i} className="flex items-start gap-2">
            <span className={cn('text-xs mt-0.5 shrink-0 font-mono', impactColor(ev.impact))}>
              {impactLabel(ev.impact)}
            </span>
            <div className="min-w-0">
              <p className="text-primary text-xs font-medium truncate">{ev.title}</p>
              <p className="text-muted text-[10px] num">{ev.currency} · {ev.time}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export function MarketContextPanel() {
  const { regime, eie, isLoading } = useDashboard()
  const { data: fullRegime } = useMarketRegime()

  // Prefer full RegimeData from dedicated endpoint for ICT signals
  const regimeData: RegimeData | RegimeSummary | null = fullRegime ?? regime

  if (isLoading) {
    return (
      <div className="h-full bg-surface overflow-y-auto">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="px-4 py-3 border-b border-border/60">
            <Skeleton rows={3} />
          </div>
        ))}
      </div>
    )
  }

  return (
    <aside className="h-full bg-surface overflow-y-auto flex flex-col">
      <div className="px-4 py-2 border-b border-border sticky top-0 bg-surface z-10">
        <p className="label">MARKET CONTEXT</p>
      </div>

      <div className="flex-1 overflow-y-auto">
        <RegimeSection regime={regimeData} />
        <Separator />
        <SessionSection />
        <Separator />
        <ICTSection regime={fullRegime ?? null} />
        {fullRegime && 'ict' in fullRegime && <Separator />}
        <EventsSection events={(eie as { upcoming?: EIEEvent[] })?.upcoming} />

        {/* Narrative snippet */}
        {regimeData?.narrative && (
          <>
            <Separator />
            <div className="px-4 py-3">
              <span className="label block mb-2">NARRATIVE</span>
              <p className="text-secondary text-xs leading-relaxed">{regimeData.narrative}</p>
            </div>
          </>
        )}
      </div>
    </aside>
  )
}
