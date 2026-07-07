'use client'
import { useDashboard } from '@/hooks/useDashboard'
import { Badge, RecommendationBadge } from '@/components/ui/Badge'
import { LivePulse } from '@/components/ui/LivePulse'
import { Skeleton } from '@/components/ui/Skeleton'
import { cn } from '@/utils/cn'
import { confidenceToPercent, toDisplayLabel } from '@/utils/format'
import { useWSStore } from '@/stores/wsStore'
import { useUIStore } from '@/stores/uiStore'
import type { RegimeData, RegimeSummary } from '@/types/api'

function ChevronIcon({ open }: { open: boolean }) {
  return (
    <svg viewBox="0 0 10 10" fill="none" className={cn('w-3 h-3 transition-transform', open && 'rotate-180')} stroke="currentColor" strokeWidth="1.5">
      <path d="M2 3.5l3 3 3-3"/>
    </svg>
  )
}

function regimeBadgeVariant(dominant: string): 'manipulation' | 'expansion' | 'consolidation' {
  const d = dominant?.toUpperCase()
  if (d?.includes('MANIPULATION')) return 'manipulation'
  if (d?.includes('EXPANSION'))    return 'expansion'
  return 'consolidation'
}

function biasBadgeVariant(bias: string | undefined): 'buy' | 'sell' | 'hold' {
  const b = bias?.toUpperCase()
  if (b === 'BULLISH') return 'buy'
  if (b === 'BEARISH') return 'sell'
  return 'hold'
}

function riskColor(risk: string | undefined): string {
  switch (risk?.toUpperCase()) {
    case 'LOW':      return 'text-green-400'
    case 'MEDIUM':   return 'text-yellow-400'
    case 'HIGH':     return 'text-orange-400'
    case 'CRITICAL': return 'text-red-400'
    default:         return 'text-secondary'
  }
}

export function AIMarketSummary() {
  const { decision, regime, mia, eie, isLoading } = useDashboard()
  const { connected } = useWSStore()
  const { summaryExpanded, toggleSummary } = useUIStore()

  // Prefer Groq market_summary; fall back to regime narrative if MIA hasn't generated yet
  const groqNarrative = mia?.market_summary
  const regimeNarrative = (regime as RegimeData | RegimeSummary | null)?.narrative
  const narrative = groqNarrative ?? regimeNarrative ?? 'Awaiting AI market analysis…'
  const isGroq = !!groqNarrative && !mia?.is_fallback

  const dominant = (regime as RegimeData | RegimeSummary | null)?.dominant
  const eieCount = eie?.active_count ?? 0

  return (
    <div className="bg-surface border-b border-border shrink-0">
      <div className="flex items-start gap-3 px-4 py-3">

        {/* Left: AI icon + narrative */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1.5">
            <div className="w-5 h-5 rounded bg-gold/10 border border-gold/30 flex items-center justify-center shrink-0">
              <span className="text-gold text-[10px] font-black">AI</span>
            </div>
            <span className="label text-secondary">MARKET NARRATIVE</span>
            {isGroq && <span className="text-[10px] text-gold/60 font-medium">GROQ</span>}
            <LivePulse connected={connected} size="sm" />
          </div>

          {isLoading ? (
            <Skeleton rows={2} />
          ) : (
            <p
              className={cn(
                'text-primary text-sm leading-relaxed transition-all duration-300',
                !summaryExpanded && 'line-clamp-2',
              )}
            >
              {narrative}
            </p>
          )}

          {/* Expanded metadata row */}
          {!isLoading && summaryExpanded && (
            <div className="flex items-center gap-3 mt-2 flex-wrap">
              {dominant && (
                <Badge variant={regimeBadgeVariant(dominant)} size="xs">
                  {dominant.charAt(0) + dominant.slice(1).toLowerCase()}
                </Badge>
              )}
              {mia?.market_bias && (
                <Badge variant={biasBadgeVariant(mia.market_bias)} size="xs">
                  {mia.market_bias}
                </Badge>
              )}
              {mia?.confidence != null && (
                <span className="text-secondary text-xs">
                  {Math.round(mia.confidence * 100)}% confidence
                </span>
              )}
              {mia?.risk_level && (
                <span className={cn('text-xs font-medium', riskColor(mia.risk_level))}>
                  {mia.risk_level} RISK
                </span>
              )}
              {eieCount > 0 && (
                <span className="text-wait text-xs">{eieCount} economic event{eieCount > 1 ? 's' : ''} active</span>
              )}
              {mia?.is_fallback && (
                <span className="text-muted text-xs italic">AI unavailable — technical analysis shown</span>
              )}
            </div>
          )}
        </div>

        {/* Right: decision summary */}
        <div className="flex items-center gap-3 shrink-0">
          {decision && (
            <>
              <div className="text-right hidden sm:block">
                <RecommendationBadge rec={decision.recommendation} size="md" />
                <p className="num text-secondary text-xs mt-1">
                  {confidenceToPercent(decision.confidence)}% confidence
                </p>
              </div>
              <div className="hidden sm:block w-px h-8 bg-border" />
            </>
          )}

          <button
            onClick={toggleSummary}
            className="text-muted hover:text-secondary transition-colors p-1"
            aria-label={summaryExpanded ? 'Collapse summary' : 'Expand summary'}
          >
            <ChevronIcon open={summaryExpanded} />
          </button>
        </div>
      </div>
    </div>
  )
}
