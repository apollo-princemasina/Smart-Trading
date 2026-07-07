'use client'
import { useDecisionHistory } from '@/hooks/useDecisions'
import { useWSStore } from '@/stores/wsStore'
import { cn } from '@/utils/cn'
import { confidenceToPercent, formatTime, formatRelative } from '@/utils/format'
import { Skeleton } from '@/components/ui/Skeleton'
import type { DecisionHistoryItem, Recommendation } from '@/types/api'

const REC_STYLE: Record<Recommendation, { border: string; text: string; dot: string }> = {
  BUY:  { border: 'border-buy-dim',  text: 'text-buy',  dot: 'bg-buy' },
  SELL: { border: 'border-sell-dim', text: 'text-sell', dot: 'bg-sell' },
  WAIT: { border: 'border-wait-dim', text: 'text-wait', dot: 'bg-wait' },
}

function DecisionCard({ item, isLatest }: { item: DecisionHistoryItem; isLatest?: boolean }) {
  const style   = REC_STYLE[item.recommendation] ?? REC_STYLE.WAIT
  const confPct = confidenceToPercent(item.confidence)
  const agrPct  = Math.round(item.agreement_score * 100)

  return (
    <div
      className={cn(
        'shrink-0 w-44 bg-surface border rounded-lg p-3 flex flex-col gap-2 transition-all',
        style.border,
        isLatest && 'ring-1 ring-gold/30',
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <div className={cn('w-1.5 h-1.5 rounded-full', style.dot)} />
          <span className={cn('text-xs font-black tracking-wider', style.text)}>
            {item.recommendation}
          </span>
        </div>
        <span className="text-muted text-[10px] num">{formatRelative(item.generated_at)}</span>
      </div>

      {/* Confidence bar */}
      <div>
        <div className="flex items-center justify-between mb-1">
          <span className="text-muted text-[10px]">Confidence</span>
          <span className={cn('num text-[10px] font-semibold', style.text)}>{confPct}%</span>
        </div>
        <div className="h-1 bg-navy-700 rounded-full overflow-hidden">
          <div
            className={cn(
              'h-full rounded-full transition-all',
              item.recommendation === 'BUY'  ? 'bg-buy' :
              item.recommendation === 'SELL' ? 'bg-sell' : 'bg-wait',
            )}
            style={{ width: `${confPct}%` }}
          />
        </div>
      </div>

      {/* Metadata */}
      <div className="flex items-center justify-between">
        <span className="text-secondary text-[10px]">{item.strength}</span>
        <span className="text-muted text-[10px] num">Agr {agrPct}%</span>
      </div>
    </div>
  )
}

export function DecisionTimeline() {
  const { latestDecision } = useWSStore()
  const { data, isLoading } = useDecisionHistory({ page_size: 20 })
  const decisions = data?.decisions ?? []

  return (
    <div className="shrink-0 border-t border-border bg-surface">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-border/60">
        <p className="label">DECISION TIMELINE</p>
        {data?.total != null && (
          <span className="text-muted text-[10px] num">{data.total} total</span>
        )}
      </div>

      {/* Scrollable row */}
      <div className="flex items-start gap-3 px-4 py-3 overflow-x-auto">
        {isLoading ? (
          Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="shrink-0 w-44 h-24 bg-navy-700 rounded-lg animate-pulse" />
          ))
        ) : decisions.length === 0 ? (
          <p className="text-muted text-xs py-4">No decision history yet</p>
        ) : (
          decisions.map((item, i) => (
            <DecisionCard key={item.id} item={item} isLatest={i === 0} />
          ))
        )}
      </div>
    </div>
  )
}
