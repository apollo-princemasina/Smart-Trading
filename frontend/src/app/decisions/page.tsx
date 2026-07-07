'use client'
import { useState } from 'react'
import { AppShell } from '@/components/layout/AppShell'
import { Card } from '@/components/ui/Card'
import { Badge, RecommendationBadge } from '@/components/ui/Badge'
import { Skeleton } from '@/components/ui/Skeleton'
import { useDecisionHistory } from '@/hooks/useDecisions'
import { confidenceToPercent, formatDateTime, scoreToPercent, consensusToLabel } from '@/utils/format'
import { cn } from '@/utils/cn'
import type { Recommendation, Strength, DecisionHistoryItem } from '@/types/api'

const REC_OPTIONS:      Array<{ label: string; value: string }> = [
  { label: 'All',  value: '' },
  { label: 'BUY',  value: 'BUY' },
  { label: 'SELL', value: 'SELL' },
  { label: 'WAIT', value: 'WAIT' },
]

const STR_OPTIONS: Array<{ label: string; value: string }> = [
  { label: 'All',      value: '' },
  { label: 'Strong',   value: 'STRONG' },
  { label: 'Moderate', value: 'MODERATE' },
  { label: 'Weak',     value: 'WEAK' },
]

function FilterButton({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'px-3 py-1 text-xs rounded border transition-all',
        active
          ? 'bg-gold/10 border-gold/50 text-gold font-semibold'
          : 'border-border text-secondary hover:text-primary hover:border-border/80',
      )}
    >
      {label}
    </button>
  )
}

function DecisionRow({ item }: { item: DecisionHistoryItem }) {
  const confPct = confidenceToPercent(item.confidence)
  const agrPct  = scoreToPercent(item.agreement_score)
  const conPct  = scoreToPercent(item.conflict_score)

  return (
    <tr className="border-b border-border/60 hover:bg-navy-700/30 transition-colors">
      <td className="py-3 px-4">
        <span className="num text-secondary text-xs">{formatDateTime(item.generated_at)}</span>
      </td>
      <td className="py-3 px-4">
        <RecommendationBadge rec={item.recommendation} size="sm" />
      </td>
      <td className="py-3 px-4">
        <Badge
          variant={item.strength === 'STRONG' ? 'buy' : item.strength === 'MODERATE' ? 'wait' : 'muted'}
          size="xs"
        >
          {item.strength}
        </Badge>
      </td>
      <td className="py-3 px-4">
        <div className="flex items-center gap-2">
          <div className="w-16 h-1 bg-navy-700 rounded-full overflow-hidden">
            <div
              className={cn(
                'h-full rounded-full',
                confPct >= 70 ? 'bg-buy' : confPct >= 40 ? 'bg-wait' : 'bg-sell',
              )}
              style={{ width: `${confPct}%` }}
            />
          </div>
          <span className="num text-primary text-xs font-semibold">{confPct}%</span>
        </div>
      </td>
      <td className="py-3 px-4 hidden md:table-cell">
        <span className="num text-buy text-xs">{agrPct}%</span>
      </td>
      <td className="py-3 px-4 hidden md:table-cell">
        <span className="num text-sell text-xs">{conPct}%</span>
      </td>
      <td className="py-3 px-4 hidden lg:table-cell">
        <span className="text-secondary text-xs">{consensusToLabel(item.consensus_level)}</span>
      </td>
      <td className="py-3 px-4 hidden lg:table-cell">
        <div className="flex items-center gap-1">
          {item.has_ml  && <Badge variant="blue" size="xs">ML</Badge>}
          {item.has_eie && <Badge variant="gold" size="xs">EIE</Badge>}
          {item.has_mia && <Badge variant="muted" size="xs">MIA</Badge>}
        </div>
      </td>
    </tr>
  )
}

export default function DecisionsPage() {
  const [rec, setRec]   = useState('')
  const [str, setStr]   = useState('')
  const [page, setPage] = useState(1)
  const PAGE_SIZE = 25

  const { data, isLoading, isError } = useDecisionHistory({
    recommendation: rec || undefined,
    strength:       str || undefined,
    page,
    page_size: PAGE_SIZE,
  })

  const total     = data?.total ?? 0
  const totalPages = Math.ceil(total / PAGE_SIZE)

  return (
    <AppShell>
      <div className="h-full flex flex-col overflow-hidden">

        {/* Page header */}
        <div className="px-6 py-4 border-b border-border shrink-0">
          <h1 className="text-primary font-bold text-lg">Decision History</h1>
          <p className="text-secondary text-sm mt-0.5">All AI-generated trading decisions with confidence scores</p>
        </div>

        <div className="flex-1 overflow-y-auto p-6">

          {/* Filters */}
          <div className="flex flex-wrap gap-4 mb-6">
            <div>
              <p className="label mb-2">RECOMMENDATION</p>
              <div className="flex gap-2">
                {REC_OPTIONS.map(o => (
                  <FilterButton key={o.value} label={o.label} active={rec === o.value} onClick={() => { setRec(o.value); setPage(1) }} />
                ))}
              </div>
            </div>
            <div>
              <p className="label mb-2">STRENGTH</p>
              <div className="flex gap-2">
                {STR_OPTIONS.map(o => (
                  <FilterButton key={o.value} label={o.label} active={str === o.value} onClick={() => { setStr(o.value); setPage(1) }} />
                ))}
              </div>
            </div>
          </div>

          {/* Table */}
          <Card noPadding>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border">
                    {['Time', 'Decision', 'Strength', 'Confidence', 'Agreement', 'Conflict', 'Consensus', 'Sources'].map((h, i) => (
                      <th
                        key={h}
                        className={cn(
                          'text-left py-3 px-4 label',
                          i >= 4 && 'hidden md:table-cell',
                          i >= 6 && 'hidden lg:table-cell',
                        )}
                      >
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {isLoading ? (
                    Array.from({ length: 10 }).map((_, i) => (
                      <tr key={i} className="border-b border-border/60">
                        {Array.from({ length: 8 }).map((_, j) => (
                          <td key={j} className="py-3 px-4"><Skeleton className="h-3 w-16" /></td>
                        ))}
                      </tr>
                    ))
                  ) : isError ? (
                    <tr><td colSpan={8} className="text-center py-12 text-secondary text-sm">Failed to load decisions</td></tr>
                  ) : !data?.decisions.length ? (
                    <tr><td colSpan={8} className="text-center py-12 text-muted text-sm">No decisions found</td></tr>
                  ) : (
                    data.decisions.map(item => <DecisionRow key={item.id} item={item} />)
                  )}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between mt-4">
              <span className="text-secondary text-xs">
                {total} decisions · Page {page} of {totalPages}
              </span>
              <div className="flex gap-2">
                <button
                  disabled={page <= 1}
                  onClick={() => setPage(p => p - 1)}
                  className="px-3 py-1 text-xs border border-border rounded text-secondary hover:text-primary disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  Previous
                </button>
                <button
                  disabled={page >= totalPages}
                  onClick={() => setPage(p => p + 1)}
                  className="px-3 py-1 text-xs border border-border rounded text-secondary hover:text-primary disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  Next
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </AppShell>
  )
}
