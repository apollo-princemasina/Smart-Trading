'use client'
import { useState } from 'react'
import { AppShell } from '@/components/layout/AppShell'
import { Skeleton } from '@/components/ui/Skeleton'
import { usePredictionHistory } from '@/hooks/useDecisions'
import { cn } from '@/utils/cn'

const DIR_OPTIONS = [
  { label: 'All',  value: '' },
  { label: 'BUY',  value: 'BUY' },
  { label: 'SELL', value: 'SELL' },
  { label: 'HOLD', value: 'HOLD' },
]

const DIR_STYLE = {
  BUY:  { badge: 'text-emerald-400 bg-emerald-400/10 border-emerald-400/25', dot: 'bg-emerald-400' },
  SELL: { badge: 'text-red-400 bg-red-400/10 border-red-400/25',             dot: 'bg-red-400'     },
  HOLD: { badge: 'text-slate-400 bg-white/5 border-white/10',                dot: 'bg-slate-500'   },
} as const

// ── Stacked probability bar: one bar, three segments ─────────────────────────
function ProbStack({ buy, sell, hold }: { buy: number; sell: number; hold: number }) {
  const b = Math.round((buy  ?? 0) * 100)
  const s = Math.round((sell ?? 0) * 100)
  const h = Math.round((hold ?? 0) * 100)
  return (
    <div className="flex items-center gap-2 w-[140px]">
      {/* Stacked bar */}
      <div className="flex h-1.5 flex-1 rounded-full overflow-hidden">
        <div className="bg-emerald-500 h-full transition-all" style={{ width: `${b}%` }} />
        <div className="bg-red-500 h-full transition-all"     style={{ width: `${s}%` }} />
        <div className="bg-slate-600 h-full transition-all"   style={{ width: `${h}%` }} />
      </div>
      {/* Labels */}
      <div className="flex gap-2 shrink-0">
        <span className="num text-[10px] text-emerald-400 font-semibold w-[26px] text-right">{b}%</span>
        <span className="num text-[10px] text-red-400     font-semibold w-[26px] text-right">{s}%</span>
        <span className="num text-[10px] text-slate-400   font-semibold w-[26px] text-right">{h}%</span>
      </div>
    </div>
  )
}

// ── Column header labels aligned to data ─────────────────────────────────────
function ProbHeader() {
  return (
    <div className="flex items-center gap-2 w-[140px]">
      <div className="flex-1" />
      <div className="flex gap-2 shrink-0">
        <span className="text-[9px] text-muted uppercase w-[26px] text-right tracking-wide">B</span>
        <span className="text-[9px] text-muted uppercase w-[26px] text-right tracking-wide">S</span>
        <span className="text-[9px] text-muted uppercase w-[26px] text-right tracking-wide">H</span>
      </div>
    </div>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────────
export default function PredictionsPage() {
  const [dir,  setDir]  = useState('')
  const [page, setPage] = useState(1)
  const PAGE_SIZE = 50

  const { data, isLoading, isError } = usePredictionHistory({
    direction: dir || undefined,
    page,
    page_size: PAGE_SIZE,
  })

  const total      = data?.total ?? 0
  const totalPages = Math.ceil(total / PAGE_SIZE)

  return (
    <AppShell>
      <div className="h-full flex flex-col overflow-hidden">

        {/* Header */}
        <div className="px-6 py-4 border-b border-border shrink-0 flex items-center justify-between">
          <div>
            <h1 className="text-primary font-bold text-lg tracking-tight">Model Intelligence Log</h1>
            <p className="text-secondary text-sm mt-0.5">Every bar — direction, probabilities, session weighting, regime</p>
          </div>
          <div className="flex items-center gap-1">
            {DIR_OPTIONS.map(o => (
              <button
                key={o.value}
                onClick={() => { setDir(o.value); setPage(1) }}
                className={cn(
                  'px-3 py-1 text-xs rounded border transition-all',
                  dir === o.value
                    ? 'bg-gold/10 border-gold/50 text-gold font-semibold'
                    : 'border-border text-secondary hover:text-primary hover:border-border-hover',
                )}
              >
                {o.label}
              </button>
            ))}
          </div>
        </div>

        {/* Table */}
        <div className="flex-1 overflow-auto">
          <table className="w-full border-collapse" style={{ tableLayout: 'fixed' }}>

            {/* Fixed column widths */}
            <colgroup>
              <col style={{ width: 80 }}  />  {/* TIME */}
              <col style={{ width: 72 }}  />  {/* SIGNAL */}
              <col style={{ width: 60 }}  />  {/* CONF */}
              <col style={{ width: 176 }} />  {/* PROBABILITY */}
              <col style={{ width: 110 }} />  {/* SESSION */}
              <col style={{ width: 110 }} />  {/* REGIME */}
              <col style={{ width: 88 }}  />  {/* ENTRY */}
              <col style={{ width: 100 }} />  {/* STOP LOSS */}
              <col style={{ width: 100 }} />  {/* TAKE PROFIT */}
              <col style={{ width: 52 }}  />  {/* R:R */}
            </colgroup>

            <thead className="sticky top-0 z-10 bg-surface border-b border-border">
              <tr>
                <th className="py-2.5 pl-4 pr-2 text-left">
                  <span className="label text-[10px]">TIME</span>
                </th>
                <th className="py-2.5 px-2 text-left">
                  <span className="label text-[10px]">SIGNAL</span>
                </th>
                <th className="py-2.5 px-2 text-left">
                  <span className="label text-[10px]">CONF</span>
                </th>
                <th className="py-2.5 px-2 text-left">
                  <div className="flex flex-col gap-0.5">
                    <span className="label text-[10px]">PROBABILITY</span>
                    <ProbHeader />
                  </div>
                </th>
                <th className="py-2.5 px-2 text-left">
                  <span className="label text-[10px]">SESSION</span>
                </th>
                <th className="py-2.5 px-2 text-left">
                  <span className="label text-[10px]">REGIME</span>
                </th>
                <th className="py-2.5 px-2 text-right">
                  <span className="label text-[10px]">ENTRY</span>
                </th>
                <th className="py-2.5 px-2 text-right">
                  <span className="label text-[10px] text-red-400/70">STOP LOSS</span>
                </th>
                <th className="py-2.5 px-2 text-right">
                  <span className="label text-[10px] text-emerald-400/70">TAKE PROFIT</span>
                </th>
                <th className="py-2.5 pr-4 pl-2 text-right">
                  <span className="label text-[10px]">R:R</span>
                </th>
              </tr>
            </thead>

            <tbody>
              {isLoading ? (
                Array.from({ length: 14 }).map((_, i) => (
                  <tr key={i} className="border-b border-border/30">
                    {Array.from({ length: 9 }).map((_, j) => (
                      <td key={j} className="py-3 px-2">
                        <Skeleton className="h-3" style={{ width: j === 3 ? '100%' : '60%' }} />
                      </td>
                    ))}
                  </tr>
                ))
              ) : isError ? (
                <tr>
                  <td colSpan={9} className="text-center py-20 text-secondary text-sm">
                    Failed to load predictions
                  </td>
                </tr>
              ) : !data?.predictions?.length ? (
                <tr>
                  <td colSpan={9} className="text-center py-20 text-muted text-sm">
                    No predictions found
                  </td>
                </tr>
              ) : (
                data.predictions.map((p, i) => {
                  const dir      = (p.direction ?? 'HOLD') as keyof typeof DIR_STYLE
                  const style    = DIR_STYLE[dir] ?? DIR_STYLE.HOLD
                  const confPct  = Math.round((p.confidence ?? 0) * 100)
                  const rawPct   = p.raw_confidence != null ? Math.round(p.raw_confidence * 100) : null
                  const confColor =
                    confPct >= 70 ? 'text-emerald-400' :
                    confPct >= 55 ? 'text-amber-400'   : 'text-secondary'
                  const session  = p.session?.replace(/_/g, ' ') ?? '—'
                  const mult     = p.session_mult
                  const regime   = p.regime ?? '—'
                  const regColor =
                    regime === 'EXPANSION'    ? 'text-blue-400'  :
                    regime === 'MANIPULATION' ? 'text-amber-400' : 'text-secondary'

                  return (
                    <tr
                      key={p.id}
                      className={cn(
                        'border-b border-border/25 align-middle transition-colors',
                        i % 2 !== 0 ? 'bg-white/[0.012]' : '',
                        dir === 'BUY'  ? 'hover:bg-emerald-900/10' :
                        dir === 'SELL' ? 'hover:bg-red-900/10'     :
                                         'hover:bg-white/[0.025]',
                      )}
                    >
                      {/* TIME */}
                      <td className="py-3 pl-4 pr-2 whitespace-nowrap align-middle">
                        <span className="num text-primary text-xs block">
                          {new Date(p.signal_time).toLocaleTimeString('en-GB', {
                            hour: '2-digit', minute: '2-digit', timeZone: 'UTC',
                          })}
                        </span>
                        <span className="num text-muted text-[10px] block leading-tight">
                          {new Date(p.signal_time).toLocaleDateString('en-GB', {
                            day: '2-digit', month: 'short', timeZone: 'UTC',
                          })}
                        </span>
                      </td>

                      {/* SIGNAL */}
                      <td className="py-3 px-2 align-middle">
                        <span className={cn(
                          'inline-flex items-center gap-1.5 px-2 py-0.5 rounded border text-[10px] font-bold tracking-widest',
                          style.badge,
                        )}>
                          <span className={cn('w-1.5 h-1.5 rounded-full shrink-0', style.dot)} />
                          {dir}
                        </span>
                        {p.demoted && p.raw_direction && p.raw_direction !== p.direction && (
                          <span className="block text-[9px] text-amber-500/80 font-mono mt-0.5 pl-0.5">
                            ↓ {p.raw_direction}
                          </span>
                        )}
                      </td>

                      {/* CONF */}
                      <td className="py-3 px-2 align-middle">
                        <span className={cn('num text-sm font-bold tabular-nums', confColor)}>
                          {confPct}%
                        </span>
                        {rawPct != null && rawPct !== confPct && (
                          <span className="num block text-[10px] text-muted leading-tight">
                            {rawPct}%
                          </span>
                        )}
                      </td>

                      {/* PROBABILITY */}
                      <td className="py-3 px-2 align-middle">
                        <ProbStack
                          buy={p.prob_buy ?? 0}
                          sell={p.prob_sell ?? 0}
                          hold={p.prob_hold ?? 0}
                        />
                      </td>

                      {/* SESSION */}
                      <td className="py-3 px-2 align-middle">
                        <span className="text-xs text-primary block truncate">{session}</span>
                        {mult != null && (
                          <span className="num text-[10px] text-muted block leading-tight">×{mult.toFixed(2)}</span>
                        )}
                      </td>

                      {/* REGIME */}
                      <td className="py-3 px-2 align-middle">
                        <span className={cn('text-xs font-semibold truncate block', regColor)}>
                          {regime}
                        </span>
                      </td>

                      {/* ENTRY */}
                      <td className="py-3 px-2 align-middle text-right">
                        <span className="num text-primary text-xs font-semibold">
                          {p.close?.toFixed(5) ?? '—'}
                        </span>
                      </td>

                      {/* STOP LOSS — only meaningful for directional signals */}
                      <td className="py-3 px-2 align-middle text-right">
                        {p.direction !== 'HOLD' && p.sl_price != null ? (
                          <>
                            <span className="num text-red-400 text-xs font-semibold block">
                              {p.sl_price.toFixed(5)}
                            </span>
                            <span className="num text-red-400/50 text-[10px] block leading-tight">
                              {p.sl_pips != null ? `−${p.sl_pips}p` : ''}
                            </span>
                          </>
                        ) : (
                          <span className="text-border text-xs">—</span>
                        )}
                      </td>

                      {/* TAKE PROFIT */}
                      <td className="py-3 px-2 align-middle text-right">
                        {p.direction !== 'HOLD' && p.tp_price != null ? (
                          <>
                            <span className="num text-emerald-400 text-xs font-semibold block">
                              {p.tp_price.toFixed(5)}
                            </span>
                            <span className="num text-emerald-400/50 text-[10px] block leading-tight">
                              {p.tp_pips != null ? `+${p.tp_pips}p` : ''}
                            </span>
                          </>
                        ) : (
                          <span className="text-border text-xs">—</span>
                        )}
                      </td>

                      {/* R:R */}
                      <td className="py-3 pr-4 pl-2 align-middle text-right">
                        {p.direction !== 'HOLD' && p.tp_pips != null && p.sl_pips != null ? (
                          <span className="num text-secondary text-xs font-semibold">
                            {(p.tp_pips / p.sl_pips).toFixed(1)}:1
                          </span>
                        ) : (
                          <span className="text-border text-xs">—</span>
                        )}
                      </td>
                    </tr>
                  )
                })
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="shrink-0 flex items-center justify-between px-6 py-3 border-t border-border">
            <span className="text-secondary text-xs">{total} bars · page {page} of {totalPages}</span>
            <div className="flex gap-2">
              <button
                disabled={page <= 1}
                onClick={() => setPage(p => p - 1)}
                className="px-3 py-1 text-xs border border-border rounded text-secondary hover:text-primary disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
              >
                Previous
              </button>
              <button
                disabled={page >= totalPages}
                onClick={() => setPage(p => p + 1)}
                className="px-3 py-1 text-xs border border-border rounded text-secondary hover:text-primary disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>
    </AppShell>
  )
}
