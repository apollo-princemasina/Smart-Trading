'use client'
import { useDashboard, useMarketRegime } from '@/hooks/useDashboard'
import { useWSStore } from '@/stores/wsStore'
import { LivePulse } from '@/components/ui/LivePulse'
import { Skeleton } from '@/components/ui/Skeleton'
import { Separator } from '@/components/ui/Separator'
import { Badge } from '@/components/ui/Badge'
import { cn } from '@/utils/cn'
import { formatPrice, formatRelative, scoreToPercent } from '@/utils/format'
import type { Signal, RegimeData, RegimeSummary, ConvictionData, ConvictionLevel } from '@/types/api'

// ── Direction config ──────────────────────────────────────────────────────────

const DIR_CFG = {
  BUY:  { label: '▲ BUY',  color: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/30' },
  SELL: { label: '▼ SELL', color: 'text-red-400',     bg: 'bg-red-500/10',     border: 'border-red-500/30'     },
  HOLD: { label: '— HOLD', color: 'text-secondary',   bg: 'bg-navy-700',       border: 'border-border'         },
} as const

// ── Sub-components ────────────────────────────────────────────────────────────

function ProbBar({ label, pct, barClass }: { label: string; pct: number; barClass: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-muted text-[10px] font-mono w-8 shrink-0">{label}</span>
      <div className="flex-1 h-1.5 bg-navy-800 rounded-full overflow-hidden">
        <div className={cn('h-full rounded-full transition-all duration-700', barClass)} style={{ width: `${pct}%` }} />
      </div>
      <span className="num text-[10px] font-bold w-9 text-right text-primary">{pct}%</span>
    </div>
  )
}

function Stat({ label, value, valueClass }: { label: string; value: React.ReactNode; valueClass?: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-muted text-[10px] uppercase tracking-wide">{label}</span>
      <span className={cn('num text-xs font-semibold text-primary truncate', valueClass)}>{value ?? '—'}</span>
    </div>
  )
}

const CONVICTION_CFG: Record<ConvictionLevel, { label: string; color: string; bg: string; border: string }> = {
  HIGH_CONVICTION:  { label: '◆ HIGH CONVICTION',  color: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/40' },
  SETUP_FORMING:    { label: '◈ SETUP FORMING',     color: 'text-amber-400',   bg: 'bg-amber-500/10',   border: 'border-amber-500/40'   },
  DIRECTIONAL_BIAS: { label: '◇ DIRECTIONAL BIAS',  color: 'text-blue-400',    bg: 'bg-blue-500/10',    border: 'border-blue-500/30'    },
  CONFLICTED:       { label: '✕ CONFLICTED',         color: 'text-red-400',     bg: 'bg-red-500/10',     border: 'border-red-500/30'     },
  NEUTRAL:          { label: '— NEUTRAL',            color: 'text-muted',       bg: 'bg-navy-700',       border: 'border-border'         },
}

function ConvictionBlock({ c }: { c: ConvictionData }) {
  const cfg = CONVICTION_CFG[c.level] ?? CONVICTION_CFG.NEUTRAL
  const dir4Cfg = DIR_CFG[c.direction_4b] ?? DIR_CFG.HOLD
  const dir8Cfg = DIR_CFG[c.direction_8b] ?? DIR_CFG.HOLD
  return (
    <div className={cn('rounded border px-2.5 py-2 shrink-0', cfg.bg, cfg.border)}>
      <div className="flex items-center justify-between mb-1.5">
        <span className={cn('text-[11px] font-bold tracking-wide', cfg.color)}>{cfg.label}</span>
        <span className="text-muted text-[9px]">MULTI-MODEL</span>
      </div>
      <div className="flex gap-3">
        <div className="flex flex-col gap-0.5">
          <span className="text-muted text-[9px] uppercase tracking-wide">1h (4b)</span>
          <span className={cn('text-[11px] font-bold', dir4Cfg.color)}>{c.direction_4b}</span>
          <span className="num text-[9px] text-muted">{Math.round(Math.max(c.prob_buy_4b, c.prob_sell_4b, c.prob_hold_4b) * 100)}%</span>
        </div>
        <div className="flex flex-col gap-0.5">
          <span className="text-muted text-[9px] uppercase tracking-wide">2h (8b)</span>
          <span className={cn('text-[11px] font-bold', dir8Cfg.color)}>{c.direction_8b}</span>
          <span className="num text-[9px] text-muted">{Math.round(Math.max(c.prob_buy_8b, c.prob_sell_8b, c.prob_hold_8b) * 100)}%</span>
        </div>
        <p className="text-secondary text-[10px] leading-snug flex-1 line-clamp-2">{c.description}</p>
      </div>
    </div>
  )
}

// ── Main panel ────────────────────────────────────────────────────────────────

export function ChartDecisionPanel() {
  const { signal, regime, decision, isLoading } = useDashboard()
  const { latestSignal, connected } = useWSStore()
  const { data: fullRegime } = useMarketRegime()

  const sig    = (latestSignal ?? (signal as unknown as Signal | null))
  const reg    = fullRegime ?? (regime as RegimeData | RegimeSummary | null)
  const regAny = reg as any
  const sigAny = sig as any

  if (isLoading && !sig) {
    return (
      <div className="h-full flex flex-col bg-bg p-3 gap-3">
        <Skeleton variant="block" className="h-16" />
        <Skeleton variant="block" className="h-10" />
        <Skeleton rows={4} />
      </div>
    )
  }

  const dir     = (sig?.direction ?? 'HOLD') as keyof typeof DIR_CFG
  const cfg     = DIR_CFG[dir] ?? DIR_CFG.HOLD
  const confPct = sig?.confidence != null
    ? Math.round(sig.confidence <= 1 ? sig.confidence * 100 : sig.confidence)
    : null
  const rawPct = sigAny?.raw_confidence != null
    ? Math.round(sigAny.raw_confidence <= 1 ? sigAny.raw_confidence * 100 : sigAny.raw_confidence)
    : null
  const demoted  = sigAny?.demoted ?? false
  const rawDir   = sigAny?.raw_direction ?? dir
  const probBuy  = Math.round((sigAny?.prob_buy  ?? 0) * 100)
  const probSell = Math.round((sigAny?.prob_sell ?? 0) * 100)
  const probHold = Math.round((sigAny?.prob_hold ?? 0) * 100)

  const dominant = regAny?.dominant ?? (typeof sigAny?.regime === 'string' ? sigAny.regime : null)
  const adx      = regAny?.adx
  const pdZone   = regAny?.pd_zone

  // ICT active flags
  const ict    = regAny?.ict
  const ictActive = ict ? [
    ['Sweep',   ict.liquidity_sweep, ict.sweep_direction],
    ['ChoCH',   ict.choch_detected,  ict.choch_direction],
    ['BoS',     ict.bos_detected,    ict.bos_direction],
    ['FVG',     ict.fvg_active,      ict.fvg_direction],
    ['OB',      ict.ob_active,       ict.ob_direction],
  ].filter(([, v]) => v) : []

  const decConf = decision
    ? Math.round(decision.confidence <= 1 ? decision.confidence * 100 : decision.confidence)
    : null

  const conviction             = (sigAny?.conviction ?? null) as ConvictionData | null
  const setupFormingAlert      = (sigAny?.setup_forming_alert ?? null) as string | null
  const convictionGateApplied  = sigAny?.conviction_gate_applied ?? false

  // ICT State Machine
  const ictObEntry    = sigAny?.ict_ob_entry   ?? false
  const ictSmState    = sigAny?.ict_sm_state   ?? 'IDLE'
  const ictSmDir      = sigAny?.ict_sm_direction ?? null
  const obBullTop     = sigAny?.ob_bullish_top    ?? null
  const obBullBottom  = sigAny?.ob_bullish_bottom ?? null
  const obBearTop     = sigAny?.ob_bearish_top    ?? null
  const obBearBottom  = sigAny?.ob_bearish_bottom ?? null
  const activeObTop    = ictSmDir === 'BUY'  ? obBullTop    : ictSmDir === 'SELL' ? obBearTop    : null
  const activeObBottom = ictSmDir === 'BUY'  ? obBullBottom : ictSmDir === 'SELL' ? obBearBottom : null

  return (
    <div className="h-full flex flex-col bg-bg overflow-hidden">

      {/* Header */}
      <div className="px-3 py-2 border-b border-border bg-surface shrink-0 flex items-center justify-between">
        <p className="label">ML SIGNAL INTELLIGENCE</p>
        <div className="flex items-center gap-2">
          {sig?.signal_time && (() => {
            const utcIso = sig.signal_time.includes('Z') || sig.signal_time.includes('+')
              ? sig.signal_time : sig.signal_time + 'Z'
            const openMs  = new Date(utcIso).getTime()
            const closeMs = openMs + 15 * 60 * 1000   // bar close = open + 15 min
            const openHHMM  = new Date(openMs ).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', timeZone: 'UTC' })
            const closeHHMM = new Date(closeMs).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', timeZone: 'UTC' })
            const agoSec  = Math.floor((Date.now() - closeMs) / 1000)
            const agoStr  = agoSec < 60 ? `${agoSec}s ago`
                          : agoSec < 3600 ? `${Math.floor(agoSec / 60)}m ago`
                          : `${Math.floor(agoSec / 3600)}h ago`
            return (
              <span className="num text-muted text-[10px]">
                bar {openHHMM}–{closeHHMM} UTC · {agoStr}
              </span>
            )
          })()}
          <LivePulse connected={connected} size="sm" />
        </div>
      </div>

      {/* Content — scrollable so conviction + decision fusion never get clipped */}
      <div className="flex-1 flex flex-col px-3 py-2 gap-2 min-h-0 overflow-y-auto">

        {/* Direction + probabilities side-by-side */}
        <div className="flex gap-3 shrink-0">

          {/* Direction hero */}
          <div className={cn('flex-1 rounded border px-3 py-2 flex flex-col justify-center', cfg.bg, cfg.border)}>
            {sig ? (
              <>
                <span className={cn('text-xl font-black tracking-widest', cfg.color)}>{cfg.label}</span>
                <div className="flex items-baseline gap-2 mt-0.5">
                  <span className={cn('num text-2xl font-black', cfg.color)}>{confPct}%</span>
                  {rawPct != null && rawPct !== confPct && (
                    <span className="num text-[10px] text-muted">raw {rawPct}%</span>
                  )}
                </div>
                <span className="text-[10px] text-muted mt-0.5">
                  {ictObEntry
                    ? 'ICT Order Block entry — precise OB-edge SL'
                    : convictionGateApplied
                      ? 'HOLD — Strategy B gate (no multi-model agreement)'
                      : demoted && rawDir !== dir
                        ? `${rawDir}→HOLD (session demoted)`
                        : 'EURUSD M15'}
                </span>
              </>
            ) : (
              <span className="text-secondary text-xs">Awaiting signal...</span>
            )}
          </div>

          {/* Probability bars */}
          {sig && (probBuy > 0 || probSell > 0 || probHold > 0) && (
            <div className="flex-1 flex flex-col justify-center gap-1.5">
              <ProbBar label="BUY"  pct={probBuy}  barClass="bg-emerald-500" />
              <ProbBar label="SELL" pct={probSell} barClass="bg-red-500" />
              <ProbBar label="HOLD" pct={probHold} barClass="bg-slate-500" />
            </div>
          )}
        </div>

        {/* ICT OB Entry confirmed */}
        {ictObEntry && (
          <div className="rounded border border-cyan-500/40 bg-cyan-500/10 px-2.5 py-1.5 shrink-0 flex items-center gap-2">
            <span className="text-cyan-400 font-bold text-[11px] tracking-wide shrink-0">◆ ICT OB ENTRY</span>
            <span className="text-cyan-300 text-[11px]">
              Price rejected from Order Block — SL pinned to OB edge, not ATR
            </span>
          </div>
        )}

        {/* ICT SM ARMED — setup watching for OB retracement */}
        {!ictObEntry && ictSmState === 'ARMED' && ictSmDir && (
          <div className="rounded border border-amber-500/40 bg-amber-500/10 px-2.5 py-1.5 shrink-0">
            <div className="flex items-center gap-2 mb-0.5">
              <span className="text-amber-400 font-bold text-[11px] tracking-wide shrink-0">◉ ICT ARMED</span>
              <span className="text-amber-300 text-[11px]">
                {ictSmDir} OB retracement watch active
              </span>
            </div>
            {activeObTop != null && activeObBottom != null && (
              <div className="flex gap-3 mt-0.5">
                <span className="text-muted text-[10px]">OB Top <span className="num text-amber-300 font-semibold">{Number(activeObTop).toFixed(5)}</span></span>
                <span className="text-muted text-[10px]">OB Bot <span className="num text-amber-300 font-semibold">{Number(activeObBottom).toFixed(5)}</span></span>
                <span className="text-muted text-[10px] ml-auto">Enter when price touches OB</span>
              </div>
            )}
          </div>
        )}

        {/* ICT SM OB_TESTED — price is at the OB zone */}
        {!ictObEntry && ictSmState === 'OB_TESTED' && ictSmDir && (
          <div className="rounded border border-orange-500/50 bg-orange-500/10 px-2.5 py-1.5 shrink-0">
            <div className="flex items-center gap-2 mb-0.5">
              <span className="text-orange-400 font-bold text-[11px] tracking-wide shrink-0">⊙ ICT OB TESTED</span>
              <span className="text-orange-300 text-[11px]">
                Price inside {ictSmDir} Order Block — waiting for close confirmation
              </span>
            </div>
            {activeObTop != null && activeObBottom != null && (
              <div className="flex gap-3 mt-0.5">
                <span className="text-muted text-[10px]">OB <span className="num text-orange-300 font-semibold">{Number(activeObBottom).toFixed(5)} – {Number(activeObTop).toFixed(5)}</span></span>
              </div>
            )}
          </div>
        )}

        {/* Strategy B gate — SETUP FORMING alert */}
        {setupFormingAlert && convictionGateApplied && (
          <div className="rounded border border-amber-500/40 bg-amber-500/10 px-2.5 py-1.5 shrink-0 flex items-center gap-2">
            <span className="text-amber-400 font-bold text-[11px] tracking-wide shrink-0">◈ SETUP FORMING</span>
            <span className="text-amber-300 text-[11px]">
              Structural <span className="font-bold">{setupFormingAlert}</span> building — awaiting 15-min confirmation before Strategy B fires
            </span>
          </div>
        )}

        <Separator />

        {/* Trade levels — only when a directional signal is active */}
        {sig && dir !== 'HOLD' && sigAny?.tp_price != null && sigAny?.sl_price != null ? (
          <div className="shrink-0 space-y-1.5">
            {/* Entry / SL / TP price levels */}
            <div className="grid grid-cols-3 gap-2">
              {/* Entry */}
              <div className="rounded border border-border bg-surface px-2.5 py-2">
                <span className="text-muted text-[9px] uppercase tracking-wider block mb-0.5">Entry</span>
                <span className="num text-primary text-xs font-bold">{Number(sigAny.close).toFixed(5)}</span>
              </div>
              {/* Stop Loss */}
              <div className="rounded border border-red-500/30 bg-red-500/8 px-2.5 py-2">
                <span className="text-red-400/70 text-[9px] uppercase tracking-wider block mb-0.5">Stop Loss</span>
                <span className="num text-red-400 text-xs font-bold">{Number(sigAny.sl_price).toFixed(5)}</span>
                <span className="num text-red-400/60 text-[10px] block leading-tight">
                  {sigAny.sl_pips != null ? `−${Number(sigAny.sl_pips).toFixed(1)} pips` : ''}
                </span>
              </div>
              {/* Take Profit */}
              <div className="rounded border border-emerald-500/30 bg-emerald-500/8 px-2.5 py-2">
                <span className="text-emerald-400/70 text-[9px] uppercase tracking-wider block mb-0.5">Take Profit</span>
                <span className="num text-emerald-400 text-xs font-bold">{Number(sigAny.tp_price).toFixed(5)}</span>
                <span className="num text-emerald-400/60 text-[10px] block leading-tight">
                  {sigAny.tp_pips != null ? `+${Number(sigAny.tp_pips).toFixed(1)} pips` : ''}
                </span>
              </div>
            </div>
            {/* R:R + context row */}
            <div className="flex items-center gap-4 px-0.5">
              <div className="flex items-center gap-1.5">
                <span className="text-muted text-[10px]">R:R</span>
                <span className="num text-primary text-[11px] font-bold">
                  {sigAny.tp_pips != null && sigAny.sl_pips != null
                    ? `${(sigAny.tp_pips / sigAny.sl_pips).toFixed(1)}:1`
                    : '—'}
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="text-muted text-[10px]">ATR</span>
                <span className="num text-secondary text-[11px]">
                  {sigAny.atr_pips != null ? `${Number(sigAny.atr_pips).toFixed(1)}p` : '—'}
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="text-muted text-[10px]">Session</span>
                <span className="text-secondary text-[11px]">
                  {sigAny?.session ?? '—'}{sigAny?.session_mult != null ? ` ×${Number(sigAny.session_mult).toFixed(2)}` : ''}
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="text-muted text-[10px]">Regime</span>
                <span className="text-secondary text-[11px]">{dominant ?? '—'}</span>
              </div>
            </div>
          </div>
        ) : sig ? (
          /* HOLD state — compact inline context strip */
          <div className="shrink-0 space-y-1">
            <div className="flex flex-wrap gap-x-5 gap-y-0.5">
              <span className="text-muted text-[10px]">Price <span className="num text-primary font-semibold">{formatPrice(sigAny?.close ?? null)}</span></span>
              <span className="text-muted text-[10px]">ATR <span className="num text-primary font-semibold">{sigAny?.atr_pips != null ? `${Number(sigAny.atr_pips).toFixed(1)}p` : '—'}</span></span>
              <span className="text-muted text-[10px]">Session <span className="text-primary font-semibold">{sigAny?.session?.replace(/_/g,' ') ?? '—'}{sigAny?.session_mult != null ? ` ×${Number(sigAny.session_mult).toFixed(2)}` : ''}</span></span>
            </div>
            <div className="flex flex-wrap gap-x-5 gap-y-0.5">
              <span className="text-muted text-[10px]">Regime <span className="text-primary font-semibold">{dominant ?? '—'}</span></span>
              {adx    != null && <span className="text-muted text-[10px]">ADX <span className="num text-primary font-semibold">{Number(adx).toFixed(1)}</span></span>}
              {pdZone && <span className="text-muted text-[10px]">PD Zone <span className="text-primary font-semibold">{pdZone}</span></span>}
            </div>
          </div>
        ) : null}

        {/* Multi-model conviction */}
        {conviction && (
          <>
            <Separator />
            <ConvictionBlock c={conviction} />
          </>
        )}

        {/* Decision fusion + ICT — compact single row */}
        {(decision || ictActive.length > 0) && (
          <div className="shrink-0 rounded border border-border bg-surface/50 px-2.5 py-2 space-y-1.5">
            {decision && (
              <>
                <div className="flex items-center justify-between">
                  <span className="label text-[10px]">DECISION FUSION</span>
                  <div className="flex items-center gap-1">
                    {decision.has_ml  && <Badge variant="blue"  size="xs">ML</Badge>}
                    {decision.has_eie && <Badge variant="gold"  size="xs">EIE</Badge>}
                    {decision.has_mia && <Badge variant="muted" size="xs">MIA</Badge>}
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  <div className="flex flex-col gap-0.5">
                    <span className="text-muted text-[9px] uppercase tracking-wide">Overall</span>
                    <span className={cn('text-sm font-black tracking-wider',
                      decision.recommendation === 'BUY'  ? 'text-emerald-400' :
                      decision.recommendation === 'SELL' ? 'text-red-400'     : 'text-wait'
                    )}>{decision.recommendation}</span>
                  </div>
                  <div className="flex flex-col gap-0.5">
                    <span className="text-muted text-[9px] uppercase tracking-wide">Strength</span>
                    <span className="text-primary text-xs font-semibold">{decision.strength}</span>
                  </div>
                  <div className="flex flex-col gap-0.5">
                    <span className="text-muted text-[9px] uppercase tracking-wide">Conf</span>
                    <span className="num text-primary text-xs font-semibold">{decConf}%</span>
                  </div>
                  <div className="flex flex-col gap-0.5">
                    <span className="text-muted text-[9px] uppercase tracking-wide">Agreement</span>
                    <span className="num text-primary text-xs font-semibold">{scoreToPercent(decision.agreement_score)}%</span>
                  </div>
                  {ictActive.length > 0 && (
                    <div className="ml-auto flex flex-col gap-0.5">
                      <span className="label text-[9px] mb-0.5">ICT</span>
                      {ictActive.map(([label, , d]) => (
                        <div key={label as string} className="flex items-center gap-1">
                          <span className="text-[10px] text-secondary">{label as string}</span>
                          {d && (
                            <span className={cn('text-[10px] font-bold',
                              (d as string).toUpperCase().includes('BUL') ? 'text-emerald-400' : 'text-red-400'
                            )}>{d as string}</span>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </>
            )}
            {/* ICT only (no decision yet) */}
            {!decision && ictActive.length > 0 && (
              <div>
                <span className="label text-[10px] block mb-1">ICT</span>
                <div className="flex gap-3">
                  {ictActive.map(([label, , d]) => (
                    <div key={label as string} className="flex items-center gap-1">
                      <span className="text-xs text-secondary">{label as string}</span>
                      {d && (
                        <span className={cn('text-[10px] font-bold',
                          (d as string).toUpperCase().includes('BUL') ? 'text-emerald-400' : 'text-red-400'
                        )}>{d as string}</span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}


      </div>
    </div>
  )
}
