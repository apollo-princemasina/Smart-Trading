'use client'
import { useKillzone, getSignalImportance } from '@/hooks/useKillzone'
import type { Signal } from '@/types'

const DIR_COLOR = { BUY: '#10B981', SELL: '#EF4444', HOLD: '#F59E0B' }
const DIR_ARROW = { BUY: '▲', SELL: '▼', HOLD: '◆' }

function ProbBar({ sell, hold, buy }: { sell: number; hold: number; buy: number }) {
  return (
    <div className="mt-4">
      <div className="flex text-xs text-muted mb-1 justify-between">
        <span>SELL {Math.round(sell * 100)}%</span>
        <span>HOLD {Math.round(hold * 100)}%</span>
        <span>BUY {Math.round(buy * 100)}%</span>
      </div>
      <div className="flex h-2 rounded-full overflow-hidden">
        <div style={{ width: `${sell * 100}%`, background: '#EF4444' }} />
        <div style={{ width: `${hold * 100}%`, background: '#374151' }} />
        <div style={{ width: `${buy  * 100}%`, background: '#10B981' }} />
      </div>
    </div>
  )
}

function Row({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div className="flex justify-between py-1.5 border-b border-border/40">
      <span className="text-muted text-xs">{label}</span>
      <span className={`text-xs tabular-nums font-mono ${accent ?? 'text-white'}`}>{value}</span>
    </div>
  )
}

export default function SignalPanel({ signal }: { signal: Signal | null }) {
  const killzone = useKillzone()

  if (!signal) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-6 text-muted text-xs text-center">
        <div className="text-4xl mb-4 opacity-20">◆</div>
        <div>Waiting for signal…</div>
        <div className="mt-1 opacity-60">Next check at next M15 bar close</div>
      </div>
    )
  }

  const dir        = signal.direction as 'BUY' | 'SELL' | 'HOLD'
  const color      = DIR_COLOR[dir]
  const arrow      = DIR_ARROW[dir]
  const conf       = Math.round(signal.confidence * 100)
  const ts         = new Date(signal.signal_time).toUTCString().slice(5, 22)
  const importance = getSignalImportance(killzone, dir)

  // Signal age — stale if > 30 min (2 M15 bars)
  const ageMs   = Date.now() - new Date(signal.signal_time).getTime()
  const ageMin  = Math.floor(ageMs / 60_000)
  const ageText = ageMin < 60
    ? `${ageMin}m ago`
    : `${Math.floor(ageMin / 60)}h ${ageMin % 60}m ago`
  const isStale = ageMin > 30

  return (
    <div className="flex flex-col h-full p-4 overflow-y-auto">
      <div className="flex items-center justify-between mb-4">
        <div className="text-xs text-muted uppercase tracking-wider">Signal</div>
        {isStale && (
          <span className="text-xs px-1.5 py-0.5 rounded" style={{ color: '#F59E0B', background: '#F59E0B18', border: '1px solid #F59E0B44' }}>
            STALE
          </span>
        )}
      </div>

      {/* Direction card */}
      <div
        className="rounded-lg p-4 mb-4 text-center"
        style={{
          border: `1px solid ${isStale ? '#F59E0B44' : color + '22'}`,
          background: `${isStale ? '#F59E0B' : color}0a`,
          boxShadow: importance.glow && !isStale ? `0 0 18px ${importance.color}44` : undefined,
          opacity: isStale ? 0.75 : 1,
        }}
      >
        <div className="text-5xl mb-1" style={{ color: isStale ? '#F59E0B' : color }}>{arrow}</div>
        <div className="text-2xl font-bold tracking-widest" style={{ color: isStale ? '#F59E0B' : color }}>{dir}</div>
        <div className="text-sm mt-1" style={{ color: `${isStale ? '#F59E0B' : color}cc` }}>{conf}% confidence</div>

        {/* Show raw model confidence when session weighting was applied */}
        {signal.raw_confidence && Math.abs(signal.raw_confidence - signal.confidence) > 0.005 && (
          <div className="text-xs text-muted mt-0.5 tabular-nums">
            model: {Math.round(signal.raw_confidence * 100)}% × {signal.session_mult?.toFixed(2)} session
          </div>
        )}

        {/* Importance badge */}
        <div className="mt-2 flex justify-center">
          <span
            className="px-2 py-0.5 rounded text-xs font-bold tracking-widest"
            style={{
              color: importance.color,
              background: `${importance.color}18`,
              border: `1px solid ${importance.color}55`,
            }}
          >
            {importance.label}
          </span>
        </div>
      </div>

      {/* Trade details */}
      <Row label="Entry"    value={signal.close?.toFixed(5) ?? '—'} />
      <Row label="TP"       value={signal.tp_price ? `${signal.tp_price.toFixed(5)}  (+${signal.tp_pips}p)` : '—'} accent="text-bull" />
      <Row label="SL"       value={signal.sl_price ? `${signal.sl_price.toFixed(5)}  (−${signal.sl_pips}p)` : '—'} accent="text-bear" />
      <Row label="R:R"      value={signal.tp_pips && signal.sl_pips ? `${(signal.tp_pips / signal.sl_pips).toFixed(1)}` : '—'} />
      <Row label="ATR"      value={signal.atr_pips ? `${signal.atr_pips} pips` : '—'} />
      <Row label="Signal at" value={ts} />

      {/* Regime badge */}
      {signal.regime && typeof signal.regime === 'object' && (
        <div className="mt-3 px-2 py-1 rounded bg-border/30 text-xs text-center">
          <span className="text-muted">Regime: </span>
          <span className="font-semibold" style={{
            color: signal.regime.dominant === 'MANIPULATION' ? '#8B5CF6'
                 : signal.regime.dominant === 'EXPANSION' ? '#3B82F6' : '#6B7280'
          }}>
            {signal.regime.dominant}
          </span>
        </div>
      )}

      {/* Probability bar */}
      <ProbBar sell={signal.prob_sell} hold={signal.prob_hold} buy={signal.prob_buy} />
    </div>
  )
}
