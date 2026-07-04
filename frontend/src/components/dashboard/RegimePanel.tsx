'use client'
import { useKillzone } from '@/hooks/useKillzone'
import type { RegimeData } from '@/types'

const REGIME_COLORS: Record<string, string> = {
  CONSOLIDATION: '#6B7280',
  EXPANSION:     '#3B82F6',
  MANIPULATION:  '#8B5CF6',
}

const ICT_LABEL: Record<string, string> = {
  BULLISH: 'BULL',
  BEARISH: 'BEAR',
  NONE:    'NONE',
}

function Bar({ label, value, dominant }: { label: string; value: number; dominant: boolean }) {
  const color = REGIME_COLORS[label] ?? '#6B7280'
  const pct   = Math.round(value * 100)
  return (
    <div className="mb-3">
      <div className="flex justify-between mb-1">
        <span className={`text-xs ${dominant ? 'text-white font-semibold' : 'text-muted'}`}>{label}</span>
        <span className={`text-xs tabular-nums ${dominant ? 'text-white' : 'text-muted'}`}>{pct}%</span>
      </div>
      <div className="h-1.5 bg-border rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, backgroundColor: color, opacity: dominant ? 1 : 0.45 }}
        />
      </div>
    </div>
  )
}

function KillzoneBadge({ killzone }: { killzone: ReturnType<typeof useKillzone> }) {
  return (
    <div className="mt-4 rounded p-3" style={{ background: `${killzone.color}12`, border: `1px solid ${killzone.color}33` }}>
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-muted uppercase tracking-wider">Session</span>
        {killzone.active && (
          <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: killzone.color }} />
        )}
      </div>
      <div className="text-xs font-bold mb-0.5" style={{ color: killzone.color }}>{killzone.name}</div>
      <div className="text-xs text-muted">{killzone.utcRange} UTC</div>
      <div className="text-xs text-muted/70 mt-1 leading-snug">{killzone.description}</div>
    </div>
  )
}

function BoolRow({ label, value, direction }: { label: string; value: boolean; direction?: string }) {
  const dir = direction && direction !== 'NONE' ? ` ${ICT_LABEL[direction] ?? direction}` : ''
  return (
    <div className="flex justify-between py-0.5 border-b border-border/40">
      <span className="text-muted text-xs">{label}</span>
      <span className={`text-xs font-mono ${value ? 'text-bull' : 'text-muted/50'}`}>
        {value ? `YES${dir}` : 'NO'}
      </span>
    </div>
  )
}

export default function RegimePanel({ regime }: { regime: RegimeData | null }) {
  const killzone = useKillzone()

  if (!regime) {
    return (
      <div className="flex flex-col gap-2 p-4 h-full">
        <div className="text-xs text-muted uppercase tracking-wider mb-3">Market Regime</div>
        <div className="text-muted text-xs">Waiting for first inference cycle…</div>

        {/* Show killzone even while waiting for regime */}
        <KillzoneBadge killzone={killzone} />
      </div>
    )
  }

  const { dominant, scores, ict, bias, pd_zone, atr_pips, adx } = regime

  return (
    <div className="flex flex-col h-full p-4 overflow-y-auto">
      <div className="text-xs text-muted uppercase tracking-wider mb-4">Market Regime</div>

      {/* Regime bars */}
      <Bar label="MANIPULATION"  value={scores.manipulation}  dominant={dominant === 'MANIPULATION'} />
      <Bar label="EXPANSION"     value={scores.expansion}     dominant={dominant === 'EXPANSION'} />
      <Bar label="CONSOLIDATION" value={scores.consolidation} dominant={dominant === 'CONSOLIDATION'} />

      {/* Context */}
      <div className="mt-4 mb-3 grid grid-cols-2 gap-2">
        <div className="bg-border/30 rounded p-2">
          <div className="text-muted text-xs mb-0.5">Bias</div>
          <div className={`text-xs font-semibold ${bias === 'BULLISH' ? 'text-bull' : bias === 'BEARISH' ? 'text-bear' : 'text-muted'}`}>{bias}</div>
        </div>
        <div className="bg-border/30 rounded p-2">
          <div className="text-muted text-xs mb-0.5">P/D Zone</div>
          <div className={`text-xs font-semibold ${pd_zone === 'PREMIUM' ? 'text-bear' : pd_zone === 'DISCOUNT' ? 'text-bull' : 'text-muted'}`}>{pd_zone}</div>
        </div>
        {atr_pips && (
          <div className="bg-border/30 rounded p-2">
            <div className="text-muted text-xs mb-0.5">ATR</div>
            <div className="text-xs text-white tabular-nums">{atr_pips}p</div>
          </div>
        )}
        {adx && (
          <div className="bg-border/30 rounded p-2">
            <div className="text-muted text-xs mb-0.5">ADX</div>
            <div className={`text-xs tabular-nums ${adx >= 40 ? 'text-bull' : adx >= 25 ? 'text-hold' : 'text-muted'}`}>{adx.toFixed(1)}</div>
          </div>
        )}
      </div>

      {/* ICT Signals */}
      <div className="text-xs text-muted uppercase tracking-wider mb-2 mt-1">ICT / SMC Signals</div>
      <div className="space-y-0">
        <BoolRow label="Liquidity Sweep"  value={ict.liquidity_sweep}  direction={ict.sweep_direction} />
        <BoolRow label="Sweep Rejected"   value={ict.sweep_rejected} />
        <BoolRow label="Sweep Confirmed"  value={ict.sweep_confirmed} />
        <BoolRow label="CHoCH"            value={ict.choch_detected}  direction={ict.choch_direction} />
        <BoolRow label="BOS"              value={ict.bos_detected}    direction={ict.bos_direction} />
        <BoolRow label="FVG Active"       value={ict.fvg_active}      direction={ict.fvg_direction} />
        <BoolRow label="OB Active"        value={ict.ob_active}       direction={ict.ob_direction} />
        <BoolRow label="Price in OB"      value={ict.in_order_block} />
      </div>

      {/* Session / Killzone context */}
      <KillzoneBadge killzone={killzone} />
    </div>
  )
}
