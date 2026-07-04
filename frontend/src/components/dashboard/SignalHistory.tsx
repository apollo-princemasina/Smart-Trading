'use client'
import type { Signal } from '@/types'

const DIR_COLOR = { BUY: '#10B981', SELL: '#EF4444', HOLD: '#F59E0B' }

export default function SignalHistory({ signals }: { signals: Signal[] }) {
  if (signals.length === 0) {
    return (
      <div className="px-4 py-2 text-xs text-muted border-t border-border">
        No signals yet — history will appear here after the first inference cycle.
      </div>
    )
  }

  return (
    <div className="flex items-center gap-2 px-4 py-2 overflow-x-auto border-t border-border">
      <span className="text-xs text-muted shrink-0">History:</span>
      {[...signals].reverse().slice(0, 12).map((s, i) => {
        const dir   = s.direction as keyof typeof DIR_COLOR
        const color = DIR_COLOR[dir] ?? '#6B7280'
        const time  = new Date(s.signal_time).toUTCString().slice(17, 22)
        const conf  = Math.round(s.confidence * 100)
        return (
          <div
            key={s.id ?? i}
            className="shrink-0 flex items-center gap-1 px-2 py-1 rounded text-xs border"
            style={{ borderColor: `${color}44`, background: `${color}11` }}
          >
            <span style={{ color }} className="font-bold">{dir}</span>
            <span className="text-muted">{conf}%</span>
            <span className="text-muted/60">{time}</span>
          </div>
        )
      })}
    </div>
  )
}
