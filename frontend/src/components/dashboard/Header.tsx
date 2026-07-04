'use client'
import { useEffect, useState } from 'react'
import { useKillzone } from '@/hooks/useKillzone'

interface Props {
  close: number | null
  connected: boolean
}

export default function Header({ close, connected }: Props) {
  const [utc, setUtc] = useState('')
  const killzone = useKillzone()

  useEffect(() => {
    const tick = () => {
      const now = new Date()
      setUtc(now.toUTCString().slice(17, 25) + ' UTC')
    }
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-surface text-xs">
      <div className="flex items-center gap-6">
        <span className="text-gold font-bold text-sm tracking-widest">MFIP</span>
        <span className="text-muted">EURUSD</span>
        <span className="text-muted">M15</span>
        {close && (
          <span className="text-white font-semibold text-sm tabular-nums">
            {close.toFixed(5)}
          </span>
        )}
        <span className="text-muted">Spread 1.5p</span>

        {/* Killzone badge */}
        <span
          className="px-2 py-0.5 rounded text-xs font-semibold tracking-wider"
          style={{
            color: killzone.color,
            background: `${killzone.color}18`,
            border: `1px solid ${killzone.color}44`,
          }}
          title={`${killzone.utcRange} UTC — ${killzone.description}`}
        >
          {killzone.name}
        </span>
      </div>

      <div className="flex items-center gap-4">
        <span className="text-muted tabular-nums">{utc}</span>
        <div className="flex items-center gap-2">
          <span
            className={`inline-block w-2 h-2 rounded-full ${connected ? 'bg-bull animate-pulse' : 'bg-bear'}`}
          />
          <span className={connected ? 'text-bull' : 'text-bear'}>
            {connected ? 'LIVE' : 'OFFLINE'}
          </span>
        </div>
      </div>
    </div>
  )
}
