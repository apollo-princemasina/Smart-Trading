'use client'
import { useCallback, useEffect, useRef, useState } from 'react'
import dynamic from 'next/dynamic'
import Header from '@/components/dashboard/Header'
import RegimePanel from '@/components/dashboard/RegimePanel'
import SignalPanel from '@/components/dashboard/SignalPanel'
import SignalHistory from '@/components/dashboard/SignalHistory'
import { useWebSocket } from '@/hooks/useWebSocket'
import { api } from '@/lib/api'
import type { Candle, RegimeData, Signal, WSMessage } from '@/types'

// Chart uses window — load client-side only
const ChartPanel = dynamic(() => import('@/components/dashboard/ChartPanel'), { ssr: false })

export default function Dashboard() {
  const [signal,  setSignal]  = useState<Signal | null>(null)
  const [regime,  setRegime]  = useState<RegimeData | null>(null)
  const [candles, setCandles] = useState<Candle[]>([])
  const [history, setHistory] = useState<Signal[]>([])

  // Handle WebSocket messages
  const onMessage = useCallback((msg: WSMessage) => {
    if (msg.event === 'signal_update') {
      const s = msg.data as Signal
      setSignal(s)
      if (s.regime && typeof s.regime === 'object') setRegime(s.regime as RegimeData)
      setHistory(prev => {
        if (prev.some(p => p.signal_time === s.signal_time)) return prev
        return [...prev.slice(-19), s]
      })
    }
    if (msg.event === 'regime_update') {
      setRegime(msg.data as RegimeData)
    }
  }, [])

  const connected = useWebSocket(onMessage)

  // Initial data load on mount
  useEffect(() => {
    api.latestPrediction()
      .then((s) => {
        setSignal(s)
        // WebSocket sends regime as object; REST sends it as string — fetch separately
        if (s.regime && typeof s.regime === 'object') setRegime(s.regime as RegimeData)
      })
      .catch(() => { /* no predictions yet */ })

    // Fetch regime directly from the market endpoint (available after first scheduler tick)
    api.regime()
      .then(r => setRegime(r))
      .catch(() => { /* not yet available — will arrive via WebSocket */ })

    api.predictions()
      .then(r => {
        // Deduplicate by signal_time — keep only one entry per bar close
        const seen = new Set<string>()
        const unique = r.predictions.filter(p => {
          if (seen.has(p.signal_time)) return false
          seen.add(p.signal_time)
          return true
        })
        setHistory(unique)
      })
      .catch(() => {})

    api.candles('M15', 400)
      .then(r => setCandles(r.candles))
      .catch(() => {})
  }, [])

  // Poll candles every 60 s to keep chart fresh
  useEffect(() => {
    const id = setInterval(() => {
      api.candles('M15', 400).then(r => setCandles(r.candles)).catch(() => {})
    }, 60_000)
    return () => clearInterval(id)
  }, [])

  const latestClose = candles.length > 0 ? candles[candles.length - 1].close : signal?.close ?? null

  return (
    <div className="flex flex-col h-screen bg-bg text-white overflow-hidden">

      {/* Header */}
      <Header close={latestClose} connected={connected} />

      {/* Main 3-column layout */}
      <div className="flex flex-1 overflow-hidden" style={{ minHeight: 0 }}>

        {/* Left — Regime + ICT */}
        <div className="w-56 shrink-0 border-r border-border overflow-y-auto bg-surface">
          <RegimePanel regime={regime} />
        </div>

        {/* Centre — Chart */}
        <div className="flex-1 overflow-hidden bg-bg">
          <ChartPanel candles={candles} signals={history} />
        </div>

        {/* Right — Signal card */}
        <div className="w-56 shrink-0 border-l border-border overflow-y-auto bg-surface">
          <SignalPanel signal={signal} />
        </div>
      </div>

      {/* Bottom — Signal history strip */}
      <SignalHistory signals={history} />
    </div>
  )
}
