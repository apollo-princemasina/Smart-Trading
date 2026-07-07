'use client'
import { create } from 'zustand'
import type { DecisionSummary, Signal, RegimeData, MIASummary, EIESummary, SystemSummary } from '@/types/api'
import type { WSMessage } from '@/types/ws'

interface WSLiveState {
  // Connection
  connected:       boolean
  connectedAt:     number | null
  reconnectCount:  number

  // Live data
  latestDecision:     DecisionSummary | null
  latestSignal:       Signal | null
  currentRegime:      RegimeData | null
  miaUpdate:          MIASummary | null
  eieUpdate:          EIESummary | null
  systemStatus:       SystemSummary | null
  lastSchedulerTick:  string | null
  lastUpdateAt:       number | null

  // Actions
  setConnected:       (connected: boolean) => void
  incrementReconnect: () => void
  handleWSMessage:    (msg: WSMessage) => void
  reset:              () => void
}

const initialData = {
  latestDecision:    null,
  latestSignal:      null,
  currentRegime:     null,
  miaUpdate:         null,
  eieUpdate:         null,
  systemStatus:      null,
  lastSchedulerTick: null,
  lastUpdateAt:      null,
}

export const useWSStore = create<WSLiveState>((set) => ({
  connected:      false,
  connectedAt:    null,
  reconnectCount: 0,
  ...initialData,

  setConnected: (connected) =>
    set(connected
      ? { connected: true, connectedAt: Date.now() }
      : { connected: false }),

  incrementReconnect: () =>
    set(s => ({ reconnectCount: s.reconnectCount + 1 })),

  handleWSMessage: (msg) =>
    set(s => {
      const now = Date.now()
      switch (msg.event) {
        case 'decision_update': {
          // DFE sends agreement/conflict scores as 0-100; normalize to 0-1 for display components
          // DFE sends technical/fundamental alignment as floats -1.0..+1.0; map to readable labels
          const raw = msg.data as any
          const alignLabel = (v: number | null | undefined): string => {
            if (v == null) return '—'
            if (v >= 0.6)  return 'BULLISH'
            if (v >= 0.2)  return 'SLIGHT BULLISH'
            if (v > -0.2)  return 'NEUTRAL'
            if (v > -0.6)  return 'SLIGHT BEARISH'
            return 'BEARISH'
          }
          const decision: DecisionSummary = {
            ...raw,
            agreement_score:       (raw.agreement_score ?? 0) / 100,
            conflict_score:        (raw.conflict_score  ?? 0) / 100,
            technical_alignment:   alignLabel(raw.technical_alignment),
            fundamental_alignment: alignLabel(raw.fundamental_alignment),
          }
          return { ...s, latestDecision: decision, lastUpdateAt: now }
        }
        case 'signal_update':
          return { ...s, latestSignal: msg.data as Signal, lastUpdateAt: now }
        case 'regime_update':
          return { ...s, currentRegime: msg.data as RegimeData, lastUpdateAt: now }
        case 'mia_update':
          return { ...s, miaUpdate: msg.data as MIASummary, lastUpdateAt: now }
        case 'eie_update':
          return { ...s, eieUpdate: msg.data as EIESummary, lastUpdateAt: now }
        case 'system_status':
          return { ...s, systemStatus: msg.data as SystemSummary, lastUpdateAt: now }
        case 'scheduler_tick':
          return { ...s, lastSchedulerTick: msg.timestamp, lastUpdateAt: now }
        default:
          return s
      }
    }),

  reset: () => set(initialData),
}))
