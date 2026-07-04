export interface Candle {
  timestamp: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface ICTSignals {
  liquidity_sweep: boolean
  sweep_direction: string
  sweep_rejected: boolean
  sweep_confirmed: boolean
  choch_detected: boolean
  choch_direction: string
  bos_detected: boolean
  bos_direction: string
  fvg_active: boolean
  fvg_direction: string
  ob_active: boolean
  ob_direction: string
  in_order_block: boolean
}

export interface RegimeScores {
  consolidation: number
  expansion: number
  manipulation: number
}

export interface RegimeData {
  dominant: string
  scores: RegimeScores
  bias: string
  pd_zone: string
  atr_pips: number | null
  atr_vs_avg?: string | null
  adx: number | null
  narrative: string
  trade_impl?: string          // WebSocket payload
  trade_implication?: string  // REST /market/regime payload
  ict: ICTSignals
}

export interface Signal {
  id: string
  signal_time: string
  symbol: string
  timeframe: string
  direction: 'BUY' | 'SELL' | 'HOLD'
  confidence: number        // session-adjusted
  raw_confidence?: number   // raw model output before session weighting
  prob_sell: number
  prob_hold: number
  prob_buy: number
  close: number
  atr_pips: number | null
  tp_price: number | null
  sl_price: number | null
  tp_pips: number | null
  sl_pips: number | null
  session?: string          // LONDON_OPEN | NY_OPEN | ASIAN | LONDON_CLOSE | DEAD_ZONE
  session_mult?: number     // e.g. 0.60 for dead zone
  regime: RegimeData | string | null
}

export interface WSMessage {
  event: string
  timestamp: string
  data: Signal | RegimeData | Record<string, unknown>
}
