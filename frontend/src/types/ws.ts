import type {
  DecisionSummary,
  RegimeData,
  Signal,
  MIASummary,
  EIESummary,
  SystemSummary,
} from './api'

export type WSEventType =
  | 'signal_update'
  | 'regime_update'
  | 'candle_update'
  | 'health_update'
  | 'decision_update'
  | 'mia_update'
  | 'eie_update'
  | 'system_status'
  | 'scheduler_tick'
  | 'model_loaded'
  | 'connection_ack'
  | 'subscription_ack'
  | 'ping'
  | 'pong'

export interface WSMessage<T = unknown> {
  event:     WSEventType
  timestamp: string
  data:      T
}

export type DecisionWSMessage   = WSMessage<DecisionSummary>
export type SignalWSMessage      = WSMessage<Signal>
export type RegimeWSMessage      = WSMessage<RegimeData>
export type MIAWSMessage         = WSMessage<MIASummary>
export type EIEWSMessage         = WSMessage<EIESummary>
export type SystemStatusMessage  = WSMessage<SystemSummary>
export type SchedulerTickMessage = WSMessage<{ tick_time: string; symbol: string; timeframe: string }>
export type ModelLoadedMessage   = WSMessage<{ model_id: string; model_version: string }>
export type ConnectionAckMessage = WSMessage<{ message: string; server_time: string }>
