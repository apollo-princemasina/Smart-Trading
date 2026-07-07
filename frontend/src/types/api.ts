// ── Core primitives ───────────────────────────────────────────────────────────

export type Direction    = 'BUY' | 'SELL' | 'HOLD'
export type Strength     = 'STRONG' | 'MODERATE' | 'WEAK'
export type Recommendation = 'BUY' | 'SELL' | 'WAIT'
export type ConsensusLevel = 'STRONG_CONSENSUS' | 'MODERATE_CONSENSUS' | 'WEAK_CONSENSUS' | 'NO_CONSENSUS'
export type RegimeDominant = 'MANIPULATION' | 'EXPANSION' | 'CONSOLIDATION'
export type ComponentStatus = 'ok' | 'degraded' | 'error' | 'stopped'
export type SystemStatus    = 'operational' | 'degraded' | 'error'
export type ValueType       = 'string' | 'bool' | 'int' | 'float' | 'json'
export type UserRole        = 'viewer' | 'analyst' | 'admin'

// ── Candles ───────────────────────────────────────────────────────────────────

export interface Candle {
  timestamp: string
  open:      number
  high:      number
  low:       number
  close:     number
  volume:    number
}

// ── ICT Signals ───────────────────────────────────────────────────────────────

export interface ICTSignals {
  liquidity_sweep:   boolean
  sweep_direction:   string
  sweep_rejected:    boolean
  sweep_confirmed:   boolean
  choch_detected:    boolean
  choch_direction:   string
  bos_detected:      boolean
  bos_direction:     string
  fvg_active:        boolean
  fvg_direction:     string
  ob_active:         boolean
  ob_direction:      string
  in_order_block:    boolean
}

// ── Market regime ─────────────────────────────────────────────────────────────

export interface RegimeScores {
  consolidation: number
  expansion:     number
  manipulation:  number
}

export interface RegimeData {
  dominant:          RegimeDominant
  scores:            RegimeScores
  bias:              string
  pd_zone:           string
  atr_pips:          number | null
  atr_vs_avg?:       string | null
  adx:               number | null
  narrative:         string
  trade_impl?:       string
  trade_implication?:string
  ict:               ICTSignals
}

// ── Multi-model conviction ─────────────────────────────────────────────────────

export type ConvictionLevel =
  | 'HIGH_CONVICTION'    // 1b + 4b + 8b all agree
  | 'SETUP_FORMING'      // 4b + 8b agree, 1b is HOLD (structural setup, entry not yet triggered)
  | 'DIRECTIONAL_BIAS'   // only one lookahead model is directional
  | 'CONFLICTED'         // lookahead models disagree
  | 'NEUTRAL'            // all HOLD

export interface ConvictionData {
  level:          ConvictionLevel
  structural_dir: Direction | null
  description:    string
  direction_4b:   Direction
  direction_8b:   Direction
  prob_buy_4b:    number
  prob_sell_4b:   number
  prob_hold_4b:   number
  prob_buy_8b:    number
  prob_sell_8b:   number
  prob_hold_8b:   number
}

// ── ML prediction / signal ────────────────────────────────────────────────────

export interface Signal {
  id:              string
  signal_time:     string
  symbol:          string
  timeframe:       string
  direction:       Direction
  raw_direction?:  Direction
  demoted?:        boolean
  confidence:      number
  raw_confidence?: number
  prob_sell:       number
  prob_hold:       number
  prob_buy:        number
  close:           number
  atr_pips:        number | null
  tp_price:        number | null
  sl_price:        number | null
  tp_pips:         number | null
  sl_pips:         number | null
  session?:                 string
  session_mult?:            number
  regime:                   RegimeData | string | null
  conviction?:              ConvictionData | null
  conviction_gate_applied?: boolean
  setup_forming_alert?:     Direction | null   // directional alert when Strategy B gate demoted to HOLD
  // ICT OB entry tracking
  ict_ob_entry?:            boolean
  ict_sm_state?:            'IDLE' | 'ARMED' | 'OB_TESTED'
  ict_sm_direction?:        Direction | null
  ob_bullish_top?:          number | null
  ob_bullish_bottom?:       number | null
  ob_bearish_top?:          number | null
  ob_bearish_bottom?:       number | null
}

// ── Dashboard snapshot ────────────────────────────────────────────────────────

export interface DecisionSummary {
  recommendation:       Recommendation
  strength:             Strength
  confidence:           number
  agreement_score:      number
  conflict_score:       number
  consensus_level:      ConsensusLevel
  technical_alignment:  string
  fundamental_alignment:string
  market_bias:          string
  has_ml:               boolean
  has_eie:              boolean
  has_mia:              boolean
  schema_version:       string
  generated_at?:        string
  primary_reasons?:     string[]
  conflicting_factors?: string[]
  risk_factors?:        string[]
  expiry_minutes?:      number | null
}

export interface PredictionSummary {
  direction:   Direction
  confidence:  number
  symbol:      string
  timeframe:   string
  signal_time: string
}

export interface RegimeSummary {
  dominant:   RegimeDominant
  narrative:  string
  bias:       string
  atr_pips?:  number | null
  scores?:    RegimeScores
}

export interface MIASummary {
  market_summary?:   string      // Groq-generated institutional market narrative
  market_bias?:      string      // BULLISH | BEARISH | NEUTRAL | UNCERTAIN
  confidence?:       number      // 0.0 – 1.0
  risk_level?:       string      // LOW | MEDIUM | HIGH | CRITICAL
  expected_duration?:string      // IMMEDIATE | SHORT_TERM | MEDIUM_TERM | LONG_TERM
  is_fallback?:      boolean
  timestamp?:        string
  // legacy fields kept for compatibility
  narrative?:        string
}

export interface EIEEvent {
  title:    string
  currency: string
  impact:   string
  time:     string
  forecast?:string
  previous?:string
}

export interface EIESummary {
  active_count: number
  upcoming?:    EIEEvent[]
  narrative?:   string
}

export interface BufferSummary {
  ready:       boolean
  bar_count?:  number
  symbol?:     string
  timeframe?:  string
}

export interface SystemSummary {
  status:          SystemStatus
  uptime_seconds:  number
  engines_online:  string[]
}

export interface DashboardSnapshot {
  decision:          DecisionSummary | null
  latest_prediction: PredictionSummary | null
  market_regime:     RegimeSummary | null
  mia_summary:       MIASummary | null
  eie_summary:       EIESummary | null
  buffer_status:     BufferSummary | null
  system_summary:    SystemSummary | null
}

// ── System health ─────────────────────────────────────────────────────────────

export interface ComponentHealth {
  status:  ComponentStatus
  [key:string]: unknown
}

export interface SystemHealthResponse {
  status:          SystemStatus
  uptime_seconds:  number
  components:      Record<string, ComponentHealth>
}

export interface SystemVersionResponse {
  api_version:      string
  schema_version:   string
  active_model_id?: string | null
}

export interface SystemLogEntry {
  id:             string
  level:          string
  component:      string
  event_type:     string
  message:        string
  details?:       Record<string, unknown>
  correlation_id?:string
  created_at:     string
}

export interface SystemLogsResponse {
  logs:  SystemLogEntry[]
  total: number
}

// ── Settings ──────────────────────────────────────────────────────────────────

export interface SettingOut {
  key:         string
  value:       unknown
  value_type:  ValueType
  category:    string
  description: string
  is_secret:   boolean
  updated_at:  string
}

export interface SettingsListResponse  { settings: SettingOut[]; total: number }
export interface SettingResponse       { setting: SettingOut }

// ── Model registry ────────────────────────────────────────────────────────────

export interface ModelRegistryItem {
  id:                     string
  model_name:             string
  model_version:          string
  bundle_path:            string
  feature_count?:         number | null
  is_active:              boolean
  git_commit?:            string | null
  feature_schema_version?:string | null
  label_version?:         string | null
  decision_schema_version?:string | null
  pipeline_version?:      string | null
  precision_buy?:         number | null
  precision_sell?:        number | null
  recall_buy?:            number | null
  recall_sell?:           number | null
  f1_buy?:                number | null
  f1_sell?:               number | null
  notes?:                 string | null
  created_at:             string
  activated_at?:          string | null
}

export interface ModelListResponse { models: ModelRegistryItem[]; total: number }
export interface ModelResponse     { model:  ModelRegistryItem | null }

// ── History ───────────────────────────────────────────────────────────────────

export interface DecisionHistoryItem {
  id:                   string
  decision_id:          string
  recommendation:       Recommendation
  strength:             Strength
  confidence:           number
  agreement_score:      number
  conflict_score:       number
  consensus_level:      ConsensusLevel
  technical_alignment:  string
  fundamental_alignment:string
  market_bias:          string
  has_ml:               boolean
  has_eie:              boolean
  has_mia:              boolean
  schema_version:       string
  generated_at:         string
}

export interface PredictionHistoryItem {
  id:             string
  symbol:         string
  timeframe:      string | null
  signal_time:    string
  // Adjusted output
  direction:      Direction
  confidence:     number
  // Raw model output
  raw_direction?: Direction
  raw_confidence?: number
  demoted?:       boolean
  // Full probability vector
  prob_buy?:      number
  prob_sell?:     number
  prob_hold?:     number
  // Session
  session?:       string
  session_mult?:  number
  // Market context
  regime?:        string | null
  close?:         number | null
  tp_price?:      number | null
  sl_price?:      number | null
  tp_pips?:       number | null
  sl_pips?:       number | null
  atr_pips?:      number | null
}

export interface DecisionHistoryResponse {
  decisions: DecisionHistoryItem[]
  total:     number
  page:      number
  page_size: number
}

export interface PredictionHistoryResponse {
  predictions: PredictionHistoryItem[]
  total:       number
  page:        number
  page_size:   number
}

// ── Auth ──────────────────────────────────────────────────────────────────────

export interface TokenResponse {
  access_token:  string
  refresh_token: string
  token_type:    string
  expires_in:    number
}

export interface UserOut {
  id:                string
  email:             string
  username:          string
  role:              UserRole
  subscription_tier: string
  is_active:         boolean
  created_at:        string
}
