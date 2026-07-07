import type {
  DashboardSnapshot,
  SystemHealthResponse,
  SystemVersionResponse,
  SystemLogsResponse,
  SettingsListResponse,
  SettingResponse,
  ModelListResponse,
  ModelResponse,
  ModelRegistryItem,
  DecisionHistoryResponse,
  PredictionHistoryResponse,
  TokenResponse,
  UserOut,
  Candle,
  Signal,
  RegimeData,
} from '@/types/api'

const BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

// ── Fetch wrapper ──────────────────────────────────────────────────────────────

class APIError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message)
    this.name = 'APIError'
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...init?.headers,
    },
    cache: 'no-store',
  })
  if (!res.ok) {
    let message = `API ${path} → ${res.status}`
    try { const body = await res.json(); message = body.detail ?? message } catch {}
    throw new APIError(res.status, message)
  }
  return res.json() as Promise<T>
}

function get<T>(path: string)       { return request<T>(path) }
function post<T>(path: string, body: unknown) {
  return request<T>(path, { method: 'POST', body: JSON.stringify(body) })
}
function put<T>(path: string, body: unknown) {
  return request<T>(path, { method: 'PUT', body: JSON.stringify(body) })
}

// ── Client ─────────────────────────────────────────────────────────────────────

export const api = {
  // ─ Dashboard ─────────────────────────────────────────────────────────────
  dashboard: {
    snapshot: () => get<DashboardSnapshot>('/api/v1/dashboard'),
  },

  // ─ System ────────────────────────────────────────────────────────────────
  system: {
    health:  () => get<SystemHealthResponse>('/api/v1/system/health'),
    status:  () => get<{ status: string; uptime_seconds: number; engines_online: string[] }>('/api/v1/system/status'),
    version: () => get<SystemVersionResponse>('/api/v1/system/version'),
    logs:    (params?: { level?: string; component?: string; limit?: number }) => {
      const q = new URLSearchParams()
      if (params?.level)     q.set('level',     params.level)
      if (params?.component) q.set('component', params.component)
      if (params?.limit)     q.set('limit',     String(params.limit))
      return get<SystemLogsResponse>(`/api/v1/system/logs?${q}`)
    },
  },

  // ─ Settings ──────────────────────────────────────────────────────────────
  settings: {
    list:    () => get<SettingsListResponse>('/api/v1/settings'),
    get:     (key: string) => get<SettingResponse>(`/api/v1/settings/${key}`),
    update:  (key: string, value: unknown) =>
      put<SettingResponse>(`/api/v1/settings/${key}`, { value }),
  },

  // ─ Model registry ─────────────────────────────────────────────────────────
  models: {
    list:     () => get<ModelListResponse>('/api/v1/models'),
    active:   () => get<ModelResponse>('/api/v1/models/active'),
    get:      (id: string) => get<ModelResponse>(`/api/v1/models/${id}`),
    register: (data: Partial<ModelRegistryItem> & { model_name: string; model_version: string; bundle_path: string }) =>
      post<ModelResponse>('/api/v1/models/register', data),
  },

  // ─ History ───────────────────────────────────────────────────────────────
  history: {
    decisions: (params?: {
      recommendation?: string
      strength?: string
      after?: string
      before?: string
      page?: number
      page_size?: number
    }) => {
      const q = new URLSearchParams()
      if (params?.recommendation) q.set('recommendation', params.recommendation)
      if (params?.strength)       q.set('strength',       params.strength)
      if (params?.after)          q.set('after',          params.after)
      if (params?.before)         q.set('before',         params.before)
      if (params?.page)           q.set('page',           String(params.page))
      if (params?.page_size)      q.set('page_size',      String(params.page_size))
      return get<DecisionHistoryResponse>(`/api/v1/history/decisions?${q}`)
    },
    predictions: (params?: {
      symbol?: string
      direction?: string
      page?: number
      page_size?: number
    }) => {
      const q = new URLSearchParams()
      if (params?.symbol)    q.set('symbol',    params.symbol)
      if (params?.direction) q.set('direction', params.direction)
      if (params?.page)      q.set('page',      String(params.page))
      if (params?.page_size) q.set('page_size', String(params.page_size))
      return get<PredictionHistoryResponse>(`/api/v1/history/predictions?${q}`)
    },
  },

  // ─ Auth ──────────────────────────────────────────────────────────────────
  auth: {
    login:   (email: string, password: string) =>
      post<TokenResponse>('/api/v1/auth/login', { email, password }),
    refresh: (refresh_token: string) =>
      post<TokenResponse>('/api/v1/auth/refresh', { refresh_token }),
    me:      () => get<UserOut>('/api/v1/auth/me'),
  },

  // ─ Phase 1 (kept for chart + predictions) ────────────────────────────────
  market: {
    candles: (tf: string, limit = 300) =>
      get<{ candles: Candle[] }>(`/api/v1/market/candles/${tf}?limit=${limit}`),
    regime: () => get<RegimeData>('/api/v1/market/regime'),
  },
  predictions: {
    latest: () => get<Signal>('/api/v1/predictions/latest'),
    list:   (page = 1) =>
      get<{ predictions: Signal[] }>(`/api/v1/predictions/?page=${page}&page_size=10`),
  },
  health: {
    live: () => get<{ status: string }>('/api/v1/health/live'),
  },
}

export { APIError }
