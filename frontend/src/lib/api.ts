const BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

export async function apiFetch<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { cache: 'no-store' })
  if (!res.ok) throw new Error(`API ${path} → ${res.status}`)
  return res.json() as Promise<T>
}

export const api = {
  latestPrediction: () => apiFetch<import('@/types').Signal>('/api/v1/predictions/latest'),
  predictions:      (page = 1) => apiFetch<{ predictions: import('@/types').Signal[] }>(`/api/v1/predictions/?page=${page}&page_size=10`),
  candles:          (tf: string, limit = 150) => apiFetch<{ candles: import('@/types').Candle[] }>(`/api/v1/market/candles/${tf}?limit=${limit}`),
  regime:           () => apiFetch<import('@/types').RegimeData>('/api/v1/market/regime'),
  health:           () => apiFetch<{ status: string }>('/api/v1/health/live'),
}
