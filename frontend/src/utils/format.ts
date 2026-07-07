// ── Price / pip formatting ─────────────────────────────────────────────────────

export function formatPrice(value: number | null | undefined, decimals = 5): string {
  if (value == null) return '—'
  return value.toFixed(decimals)
}

export function formatPips(value: number | null | undefined): string {
  if (value == null) return '—'
  return `${value.toFixed(1)} pips`
}

export function formatPercent(value: number | null | undefined, decimals = 1): string {
  if (value == null) return '—'
  return `${(value * 100).toFixed(decimals)}%`
}

export function formatConfidence(value: number | null | undefined): string {
  if (value == null) return '—'
  const pct = value <= 1 ? value * 100 : value
  return `${pct.toFixed(1)}%`
}

export function confidenceToPercent(value: number): number {
  return value <= 1 ? Math.round(value * 100) : Math.round(value)
}

// ── Time formatting ────────────────────────────────────────────────────────────

export function formatTime(iso: string | null | undefined): string {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleTimeString('en-GB', {
      hour: '2-digit', minute: '2-digit', second: '2-digit',
      timeZone: 'UTC',
    }) + ' UTC'
  } catch {
    return iso
  }
}

export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return '—'
  try {
    const d = new Date(iso)
    return d.toLocaleString('en-GB', {
      day:    '2-digit',
      month:  'short',
      hour:   '2-digit',
      minute: '2-digit',
      timeZone: 'UTC',
    }) + ' UTC'
  } catch {
    return iso
  }
}

export function formatRelative(iso: string | null | undefined): string {
  if (!iso) return '—'
  try {
    // Ensure the string is parsed as UTC — backend omits the Z suffix
    const utcIso = iso.includes('Z') || iso.includes('+') ? iso : iso + 'Z'
    const diff = Date.now() - new Date(utcIso).getTime()
    const s = Math.floor(diff / 1000)
    if (s < 60)   return `${s}s ago`
    const m = Math.floor(s / 60)
    if (m < 60)   return `${m}m ago`
    const h = Math.floor(m / 60)
    if (h < 24)   return `${h}h ago`
    return `${Math.floor(h / 24)}d ago`
  } catch {
    return iso
  }
}

export function formatUptime(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  if (h > 0) return `${h}h ${m}m`
  if (m > 0) return `${m}m ${s}s`
  return `${s}s`
}

// ── Score / agreement formatting ──────────────────────────────────────────────

export function scoreToPercent(value: number): number {
  // DFE returns scores as 0–100; normalize values > 1 to avoid double-multiply
  return Math.round(value > 1 ? value : value * 100)
}

export function formatScore(value: number | null | undefined): string {
  if (value == null) return '—'
  return `${scoreToPercent(value)}%`
}

// ── Decision/regime label helpers ─────────────────────────────────────────────

export function toDisplayLabel(raw: string | null | undefined): string {
  if (!raw) return '—'
  return raw
    .replace(/_/g, ' ')
    .replace(/([A-Z])/g, ' $1')
    .trim()
    .replace(/\b\w/g, c => c.toUpperCase())
}

export function consensusToLabel(level: string): string {
  const map: Record<string, string> = {
    STRONG_CONSENSUS:   'Strong',
    MODERATE_CONSENSUS: 'Moderate',
    WEAK_CONSENSUS:     'Weak',
    NO_CONSENSUS:       'None',
  }
  return map[level] ?? level
}

export function impactColor(impact: string): string {
  const map: Record<string, string> = {
    HIGH:   'text-sell',
    MEDIUM: 'text-wait',
    LOW:    'text-muted',
  }
  return map[impact?.toUpperCase()] ?? 'text-muted'
}

export function impactLabel(impact: string): string {
  const i = impact?.toUpperCase()
  if (i === 'HIGH')   return '●●●'
  if (i === 'MEDIUM') return '●●○'
  return '●○○'
}
