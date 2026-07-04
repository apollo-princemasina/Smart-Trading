'use client'
import { useEffect, useState } from 'react'

export type KillzoneName =
  | 'ASIAN'
  | 'LONDON OPEN'
  | 'NY OPEN'
  | 'LONDON CLOSE'
  | 'DEAD ZONE'
  | 'MARKET CLOSED'

export interface Killzone {
  name: KillzoneName
  active: boolean
  color: string
  utcRange: string
  description: string
}

// ICT killzones in UTC hours (fixed — no DST adjustment, works year-round for EURUSD)
const ZONES: Array<{ name: KillzoneName; start: number; end: number; color: string; utcRange: string; description: string }> = [
  {
    name: 'ASIAN',
    start: 0,
    end: 4,
    color: '#8B5CF6',
    utcRange: '00:00 – 04:00',
    description: 'Range formation, liquidity build-up',
  },
  {
    name: 'LONDON OPEN',
    start: 7,
    end: 10,
    color: '#3B82F6',
    utcRange: '07:00 – 10:00',
    description: 'Highest volatility window for EURUSD',
  },
  {
    name: 'NY OPEN',
    start: 12,
    end: 15,
    color: '#10B981',
    utcRange: '12:00 – 15:00',
    description: 'Second major institutional window',
  },
  {
    name: 'LONDON CLOSE',
    start: 15,
    end: 17,
    color: '#F59E0B',
    utcRange: '15:00 – 17:00',
    description: 'Retracement / position unwind zone',
  },
]

function isMarketClosed(now: Date): boolean {
  const day  = now.getUTCDay()   // 0=Sun, 1=Mon … 5=Sat, 6=Sun (wrong — 0=Sun, 6=Sat)
  // JS: 0=Sun, 1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri, 6=Sat
  const hour = now.getUTCHours()
  return (
    (day === 5 && hour >= 22) ||  // Friday after 22:00 UTC
    day === 6 ||                   // All Saturday
    (day === 0 && hour < 22)       // Sunday before 22:00 UTC
  )
}

function getKillzone(now: Date): Killzone {
  if (isMarketClosed(now)) {
    return {
      name: 'MARKET CLOSED',
      active: false,
      color: '#374151',
      utcRange: 'Fri 22:00 – Sun 22:00',
      description: 'Forex markets closed for the weekend',
    }
  }
  const utcHour = now.getUTCHours()
  const zone = ZONES.find(z => utcHour >= z.start && utcHour < z.end)
  if (zone) {
    return { ...zone, active: true }
  }
  return {
    name: 'DEAD ZONE',
    active: false,
    color: '#4B5563',
    utcRange: '04:00 – 07:00 / 17:00+',
    description: 'Low institutional participation',
  }
}

// Signal importance based on killzone + direction
export function getSignalImportance(
  killzone: Killzone,
  direction: string,
): { label: string; color: string; glow: boolean } {
  if (killzone.name === 'MARKET CLOSED') {
    return { label: 'MARKET CLOSED', color: '#374151', glow: false }
  }
  if (!killzone.active || direction === 'HOLD') {
    return { label: 'NORMAL', color: '#4B5563', glow: false }
  }
  if (killzone.name === 'LONDON OPEN' || killzone.name === 'NY OPEN') {
    return { label: 'PRIME SETUP', color: '#F0B429', glow: true }
  }
  return { label: 'HIGH', color: '#3B82F6', glow: false }
}

export function useKillzone() {
  const [killzone, setKillzone] = useState<Killzone>(() =>
    getKillzone(new Date()),
  )

  useEffect(() => {
    const update = () => setKillzone(getKillzone(new Date()))
    update()
    // Re-check every minute — killzone boundaries are on the hour
    const id = setInterval(update, 60_000)
    return () => clearInterval(id)
  }, [])

  return killzone
}
