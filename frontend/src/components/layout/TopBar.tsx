'use client'
import { useEffect, useState } from 'react'
import Link from 'next/link'
import { cn } from '@/utils/cn'
import { useUIStore } from '@/stores/uiStore'
import { useWSStore } from '@/stores/wsStore'
import { useWSContext } from '@/providers/WebSocketProvider'
import { LivePulse } from '@/components/ui/LivePulse'
import { formatPrice } from '@/utils/format'

function HamburgerIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="none" className="w-4 h-4" stroke="currentColor" strokeWidth="1.5">
      <path d="M2 4h12M2 8h12M2 12h8"/>
    </svg>
  )
}

function BellIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="none" className="w-4 h-4" stroke="currentColor" strokeWidth="1.5">
      <path d="M8 2a5 5 0 0 0-5 5v3l-1 2h12l-1-2V7a5 5 0 0 0-5-5z"/>
      <path d="M6.5 13.5a1.5 1.5 0 0 0 3 0"/>
    </svg>
  )
}

function UserIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="none" className="w-4 h-4" stroke="currentColor" strokeWidth="1.5">
      <circle cx="8" cy="5" r="3"/>
      <path d="M2 14c0-3.3 2.7-6 6-6s6 2.7 6 6"/>
    </svg>
  )
}

function ChevronDown() {
  return (
    <svg viewBox="0 0 10 10" fill="none" className="w-2.5 h-2.5" stroke="currentColor" strokeWidth="1.5">
      <path d="M2 3.5l3 3 3-3"/>
    </svg>
  )
}

const PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
const TIMEFRAMES = ['M1', 'M5', 'M15', 'H1', 'H4', 'D1']

export function TopBar() {
  const { toggleSidebar, selectedPair, selectedTimeframe, setSelectedPair, setSelectedTimeframe } = useUIStore()
  const { latestSignal } = useWSStore()
  const { connected } = useWSContext()

  const [utcTime, setUtcTime] = useState('')

  useEffect(() => {
    const tick = () => {
      const d = new Date()
      setUtcTime(d.toUTCString().split(' ')[4] + ' UTC')
    }
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [])

  const currentPrice = latestSignal?.close ?? null

  return (
    <header className="h-14 bg-surface border-b border-border flex items-center px-3 gap-3 shrink-0 z-10">

      {/* Hamburger */}
      <button
        onClick={toggleSidebar}
        className="text-secondary hover:text-primary transition-colors p-1.5 rounded hover:bg-navy-700"
        aria-label="Toggle sidebar"
      >
        <HamburgerIcon />
      </button>

      {/* Logo (visible on mobile when sidebar is closed) */}
      <Link href="/" className="flex items-center gap-2 mr-1">
        <div className="w-6 h-6 rounded bg-gold/10 border border-gold/30 flex items-center justify-center">
          <span className="text-gold font-black text-xs">M</span>
        </div>
        <span className="text-primary font-bold text-sm hidden sm:inline">MFIP</span>
      </Link>

      {/* Divider */}
      <div className="h-5 w-px bg-border hidden sm:block" />

      {/* Pair selector */}
      <div className="relative group">
        <button className="flex items-center gap-1.5 text-primary font-semibold text-sm hover:text-gold transition-colors">
          <span>{selectedPair}</span>
          <ChevronDown />
        </button>
        <div className="absolute top-full left-0 mt-1 z-50 bg-elevated border border-border rounded-lg shadow-card-lg py-1 w-28 hidden group-hover:block">
          {PAIRS.map(p => (
            <button
              key={p}
              onClick={() => setSelectedPair(p)}
              className={cn(
                'w-full text-left px-3 py-1.5 text-xs hover:bg-navy-600 transition-colors',
                p === selectedPair ? 'text-gold font-semibold' : 'text-secondary',
              )}
            >
              {p}
            </button>
          ))}
        </div>
      </div>

      {/* Timeframe selector */}
      <div className="relative group hidden sm:block">
        <button className="flex items-center gap-1 text-secondary text-xs hover:text-primary transition-colors bg-navy-700 border border-border rounded px-2 py-1">
          <span>{selectedTimeframe}</span>
          <ChevronDown />
        </button>
        <div className="absolute top-full left-0 mt-1 z-50 bg-elevated border border-border rounded-lg shadow-card-lg py-1 w-20 hidden group-hover:block">
          {TIMEFRAMES.map(tf => (
            <button
              key={tf}
              onClick={() => setSelectedTimeframe(tf)}
              className={cn(
                'w-full text-left px-3 py-1.5 text-xs hover:bg-navy-600 transition-colors',
                tf === selectedTimeframe ? 'text-gold font-semibold' : 'text-secondary',
              )}
            >
              {tf}
            </button>
          ))}
        </div>
      </div>

      {/* Price */}
      {currentPrice != null && (
        <div className="hidden md:flex items-center gap-1.5">
          <span className="num text-primary font-semibold text-sm">{formatPrice(currentPrice)}</span>
          <span className="text-muted text-xs">·</span>
          <span className="text-muted text-xs">{selectedPair}</span>
        </div>
      )}

      {/* Spacer */}
      <div className="flex-1" />

      {/* UTC clock */}
      <span className="num text-muted text-xs hidden lg:block">{utcTime}</span>

      {/* Divider */}
      <div className="h-5 w-px bg-border hidden md:block" />

      {/* Live indicator */}
      <LivePulse connected={connected} showLabel size="sm" />

      {/* Notification */}
      <button className="text-secondary hover:text-primary transition-colors p-1.5 rounded hover:bg-navy-700 relative" aria-label="Notifications">
        <BellIcon />
      </button>

      {/* User */}
      <button className="text-secondary hover:text-primary transition-colors p-1.5 rounded hover:bg-navy-700" aria-label="User profile">
        <UserIcon />
      </button>
    </header>
  )
}
