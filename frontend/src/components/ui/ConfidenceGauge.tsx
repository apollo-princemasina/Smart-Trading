'use client'
import { useMemo } from 'react'
import { cn } from '@/utils/cn'
import { confidenceToPercent } from '@/utils/format'

// ── SVG arc math ───────────────────────────────────────────────────────────────

function polarXY(cx: number, cy: number, r: number, deg: number) {
  const rad = (deg * Math.PI) / 180
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) }
}

function arcPath(cx: number, cy: number, r: number, startDeg: number, endDeg: number): string {
  const start    = polarXY(cx, cy, r, startDeg)
  const end      = polarXY(cx, cy, r, endDeg)
  const sweep    = endDeg - startDeg
  const largeArc = sweep > 180 ? 1 : 0
  return [
    `M${start.x.toFixed(2)},${start.y.toFixed(2)}`,
    `A${r},${r} 0 ${largeArc} 1 ${end.x.toFixed(2)},${end.y.toFixed(2)}`,
  ].join(' ')
}

// ── Constants ─────────────────────────────────────────────────────────────────

const CX = 100, CY = 110, R = 82
const START = 135, SWEEP = 270
const VB = '0 0 200 175'
const SW = 10 // strokeWidth

// ── Gauge color ───────────────────────────────────────────────────────────────

function gaugeColor(pct: number): string {
  if (pct >= 70) return '#10B981' // emerald
  if (pct >= 40) return '#FBBF24' // amber
  return '#F87171'                // red
}

// ── Component ─────────────────────────────────────────────────────────────────

interface ConfidenceGaugeProps {
  value:      number       // 0-100 or 0-1
  size?:      number       // container width in px
  label?:     string
  className?: string
  animate?:   boolean
}

export function ConfidenceGauge({
  value,
  size = 180,
  label = 'CONFIDENCE',
  className,
  animate = true,
}: ConfidenceGaugeProps) {
  const pct      = useMemo(() => confidenceToPercent(value), [value])
  const endAngle = useMemo(() => START + (pct / 100) * SWEEP, [pct])
  const color    = useMemo(() => gaugeColor(pct), [pct])

  const bgPath    = arcPath(CX, CY, R, START, START + SWEEP)
  const activePath = pct > 0 ? arcPath(CX, CY, R, START, endAngle) : null

  return (
    <div className={cn('flex flex-col items-center', className)} style={{ width: size }}>
      <svg
        viewBox={VB}
        width={size}
        height={size * (175 / 200)}
        overflow="visible"
        aria-label={`${label}: ${pct}%`}
        role="img"
      >
        <defs>
          <filter id="glow-gauge">
            <feGaussianBlur stdDeviation="2" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>

        {/* Background track */}
        <path
          d={bgPath}
          fill="none"
          stroke="#1E2D40"
          strokeWidth={SW}
          strokeLinecap="round"
        />

        {/* Active arc */}
        {activePath && (
          <path
            d={activePath}
            fill="none"
            stroke={color}
            strokeWidth={SW}
            strokeLinecap="round"
            filter={pct > 60 ? 'url(#glow-gauge)' : undefined}
            style={animate ? { transition: 'stroke 0.5s ease' } : undefined}
          />
        )}

        {/* Center value */}
        <text
          x={CX}
          y={CY + 4}
          textAnchor="middle"
          dominantBaseline="middle"
          fill={color}
          fontSize="34"
          fontWeight="700"
          fontFamily="var(--font-mono)"
          style={animate ? { transition: 'fill 0.5s ease' } : undefined}
        >
          {pct}
        </text>

        {/* Percent sign */}
        <text
          x={CX + 26}
          y={CY - 4}
          fill={color}
          fontSize="14"
          fontWeight="600"
          opacity="0.8"
        >
          %
        </text>

        {/* Label */}
        <text
          x={CX}
          y={CY + 26}
          textAnchor="middle"
          fill="#4B5D74"
          fontSize="9"
          fontWeight="600"
          letterSpacing="2"
          style={{ textTransform: 'uppercase' as const }}
        >
          {label}
        </text>

        {/* Min / Max ticks */}
        <text x="34" y="170" fill="#2A3F58" fontSize="9" textAnchor="middle">0</text>
        <text x="166" y="170" fill="#2A3F58" fontSize="9" textAnchor="middle">100</text>
      </svg>
    </div>
  )
}
