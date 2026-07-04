'use client'
import { useEffect, useRef } from 'react'
import type { Candle } from '@/types'

interface Props { candles: Candle[] }

export default function ChartPanel({ candles }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const chartRef = useRef<import('lightweight-charts').IChartApi | null>(null)
  const seriesRef = useRef<import('lightweight-charts').ISeriesApi<'Candlestick'> | null>(null)

  useEffect(() => {
    if (!ref.current) return

    import('lightweight-charts').then(({ createChart, CrosshairMode }) => {
      if (chartRef.current) return // already created

      const chart = createChart(ref.current!, {
        autoSize: true,
        layout: {
          background: { color: '#0D1117' },
          textColor:  '#6B7280',
        },
        grid: {
          vertLines: { color: '#21262D' },
          horzLines: { color: '#21262D' },
        },
        crosshair: { mode: CrosshairMode.Normal },
        rightPriceScale: { borderColor: '#21262D' },
        timeScale: {
          borderColor: '#21262D',
          timeVisible: true,
          secondsVisible: false,
        },
      })

      const series = chart.addCandlestickSeries({
        upColor:          '#10B981',
        downColor:        '#EF4444',
        borderUpColor:    '#10B981',
        borderDownColor:  '#EF4444',
        wickUpColor:      '#10B981',
        wickDownColor:    '#EF4444',
      })

      chartRef.current = chart
      seriesRef.current = series
    })
  }, [])

  // Feed candle data — retry until series is ready (dynamic import may lag data fetch)
  useEffect(() => {
    if (candles.length === 0) return
    const data = candles.map((c) => ({
      time:  Math.floor(new Date(c.timestamp).getTime() / 1000) as import('lightweight-charts').UTCTimestamp,
      open:  c.open,
      high:  c.high,
      low:   c.low,
      close: c.close,
    }))
    const trySet = () => {
      if (seriesRef.current) {
        seriesRef.current.setData(data)
        chartRef.current?.timeScale().fitContent()
      } else {
        setTimeout(trySet, 100)
      }
    }
    trySet()
  }, [candles])

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-2 border-b border-border text-xs text-muted">
        EURUSD · M15 · Candlestick
      </div>
      {/* Always render ref div so chart can mount; overlay loading text when no data */}
      <div className="relative flex-1">
        <div ref={ref} className="absolute inset-0" />
        {candles.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center text-muted text-xs pointer-events-none">
            Loading chart data…
          </div>
        )}
      </div>
    </div>
  )
}
