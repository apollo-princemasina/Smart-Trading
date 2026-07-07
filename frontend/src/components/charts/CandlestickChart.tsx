'use client'
import { useEffect, useRef, useCallback } from 'react'
import type { Candle, Signal } from '@/types/api'

interface Props {
  candles:   Candle[]
  signals:   Signal[]
  className?: string
}

export function CandlestickChart({ candles, signals, className }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef     = useRef<import('lightweight-charts').IChartApi | null>(null)
  const seriesRef    = useRef<import('lightweight-charts').ISeriesApi<'Candlestick'> | null>(null)

  // Mirrors props as refs so init() can read them without being in its dep array
  const candlesRef = useRef<Candle[]>(candles)
  const signalsRef = useRef<Signal[]>(signals)
  useEffect(() => { candlesRef.current = candles }, [candles])
  useEffect(() => { signalsRef.current = signals }, [signals])

  const THEME = {
    bg:        '#070C14',
    grid:      '#111D2E',
    border:    '#1E2D40',
    text:      '#7C8EA6',
    upColor:   '#10B981',
    downColor: '#F87171',
  }

  // Build the chart once. Data is applied inside init() so the Strict Mode
  // double-invoke doesn't leave us with a second chart that never gets data.
  const init = useCallback(async () => {
    if (!containerRef.current || chartRef.current) return

    const { createChart, CrosshairMode } = await import('lightweight-charts')

    // Re-check after async yield — Strict Mode may have launched a concurrent init()
    if (!containerRef.current || chartRef.current) return

    const chart = createChart(containerRef.current, {
      layout:  { background: { color: THEME.bg }, textColor: THEME.text, fontSize: 11 },
      grid:    { vertLines: { color: THEME.grid }, horzLines: { color: THEME.grid } },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: THEME.border, width: 1, style: 3 },
        horzLine: { color: THEME.border, width: 1, style: 3 },
      },
      rightPriceScale: { borderColor: THEME.border },
      timeScale: { borderColor: THEME.border, timeVisible: true, secondsVisible: false },
      handleScroll: { mouseWheel: true, pressedMouseMove: true },
      handleScale:  { mouseWheel: true, pinch: true },
    })

    const series = chart.addCandlestickSeries({
      upColor:         THEME.upColor,
      downColor:       THEME.downColor,
      borderUpColor:   THEME.upColor,
      borderDownColor: THEME.downColor,
      wickUpColor:     THEME.upColor,
      wickDownColor:   THEME.downColor,
    })

    chartRef.current  = chart
    seriesRef.current = series

    // Set candles immediately if already loaded — avoids Strict Mode double-invoke race
    const initialCandles = candlesRef.current
    if (initialCandles.length > 0) {
      series.setData(
        initialCandles.map(c => ({
          time:  Math.floor(new Date(c.timestamp).getTime() / 1000) as import('lightweight-charts').UTCTimestamp,
          open:  c.open,
          high:  c.high,
          low:   c.low,
          close: c.close,
        }))
      )
      chart.timeScale().fitContent()
    }

    const ro = new ResizeObserver(() => {
      if (containerRef.current) {
        chart.applyOptions({
          width:  containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        })
      }
    })
    ro.observe(containerRef.current)
    return () => ro.disconnect()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Mount / unmount
  useEffect(() => {
    let cleanup: (() => void) | undefined
    init().then(fn => { cleanup = fn })
    return () => {
      cleanup?.()
      chartRef.current?.remove()
      chartRef.current  = null
      seriesRef.current = null
    }
  }, [init])

  // Update candle data after initial mount when new data arrives
  useEffect(() => {
    if (!seriesRef.current || !candles.length) return
    seriesRef.current.setData(
      candles.map(c => ({
        time:  Math.floor(new Date(c.timestamp).getTime() / 1000) as import('lightweight-charts').UTCTimestamp,
        open:  c.open,
        high:  c.high,
        low:   c.low,
        close: c.close,
      }))
    )
    chartRef.current?.timeScale().fitContent()
  }, [candles])

  // Update signal markers
  useEffect(() => {
    if (!seriesRef.current || !signals.length) return
    const markers = signals
      .filter(s => s.direction !== 'HOLD')
      .map(s => ({
        time:     Math.floor(new Date(s.signal_time).getTime() / 1000) as import('lightweight-charts').UTCTimestamp,
        position: s.direction === 'BUY' ? 'belowBar' : 'aboveBar',
        color:    s.direction === 'BUY' ? '#10B981' : '#F87171',
        shape:    s.direction === 'BUY' ? 'arrowUp' : 'arrowDown',
        text:     `${s.direction} ${Math.round(s.confidence <= 1 ? s.confidence * 100 : s.confidence)}%`,
      })) as import('lightweight-charts').SeriesMarker<import('lightweight-charts').UTCTimestamp>[]
    seriesRef.current.setMarkers(markers)
  }, [signals])

  return (
    <div
      ref={containerRef}
      className={className ?? 'w-full h-full'}
      aria-label="EURUSD candlestick chart"
    />
  )
}
