'use client'
import { useEffect, useRef, useState, useCallback } from 'react'
import type { WSMessage } from '@/types'

const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? 'ws://localhost:8000/ws'

export function useWebSocket(onMessage: (msg: WSMessage) => void) {
  const wsRef     = useRef<WebSocket | null>(null)
  const retryRef  = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [connected, setConnected] = useState(false)

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen    = () => { setConnected(true) }
    ws.onclose   = () => {
      setConnected(false)
      // Reconnect after 3 s
      retryRef.current = setTimeout(connect, 3000)
    }
    ws.onerror   = () => ws.close()
    ws.onmessage = (e) => {
      try { onMessage(JSON.parse(e.data) as WSMessage) }
      catch { /* ignore malformed frames */ }
    }
  }, [onMessage])

  useEffect(() => {
    connect()
    return () => {
      if (retryRef.current) clearTimeout(retryRef.current)
      wsRef.current?.close()
    }
  }, [connect])

  return connected
}
