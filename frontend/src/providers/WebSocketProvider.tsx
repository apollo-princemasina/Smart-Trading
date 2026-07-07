'use client'
import {
  createContext,
  useContext,
  useEffect,
  useRef,
  useCallback,
  useState,
} from 'react'
import type { WSMessage } from '@/types/ws'
import { useWSStore } from '@/stores/wsStore'

const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? 'ws://localhost:8000/ws'
const RECONNECT_DELAY = 3_000

interface WSContextValue {
  connected:  boolean
  send:       (data: unknown) => void
  subscribe:  (events: string[]) => void
}

const WSContext = createContext<WSContextValue>({
  connected: false,
  send: () => {},
  subscribe: () => {},
})

export function WebSocketProvider({ children }: { children: React.ReactNode }) {
  const wsRef         = useRef<WebSocket | null>(null)
  const retryRef      = useRef<ReturnType<typeof setTimeout> | null>(null)
  const mountedRef    = useRef(true)
  const [connected, setConnected] = useState(false)

  const { setConnected: storeSetConnected, handleWSMessage, incrementReconnect } = useWSStore()

  const connect = useCallback(() => {
    if (!mountedRef.current) return
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      if (!mountedRef.current) return
      setConnected(true)
      storeSetConnected(true)
      // Subscribe to all events
      ws.send(JSON.stringify({ type: 'subscribe', events: [
        'decision_update', 'signal_update', 'regime_update',
        'mia_update', 'eie_update', 'system_status', 'scheduler_tick',
        'model_loaded', 'health_update',
      ]}))
    }

    ws.onclose = () => {
      if (!mountedRef.current) return
      setConnected(false)
      storeSetConnected(false)
      incrementReconnect()
      retryRef.current = setTimeout(connect, RECONNECT_DELAY)
    }

    ws.onerror = () => ws.close()

    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data) as WSMessage
        handleWSMessage(msg)
      } catch {
        // ignore malformed frames
      }
    }
  }, [storeSetConnected, handleWSMessage, incrementReconnect])

  useEffect(() => {
    mountedRef.current = true
    connect()
    const ping = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'ping' }))
      }
    }, 30_000)
    return () => {
      mountedRef.current = false
      clearInterval(ping)
      if (retryRef.current) clearTimeout(retryRef.current)
      wsRef.current?.close()
    }
  }, [connect])

  const send = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data))
    }
  }, [])

  const subscribe = useCallback((events: string[]) => {
    send({ type: 'subscribe', events })
  }, [send])

  return (
    <WSContext.Provider value={{ connected, send, subscribe }}>
      {children}
    </WSContext.Provider>
  )
}

export function useWSContext() {
  return useContext(WSContext)
}
