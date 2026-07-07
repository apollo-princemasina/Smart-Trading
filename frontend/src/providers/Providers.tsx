'use client'
import { QueryProvider } from './QueryProvider'
import { WebSocketProvider } from './WebSocketProvider'

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <QueryProvider>
      <WebSocketProvider>
        {children}
      </WebSocketProvider>
    </QueryProvider>
  )
}
