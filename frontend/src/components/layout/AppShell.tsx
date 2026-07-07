'use client'
import { NavSidebar } from './NavSidebar'
import { TopBar } from './TopBar'
import { ConnectionBanner } from '@/components/ui/LivePulse'
import { useWSContext } from '@/providers/WebSocketProvider'

interface AppShellProps {
  children: React.ReactNode
}

export function AppShell({ children }: AppShellProps) {
  const { connected } = useWSContext()

  return (
    <div className="flex h-screen overflow-hidden bg-bg">
      {/* Left navigation */}
      <NavSidebar />

      {/* Right: header + content */}
      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
        <TopBar />
        <ConnectionBanner connected={connected} />
        <main className="flex-1 overflow-hidden">
          {children}
        </main>
      </div>
    </div>
  )
}
