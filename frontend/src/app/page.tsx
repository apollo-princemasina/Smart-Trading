'use client'
import { AppShell } from '@/components/layout/AppShell'
import { AIMarketSummary } from '@/components/dashboard/AIMarketSummary'
import { MarketContextPanel } from '@/components/dashboard/MarketContextPanel'
import { ChartDecisionPanel } from '@/components/dashboard/ChartDecisionPanel'
import { DecisionPanel } from '@/components/dashboard/DecisionPanel'
import { DecisionTimeline } from '@/components/dashboard/DecisionTimeline'

export default function DashboardPage() {
  return (
    <AppShell>
      <div className="flex flex-col h-full overflow-hidden">

        {/* AI Market Summary — always visible, collapsible */}
        <AIMarketSummary />

        {/* 3-column main grid */}
        <div className="flex flex-1 min-h-0 overflow-hidden">

          {/* Left — Market Context */}
          <div className="hidden md:flex w-56 xl:w-60 shrink-0 border-r border-border overflow-hidden">
            <MarketContextPanel />
          </div>

          {/* Center — Chart + Decision Metrics */}
          <div className="flex-1 min-w-0 overflow-hidden">
            <ChartDecisionPanel />
          </div>

          {/* Right — Decision Intelligence (command center) */}
          <div className="hidden lg:flex w-72 xl:w-80 shrink-0 border-l border-border overflow-hidden">
            <DecisionPanel />
          </div>
        </div>

        {/* Bottom — Decision Timeline */}
        <div className="shrink-0 hidden sm:block max-h-48 overflow-hidden">
          <DecisionTimeline />
        </div>
      </div>
    </AppShell>
  )
}
