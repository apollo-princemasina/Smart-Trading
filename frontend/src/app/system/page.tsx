'use client'
import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { AppShell } from '@/components/layout/AppShell'
import { Card } from '@/components/ui/Card'
import { Badge, StatusBadge } from '@/components/ui/Badge'
import { Skeleton } from '@/components/ui/Skeleton'
import { Separator } from '@/components/ui/Separator'
import { useSystemHealth, useSystemVersion } from '@/hooks/useSystemHealth'
import { api } from '@/services/api'
import { formatUptime, formatDateTime } from '@/utils/format'
import { cn } from '@/utils/cn'
import type { ComponentHealth, ComponentStatus } from '@/types/api'

const COMPONENT_LABELS: Record<string, string> = {
  rolling_buffer:         'Rolling Buffer',
  ml_pipeline:            'ML Pipeline',
  scheduler:              'Scheduler',
  forex_factory:          'Forex Factory',
  economic_intelligence:  'Economic Intelligence',
  market_intelligence_ai: 'Market Intelligence AI',
  decision_fusion:        'Decision Fusion',
  websocket:              'WebSocket',
}

function ComponentCard({ name, health }: { name: string; health: ComponentHealth }) {
  const label = COMPONENT_LABELS[name] ?? name.replace(/_/g, ' ')
  const details = Object.entries(health)
    .filter(([k, v]) => k !== 'status' && typeof v !== 'object')
    .slice(0, 4)

  const borderColor =
    (health.status === 'ok' || health.status === 'operational') ? 'border-buy-dim' :
    health.status === 'degraded' ? 'border-wait-dim' :
    health.status === 'error'    ? 'border-sell-dim' :
    'border-border'

  return (
    <div className={cn('bg-surface border rounded-lg p-4 space-y-3', borderColor)}>
      <div className="flex items-center justify-between">
        <p className="text-primary text-sm font-semibold">{label}</p>
        <StatusBadge status={health.status as ComponentStatus} />
      </div>
      {details.length > 0 && (
        <div className="space-y-1.5">
          {details.map(([k, v]) => (
            <div key={k} className="flex items-center justify-between">
              <span className="text-muted text-xs">{k.replace(/_/g, ' ')}</span>
              <span className="num text-secondary text-xs">
                {typeof v === 'boolean' ? (v ? 'yes' : 'no') : String(v ?? '—')}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default function SystemPage() {
  const { data: health,  isLoading: healthLoading  } = useSystemHealth()
  const { data: version, isLoading: versionLoading } = useSystemVersion()
  const [logLevel, setLogLevel] = useState('')

  const { data: logsData, isLoading: logsLoading } = useQuery({
    queryKey:        ['system-logs', logLevel],
    queryFn:         () => api.system.logs({ level: logLevel || undefined, limit: 50 }),
    refetchInterval: 30_000,
  })

  return (
    <AppShell>
      <div className="h-full flex flex-col overflow-hidden">
        <div className="px-6 py-4 border-b border-border shrink-0">
          <h1 className="text-primary font-bold text-lg">System Health</h1>
          <p className="text-secondary text-sm mt-0.5">Real-time status of all MFIP engine components</p>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-6">

          {/* Overview */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {[
              {
                label: 'System Status',
                value: health
                  ? <StatusBadge status={health.status} />
                  : <Skeleton className="h-5 w-20" />,
              },
              {
                label: 'Uptime',
                value: health
                  ? <span className="num text-primary text-sm font-semibold">{formatUptime(health.uptime_seconds)}</span>
                  : <Skeleton className="h-4 w-16" />,
              },
              {
                label: 'Components OK',
                value: health
                  ? <span className="num text-buy text-sm font-semibold">
                      {Object.values(health.components).filter(c => c.status === 'ok' || c.status === 'operational').length}
                      <span className="text-muted">/{Object.keys(health.components).length}</span>
                    </span>
                  : <Skeleton className="h-4 w-12" />,
              },
              {
                label: 'API Version',
                value: version
                  ? <span className="num text-primary text-sm font-semibold">{version.api_version}</span>
                  : <Skeleton className="h-4 w-16" />,
              },
            ].map(item => (
              <Card key={item.label} className="text-center">
                <p className="label mb-2">{item.label}</p>
                <div>{item.value}</div>
              </Card>
            ))}
          </div>

          {/* Components grid */}
          <div>
            <h2 className="text-primary font-semibold text-sm mb-3">Components</h2>
            {healthLoading ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                {Array.from({ length: 8 }).map((_, i) => (
                  <div key={i} className="bg-surface border border-border rounded-lg p-4 space-y-2">
                    <Skeleton className="h-4 w-1/2" />
                    <Skeleton rows={2} />
                  </div>
                ))}
              </div>
            ) : health ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                {Object.entries(health.components).map(([name, comp]) => (
                  <ComponentCard key={name} name={name} health={comp} />
                ))}
              </div>
            ) : (
              <p className="text-secondary text-sm">Failed to load component health</p>
            )}
          </div>

          {/* Version info */}
          {version && (
            <Card title="Version Information">
              <div className="grid grid-cols-2 gap-3">
                {[
                  ['API Version',    version.api_version],
                  ['Schema Version', version.schema_version],
                  ['Active Model',   version.active_model_id ?? 'None'],
                ].map(([k, v]) => (
                  <div key={k} className="flex items-center justify-between">
                    <span className="text-muted text-xs">{k}</span>
                    <span className="num text-secondary text-xs font-mono">{v}</span>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {/* System logs */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-primary font-semibold text-sm">System Logs</h2>
              <div className="flex gap-2">
                {['', 'INFO', 'WARNING', 'ERROR'].map(level => (
                  <button
                    key={level}
                    onClick={() => setLogLevel(level)}
                    className={cn(
                      'px-3 py-1 text-xs rounded border transition-all',
                      logLevel === level
                        ? 'bg-gold/10 border-gold/50 text-gold font-semibold'
                        : 'border-border text-secondary hover:text-primary',
                    )}
                  >
                    {level || 'All'}
                  </button>
                ))}
              </div>
            </div>

            <Card noPadding>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-border">
                      {['Time', 'Level', 'Component', 'Event', 'Message'].map(h => (
                        <th key={h} className="text-left py-2.5 px-4 label">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {logsLoading ? (
                      Array.from({ length: 8 }).map((_, i) => (
                        <tr key={i} className="border-b border-border/60">
                          {Array.from({ length: 5 }).map((_, j) => (
                            <td key={j} className="py-2.5 px-4"><Skeleton className="h-3 w-16" /></td>
                          ))}
                        </tr>
                      ))
                    ) : !logsData?.logs.length ? (
                      <tr><td colSpan={5} className="text-center py-8 text-muted">No logs</td></tr>
                    ) : (
                      logsData.logs.map(log => (
                        <tr key={log.id} className="border-b border-border/60 hover:bg-navy-700/20">
                          <td className="py-2 px-4 num text-muted text-[11px]">{formatDateTime(log.created_at)}</td>
                          <td className="py-2 px-4">
                            <Badge
                              variant={log.level === 'ERROR' ? 'danger' : log.level === 'WARNING' ? 'warning' : 'muted'}
                              size="xs"
                            >
                              {log.level}
                            </Badge>
                          </td>
                          <td className="py-2 px-4 text-secondary text-[11px]">{log.component}</td>
                          <td className="py-2 px-4 num text-muted text-[11px]">{log.event_type}</td>
                          <td className="py-2 px-4 text-secondary max-w-xs truncate">{log.message}</td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </AppShell>
  )
}
