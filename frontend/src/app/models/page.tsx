'use client'
import { AppShell } from '@/components/layout/AppShell'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Skeleton } from '@/components/ui/Skeleton'
import { Separator } from '@/components/ui/Separator'
import { useModels, useActiveModel } from '@/hooks/useSystemHealth'
import { formatDateTime } from '@/utils/format'
import { cn } from '@/utils/cn'
import type { ModelRegistryItem } from '@/types/api'

function ModelCard({ model, isActive }: { model: ModelRegistryItem; isActive: boolean }) {
  return (
    <Card
      className={cn(isActive && 'ring-1 ring-gold/30')}
      title={`${model.model_name} v${model.model_version}`}
      action={
        isActive
          ? <Badge variant="gold" size="sm" dot>ACTIVE</Badge>
          : <Badge variant="muted" size="xs">Inactive</Badge>
      }
    >
      <div className="space-y-3">
        {/* Bundle path */}
        <div className="flex items-start justify-between gap-4">
          <span className="text-muted text-xs">Bundle</span>
          <span className="num text-secondary text-xs text-right truncate max-w-[200px]">{model.bundle_path}</span>
        </div>

        {/* Version info */}
        {[
          ['Git Commit',   model.git_commit?.slice(0, 7)],
          ['Feature Schema', model.feature_schema_version],
          ['Label Version', model.label_version],
          ['Pipeline',     model.pipeline_version],
          ['Feature Count', model.feature_count?.toString()],
        ].filter(([, v]) => v).map(([k, v]) => (
          <div key={k} className="flex items-center justify-between">
            <span className="text-muted text-xs">{k}</span>
            <span className="num text-secondary text-xs font-mono">{v}</span>
          </div>
        ))}

        {/* Metrics */}
        {(model.f1_buy != null || model.f1_sell != null) && (
          <>
            <Separator />
            <div className="grid grid-cols-2 gap-3">
              {[
                { label: 'F1 BUY',       value: model.f1_buy,        color: 'text-buy' },
                { label: 'F1 SELL',      value: model.f1_sell,       color: 'text-sell' },
                { label: 'Prec BUY',     value: model.precision_buy, color: 'text-buy' },
                { label: 'Prec SELL',    value: model.precision_sell,color: 'text-sell' },
              ].filter(r => r.value != null).map(row => (
                <div key={row.label} className="bg-navy-800 rounded p-2">
                  <p className="label">{row.label}</p>
                  <p className={cn('num font-bold text-sm mt-0.5', row.color)}>
                    {((row.value ?? 0) * 100).toFixed(1)}%
                  </p>
                </div>
              ))}
            </div>
          </>
        )}

        {/* Dates */}
        <Separator />
        <div className="flex items-center justify-between">
          <span className="text-muted text-xs">Registered</span>
          <span className="num text-secondary text-xs">{formatDateTime(model.created_at)}</span>
        </div>
        {model.activated_at && (
          <div className="flex items-center justify-between">
            <span className="text-muted text-xs">Activated</span>
            <span className="num text-secondary text-xs">{formatDateTime(model.activated_at)}</span>
          </div>
        )}

        {model.notes && (
          <p className="text-secondary text-xs italic border-t border-border/60 pt-2">{model.notes}</p>
        )}
      </div>
    </Card>
  )
}

export default function ModelsPage() {
  const { data: listData, isLoading }   = useModels()
  const { data: activeData }            = useActiveModel()

  const models      = listData?.models ?? []
  const activeModel = activeData?.model

  return (
    <AppShell>
      <div className="h-full flex flex-col overflow-hidden">
        <div className="px-6 py-4 border-b border-border shrink-0">
          <h1 className="text-primary font-bold text-lg">Model Registry</h1>
          <p className="text-secondary text-sm mt-0.5">
            All registered ML model versions with performance metrics and governance metadata
          </p>
        </div>

        <div className="flex-1 overflow-y-auto p-6">
          {isLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              {Array.from({ length: 3 }).map((_, i) => (
                <div key={i} className="bg-surface border border-border rounded-lg p-4 space-y-3">
                  <Skeleton className="h-4 w-1/2" />
                  <Skeleton rows={4} />
                </div>
              ))}
            </div>
          ) : !models.length ? (
            <div className="text-center py-16">
              <p className="text-secondary text-sm">No models registered yet</p>
              <p className="text-muted text-xs mt-1">Models are auto-registered on startup from the PipelineManager bundle</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              {models.map(model => (
                <ModelCard
                  key={model.id}
                  model={model}
                  isActive={model.id === activeModel?.id}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </AppShell>
  )
}
