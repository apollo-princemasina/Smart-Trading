'use client'
import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { AppShell } from '@/components/layout/AppShell'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Skeleton } from '@/components/ui/Skeleton'
import { api } from '@/services/api'
import { cn } from '@/utils/cn'
import type { SettingOut } from '@/types/api'

function SettingRow({ setting, onUpdate }: { setting: SettingOut; onUpdate: (key: string, value: string) => void }) {
  const [editing, setEditing] = useState(false)
  const [draft,   setDraft]   = useState(String(setting.value ?? ''))

  const handleSave = () => {
    onUpdate(setting.key, draft)
    setEditing(false)
  }

  return (
    <div className="flex items-start gap-4 py-3 border-b border-border/60 last:border-0">
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-0.5">
          <span className="text-primary text-sm font-medium">{setting.key}</span>
          <Badge variant="muted" size="xs">{setting.value_type}</Badge>
          {setting.is_secret && <Badge variant="warning" size="xs">secret</Badge>}
        </div>
        {setting.description && (
          <p className="text-muted text-xs">{setting.description}</p>
        )}
      </div>

      <div className="flex items-center gap-2 shrink-0">
        {editing ? (
          <>
            <input
              value={draft}
              onChange={e => setDraft(e.target.value)}
              className="bg-navy-800 border border-border rounded px-2 py-1 text-xs text-primary w-40 focus:outline-none focus:border-gold/50"
              autoFocus
              onKeyDown={e => { if (e.key === 'Enter') handleSave(); if (e.key === 'Escape') setEditing(false) }}
            />
            <button
              onClick={handleSave}
              className="px-2 py-1 text-xs bg-buy/10 border border-buy-dim text-buy rounded hover:bg-buy/20"
            >
              Save
            </button>
            <button
              onClick={() => { setEditing(false); setDraft(String(setting.value ?? '')) }}
              className="px-2 py-1 text-xs border border-border text-secondary rounded hover:text-primary"
            >
              Cancel
            </button>
          </>
        ) : (
          <>
            <span className={cn('num text-secondary text-xs', setting.is_secret && 'blur-sm')}>
              {setting.is_secret ? '••••••' : String(setting.value ?? '—')}
            </span>
            <button
              onClick={() => setEditing(true)}
              className="px-2 py-1 text-xs border border-border text-secondary rounded hover:text-primary hover:border-border/80"
            >
              Edit
            </button>
          </>
        )}
      </div>
    </div>
  )
}

export default function SettingsPage() {
  const queryClient = useQueryClient()

  const { data, isLoading } = useQuery({
    queryKey: ['settings'],
    queryFn:  () => api.settings.list(),
  })

  const updateMutation = useMutation({
    mutationFn: ({ key, value }: { key: string; value: unknown }) =>
      api.settings.update(key, value),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['settings'] }),
  })

  const settings  = data?.settings ?? []
  const byCategory = settings.reduce<Record<string, SettingOut[]>>((acc, s) => {
    const cat = s.category ?? 'general'
    if (!acc[cat]) acc[cat] = []
    acc[cat].push(s)
    return acc
  }, {})

  return (
    <AppShell>
      <div className="h-full flex flex-col overflow-hidden">
        <div className="px-6 py-4 border-b border-border shrink-0">
          <h1 className="text-primary font-bold text-lg">Settings</h1>
          <p className="text-secondary text-sm mt-0.5">Runtime-configurable settings — changes take effect immediately, no restart required</p>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {isLoading ? (
            <div className="space-y-4">
              {Array.from({ length: 3 }).map((_, i) => (
                <Card key={i} title="Loading...">
                  <div className="space-y-3">
                    <Skeleton rows={3} />
                  </div>
                </Card>
              ))}
            </div>
          ) : !Object.keys(byCategory).length ? (
            <div className="text-center py-16">
              <p className="text-secondary text-sm">No settings configured</p>
              <p className="text-muted text-xs mt-1">Settings are created via the API or on first startup</p>
            </div>
          ) : (
            Object.entries(byCategory).map(([category, items]) => (
              <Card key={category} title={category.toUpperCase()} subtitle={`${items.length} setting${items.length > 1 ? 's' : ''}`}>
                <div>
                  {items.map(setting => (
                    <SettingRow
                      key={setting.key}
                      setting={setting}
                      onUpdate={(key, value) => updateMutation.mutate({ key, value })}
                    />
                  ))}
                </div>
              </Card>
            ))
          )}

          {/* Notes */}
          <Card variant="bordered" className="opacity-60">
            <p className="text-secondary text-xs leading-relaxed">
              Settings are stored in the application database. Changes apply to the next engine cycle.
              Secret values are stored encrypted and cannot be read back after being set.
              Changes here do not restart any engine processes.
            </p>
          </Card>
        </div>
      </div>
    </AppShell>
  )
}
