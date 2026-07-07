'use client'
import { useDashboard } from '@/hooks/useDashboard'
import { useActiveModel } from '@/hooks/useSystemHealth'
import { ConfidenceGauge } from '@/components/ui/ConfidenceGauge'
import { AgreementGauge } from '@/components/ui/AgreementGauge'
import { Badge } from '@/components/ui/Badge'
import { Skeleton, SkeletonGauge } from '@/components/ui/Skeleton'
import { Separator } from '@/components/ui/Separator'
import { ProgressBar } from '@/components/ui/ProgressBar'
import { cn } from '@/utils/cn'
import { consensusToLabel, formatRelative } from '@/utils/format'
import type { Recommendation } from '@/types/api'

// ── Recommendation hero badge ──────────────────────────────────────────────────

const REC_CONFIG: Record<Recommendation, { color: string; bg: string; border: string; glow: string; label: string }> = {
  BUY:  { color: 'text-buy',  bg: 'bg-buy-bg',  border: 'border-buy-dim',  glow: 'glow-buy',  label: '▲ BUY' },
  SELL: { color: 'text-sell', bg: 'bg-sell-bg', border: 'border-sell-dim', glow: 'glow-sell', label: '▼ SELL' },
  WAIT: { color: 'text-wait', bg: 'bg-wait-bg', border: 'border-wait-dim', glow: '',           label: '— WAIT' },
}

function RecommendationHero({ rec, strength }: { rec: Recommendation; strength: string }) {
  const cfg = REC_CONFIG[rec]
  return (
    <div className={cn(
      'rounded-lg border px-4 py-3 text-center transition-all duration-500',
      cfg.bg, cfg.border, cfg.glow,
    )}>
      <p className={cn('text-2xl font-black tracking-widest', cfg.color)}>{cfg.label}</p>
      <p className={cn('text-xs font-semibold tracking-widest mt-0.5 uppercase opacity-75', cfg.color)}>
        {strength}
      </p>
    </div>
  )
}

// ── Reason list ───────────────────────────────────────────────────────────────

function ReasonList({
  items,
  variant,
}: {
  items: string[] | undefined
  variant: 'positive' | 'negative' | 'warning'
}) {
  if (!items?.length) return null

  const icon  = variant === 'positive' ? '✓' : variant === 'negative' ? '✗' : '⚠'
  const color =
    variant === 'positive' ? 'text-buy' :
    variant === 'negative' ? 'text-sell' :
    'text-wait'

  return (
    <ul className="space-y-1.5">
      {items.slice(0, 4).map((item, i) => (
        <li key={i} className="flex items-start gap-2">
          <span className={cn('shrink-0 text-xs mt-0.5 font-bold', color)}>{icon}</span>
          <span className="text-secondary text-xs leading-relaxed">{item}</span>
        </li>
      ))}
    </ul>
  )
}

// ── Main component ─────────────────────────────────────────────────────────────

export function DecisionPanel() {
  const { decision, isLoading } = useDashboard()
  const { data: modelResp }     = useActiveModel()
  const model = modelResp?.model

  const panelHeader = (
    <div className="px-4 py-2 border-b border-border sticky top-0 bg-surface z-10">
      <p className="label">DECISION INTELLIGENCE</p>
    </div>
  )

  if (isLoading) {
    return (
      <aside className="h-full bg-surface overflow-y-auto flex flex-col">
        {panelHeader}
        <div className="flex-1 p-4 space-y-4">
          <SkeletonGauge size={140} />
          <Skeleton variant="block" className="h-16" />
          <Skeleton rows={4} />
        </div>
      </aside>
    )
  }

  if (!decision) {
    return (
      <aside className="h-full bg-surface overflow-y-auto flex flex-col">
        {panelHeader}
        <div className="flex-1 flex items-center justify-center p-6">
          <div className="text-center">
            <div className="w-12 h-12 rounded-full bg-navy-700 border border-border flex items-center justify-center mx-auto mb-3">
              <span className="text-muted text-lg">⊘</span>
            </div>
            <p className="text-secondary text-sm">No decision yet</p>
            <p className="text-muted text-xs mt-1">Waiting for first scheduler tick</p>
          </div>
        </div>
      </aside>
    )
  }

  const confPct = decision.confidence <= 1
    ? Math.round(decision.confidence * 100)
    : Math.round(decision.confidence)

  return (
    <aside className="h-full bg-surface overflow-y-auto flex flex-col">
      {panelHeader}

      <div className="flex-1 overflow-y-auto">

        {/* Confidence gauge */}
        <div className="px-4 pt-4 pb-2 flex justify-center">
          <ConfidenceGauge value={confPct} size={200} animate />
        </div>

        {/* Recommendation hero */}
        <div className="px-4 pb-3">
          <RecommendationHero
            rec={decision.recommendation}
            strength={decision.strength}
          />
        </div>

        {/* Agreement / conflict */}
        <div className="px-4 pb-3">
          <AgreementGauge
            agreement={decision.agreement_score}
            conflict={decision.conflict_score}
          />
        </div>

        <Separator />

        {/* Primary reasons */}
        {decision.primary_reasons?.length ? (
          <div className="px-4 py-3">
            <span className="label block mb-2">PRIMARY REASONS</span>
            <ReasonList items={decision.primary_reasons} variant="positive" />
          </div>
        ) : null}

        {/* Conflicting factors */}
        {decision.conflicting_factors?.length ? (
          <>
            <Separator />
            <div className="px-4 py-3">
              <span className="label block mb-2">CONFLICTING FACTORS</span>
              <ReasonList items={decision.conflicting_factors} variant="negative" />
            </div>
          </>
        ) : null}

        {/* Risk factors */}
        {decision.risk_factors?.length ? (
          <>
            <Separator />
            <div className="px-4 py-3">
              <span className="label block mb-2">RISK FACTORS</span>
              <ReasonList items={decision.risk_factors} variant="warning" />
            </div>
          </>
        ) : null}

        <Separator />

        {/* Execution readiness */}
        <div className="px-4 py-3">
          <span className="label block mb-2">EXECUTION READINESS</span>
          <ProgressBar value={confPct} height="thick" showValue />
          <div className="flex items-center justify-between mt-1.5">
            <span className="text-muted text-xs">
              {consensusToLabel(decision.consensus_level)} consensus
            </span>
            {decision.expiry_minutes != null && (
              <span className="num text-secondary text-xs">
                Expires ~{decision.expiry_minutes}m
              </span>
            )}
          </div>
        </div>

        <Separator />

        {/* Signal alignment */}
        <div className="px-4 py-3 space-y-1">
          <span className="label block mb-2">SIGNAL ALIGNMENT</span>
          {[
            ['Technical', decision.technical_alignment],
            ['Fundamental', decision.fundamental_alignment],
            ['Bias', decision.market_bias],
          ].map(([k, v]) => (
            <div key={k} className="flex items-center justify-between">
              <span className="text-muted text-xs">{k}</span>
              <span className="text-secondary text-xs font-medium">{v ?? '—'}</span>
            </div>
          ))}
          <div className="flex items-center gap-2 pt-1">
            {decision.has_ml  && <Badge variant="blue" size="xs">ML</Badge>}
            {decision.has_eie && <Badge variant="gold" size="xs">EIE</Badge>}
            {decision.has_mia && <Badge variant="muted" size="xs">MIA</Badge>}
          </div>
        </div>

        {/* Model info */}
        {model && (
          <>
            <Separator />
            <div className="px-4 py-3">
              <span className="label block mb-2">MODEL</span>
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-muted text-xs">Version</span>
                  <span className="num text-secondary text-xs">{model.model_version}</span>
                </div>
                {model.git_commit && (
                  <div className="flex items-center justify-between">
                    <span className="text-muted text-xs">Commit</span>
                    <span className="num text-secondary text-xs font-mono">{model.git_commit.slice(0, 7)}</span>
                  </div>
                )}
                {model.f1_buy != null && (
                  <div className="flex items-center justify-between">
                    <span className="text-muted text-xs">F1 (BUY)</span>
                    <span className="num text-buy text-xs">{(model.f1_buy * 100).toFixed(1)}%</span>
                  </div>
                )}
                {model.f1_sell != null && (
                  <div className="flex items-center justify-between">
                    <span className="text-muted text-xs">F1 (SELL)</span>
                    <span className="num text-sell text-xs">{(model.f1_sell * 100).toFixed(1)}%</span>
                  </div>
                )}
              </div>
            </div>
          </>
        )}

        {/* Schema version */}
        {decision.schema_version && (
          <div className="px-4 pb-4">
            <span className="num text-muted text-[10px]">Schema: {decision.schema_version}</span>
          </div>
        )}
      </div>
    </aside>
  )
}
