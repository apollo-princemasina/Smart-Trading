'use client'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/services/api'
import { useWSStore } from '@/stores/wsStore'
import type { DecisionSummary, RegimeSummary, RegimeData, PredictionSummary, EIESummary, MIASummary } from '@/types/api'

export function useDashboard() {
  const { data: snapshot, isLoading, isError, dataUpdatedAt } = useQuery({
    queryKey: ['dashboard'],
    queryFn:  () => api.dashboard.snapshot(),
    refetchInterval: 30_000,
  })

  const {
    latestDecision,
    latestSignal,
    currentRegime,
    miaUpdate,
    eieUpdate,
    systemStatus,
  } = useWSStore()

  return {
    isLoading,
    isError,
    dataUpdatedAt,
    // WS data takes priority over REST snapshot
    decision: (latestDecision ?? snapshot?.decision ?? null) as DecisionSummary | null,
    signal:   (latestSignal ?? snapshot?.latest_prediction ?? null) as PredictionSummary | null,
    regime:   (currentRegime ?? snapshot?.market_regime ?? null) as RegimeData | RegimeSummary | null,
    mia:      (miaUpdate ?? snapshot?.mia_summary ?? null) as MIASummary | null,
    eie:      (eieUpdate ?? snapshot?.eie_summary ?? null) as EIESummary | null,
    buffer:   snapshot?.buffer_status ?? null,
    system:   systemStatus ?? snapshot?.system_summary ?? null,
  }
}

export function useMarketRegime() {
  return useQuery({
    queryKey: ['regime'],
    queryFn:  () => api.market.regime(),
    refetchInterval: 60_000,
    staleTime: 30_000,
  })
}

export function useCandles(tf = 'M15', limit = 300) {
  return useQuery({
    queryKey: ['candles', tf, limit],
    queryFn:  () => api.market.candles(tf, limit),
    refetchInterval: 60_000,
    staleTime: 45_000,
  })
}
