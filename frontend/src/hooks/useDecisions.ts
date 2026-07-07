'use client'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/services/api'

export function useDecisionHistory(params?: {
  recommendation?: string
  strength?:       string
  after?:          string
  before?:         string
  page?:           number
  page_size?:      number
}) {
  return useQuery({
    queryKey: ['decision-history', params],
    queryFn:  () => api.history.decisions(params),
    staleTime: 10_000,
  })
}

export function usePredictionHistory(params?: {
  symbol?:    string
  direction?: string
  page?:      number
  page_size?: number
}) {
  return useQuery({
    queryKey: ['prediction-history', params],
    queryFn:  () => api.history.predictions(params),
    staleTime: 10_000,
  })
}
