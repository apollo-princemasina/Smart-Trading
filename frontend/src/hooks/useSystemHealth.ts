'use client'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/services/api'

export function useSystemHealth() {
  return useQuery({
    queryKey:       ['system-health'],
    queryFn:        () => api.system.health(),
    refetchInterval: 15_000,
    staleTime:       10_000,
  })
}

export function useSystemVersion() {
  return useQuery({
    queryKey: ['system-version'],
    queryFn:  () => api.system.version(),
    staleTime: 60_000,
  })
}

export function useModels() {
  return useQuery({
    queryKey: ['models'],
    queryFn:  () => api.models.list(),
    staleTime: 60_000,
  })
}

export function useActiveModel() {
  return useQuery({
    queryKey: ['active-model'],
    queryFn:  () => api.models.active(),
    staleTime: 60_000,
  })
}
