'use client'
import { create } from 'zustand'

type NavSection = 'dashboard' | 'decisions' | 'predictions' | 'models' | 'system' | 'settings'

interface UIState {
  sidebarOpen:       boolean
  selectedPair:      string
  selectedTimeframe: string
  activeNav:         NavSection
  summaryExpanded:   boolean

  toggleSidebar:        () => void
  setSidebarOpen:       (v: boolean) => void
  setSelectedPair:      (v: string) => void
  setSelectedTimeframe: (v: string) => void
  setActiveNav:         (v: NavSection) => void
  toggleSummary:        () => void
}

export const useUIStore = create<UIState>((set) => ({
  sidebarOpen:       true,
  selectedPair:      'EURUSD',
  selectedTimeframe: 'M15',
  activeNav:         'dashboard',
  summaryExpanded:   true,

  toggleSidebar:        () => set(s => ({ sidebarOpen: !s.sidebarOpen })),
  setSidebarOpen:       (v) => set({ sidebarOpen: v }),
  setSelectedPair:      (v) => set({ selectedPair: v }),
  setSelectedTimeframe: (v) => set({ selectedTimeframe: v }),
  setActiveNav:         (v) => set({ activeNav: v }),
  toggleSummary:        () => set(s => ({ summaryExpanded: !s.summaryExpanded })),
}))
