import { create } from 'zustand'
import { subscribeWithSelector } from 'zustand/middleware'
import { WorkspaceLayout, PanelConfig, PanelState, PanelContextData } from '@/types/panel'

interface PanelsState {
  currentLayout: WorkspaceLayout | null
  panelStates: Record<string, PanelState>
  panelContexts: Record<string, PanelContextData>
  savedLayouts: WorkspaceLayout[]
  
  // Actions
  setCurrentLayout: (layout: WorkspaceLayout) => void
  savePanelState: (panelId: string, state: Partial<PanelState>) => void
  updatePanelContext: (panelId: string, context: Partial<PanelContextData>) => void
  createCustomLayout: (name: string, description: string, panels: PanelConfig[]) => void
  updateLayout: (layoutId: string, updates: Partial<WorkspaceLayout>) => void
  deleteLayout: (layoutId: string) => void
  setPanelSize: (panelId: string, size: number) => void
  togglePanelVisibility: (panelId: string) => void
  pinPanel: (panelId: string) => void
  groupPanels: (panelIds: string[], groupId: string) => void
  ungroupPanel: (panelId: string) => void
}

// Default layouts for different study modes
const defaultLayouts: WorkspaceLayout[] = [
  {
    id: 'concept-explorer',
    name: 'Concept Explorer',
    description: 'Emphasizing knowledge graph exploration with supporting information',
    panels: [
      { id: 'kg-main', type: 'knowledge-graph', title: 'Knowledge Graph', defaultSize: 60 },
      { id: 'qa-side', type: 'rag-qa', title: 'Q&A Assistant', defaultSize: 40 }
    ],
    direction: 'horizontal',
    studyMode: 'concept-explorer'
  },
  {
    id: 'deep-study',
    name: 'Deep Study',
    description: 'Focusing on Q&A with source reference and note-taking',
    panels: [
      { id: 'qa-main', type: 'rag-qa', title: 'Q&A Assistant', defaultSize: 40 },
      { id: 'source-viewer', type: 'source-document', title: 'Source Documents', defaultSize: 30 },
      { id: 'notes-side', type: 'notes', title: 'Study Notes', defaultSize: 30 }
    ],
    direction: 'horizontal',
    studyMode: 'deep-study'
  },
  {
    id: 'review',
    name: 'Review',
    description: 'Optimized for practice questions and knowledge assessment',
    panels: [
      { id: 'practice-main', type: 'practice-exam', title: 'Practice Questions', defaultSize: 60 },
      { id: 'performance-side', type: 'performance-analysis', title: 'Performance', defaultSize: 40 }
    ],
    direction: 'horizontal',
    studyMode: 'review'
  },
  {
    id: 'time-15-min',
    name: '15-Minute Review',
    description: 'Quick review session for busy professionals',
    panels: [
      { id: 'srs-main', type: 'srs-review', title: 'Spaced Repetition', defaultSize: 70 },
      { id: 'kg-mini', type: 'knowledge-graph', title: 'Concept Map', defaultSize: 30 }
    ],
    direction: 'horizontal',
    studyMode: 'time-optimized',
    timeOptimization: {
      duration: 15,
      priority: 'review'
    }
  }
]

const usePanelsStore = create<PanelsState>()(
  subscribeWithSelector((set) => ({
    currentLayout: defaultLayouts[0],
    panelStates: {},
    panelContexts: {},
    savedLayouts: defaultLayouts,
    
    setCurrentLayout: (layout) => set({ currentLayout: layout }),
    
    savePanelState: (panelId, state) => set((prev) => ({
      panelStates: {
        ...prev.panelStates,
        [panelId]: { ...prev.panelStates[panelId], ...state }
      }
    })),
    
    updatePanelContext: (panelId, context) => set((prev) => ({
      panelContexts: {
        ...prev.panelContexts,
        [panelId]: { ...prev.panelContexts[panelId], ...context }
      }
    })),
    
    createCustomLayout: (name, description, panels) => {
      const newLayout: WorkspaceLayout = {
        id: `custom-${Date.now()}`,
        name,
        description,
        panels,
        direction: 'horizontal',
        studyMode: 'custom'
      }
      set((prev) => ({
        savedLayouts: [...prev.savedLayouts, newLayout]
      }))
    },
    
    updateLayout: (layoutId, updates) => set((prev) => ({
      savedLayouts: prev.savedLayouts.map(layout =>
        layout.id === layoutId ? { ...layout, ...updates } : layout
      ),
      currentLayout: prev.currentLayout?.id === layoutId
        ? { ...prev.currentLayout, ...updates }
        : prev.currentLayout
    })),
    
    deleteLayout: (layoutId) => set((prev) => ({
      savedLayouts: prev.savedLayouts.filter(layout => layout.id !== layoutId)
    })),
    
    setPanelSize: (panelId, size) => set((prev) => ({
      panelStates: {
        ...prev.panelStates,
        [panelId]: { ...prev.panelStates[panelId], size }
      }
    })),
    
    togglePanelVisibility: (panelId) => set((prev) => ({
      panelStates: {
        ...prev.panelStates,
        [panelId]: {
          ...prev.panelStates[panelId],
          isVisible: !(prev.panelStates[panelId]?.isVisible ?? false)
        }
      }
    })),
    
    pinPanel: (panelId) => set((prev) => ({
      panelStates: {
        ...prev.panelStates,
        [panelId]: {
          ...prev.panelStates[panelId],
          isPinned: !prev.panelStates[panelId]?.isPinned
        }
      }
    })),
    
    groupPanels: (panelIds, groupId) => set((prev) => {
      const updates: Record<string, PanelState> = {}
      panelIds.forEach(id => {
        updates[id] = { ...prev.panelStates[id], groupId }
      })
      return {
        panelStates: { ...prev.panelStates, ...updates }
      }
    }),
    
    ungroupPanel: (panelId) => set((prev) => ({
      panelStates: {
        ...prev.panelStates,
        [panelId]: { ...prev.panelStates[panelId], groupId: undefined }
      }
    }))
  }))
)

export default usePanelsStore