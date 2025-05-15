import React from 'react'
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels'
import { useCallback, useMemo } from 'react'
import { cn } from '@/lib/utils'
import usePanelsStore from '@/stores/panels'
import type { WorkspaceLayout, PanelConfig } from '@/types/panel'
import {
  KnowledgeGraphPanel,
  QAPanel,
  SourceDocumentPanel,
  PerformanceAnalysisPanel,
  PracticeExamPanel,
  NotesPanel,
} from './index'

interface PanelContainerProps {
  className?: string
}

export function PanelContainer({ className }: PanelContainerProps) {
  const currentLayout = usePanelsStore((state) => state.currentLayout)
  const setPanelSize = usePanelsStore((state) => state.setPanelSize)
  const savePanelState = usePanelsStore((state) => state.savePanelState)

  const handlePanelResize = useCallback(
    (panelId: string, size: number) => {
      setPanelSize(panelId, size)
      savePanelState(panelId, { size })
    },
    [setPanelSize, savePanelState]
  )

  const renderPanelContent = useCallback((panel: PanelConfig) => {
    switch (panel.type) {
    case 'knowledge-graph':
      return <KnowledgeGraphPanel panel={panel} />
    case 'rag-qa':
      return <QAPanel panel={panel} />
    case 'source-document':
      return <SourceDocumentPanel panel={panel} />
    case 'performance-analysis':
      return <PerformanceAnalysisPanel panel={panel} />
    case 'practice-exam':
      return <PracticeExamPanel panel={panel} />
    case 'notes':
      return <NotesPanel panel={panel} />
    default:
      return (
        <div className="p-4">
          <p className="text-muted-foreground">Unknown panel type: {panel.type}</p>
        </div>
      )
    }
  }, [])

  const renderPanel = useCallback((panel: PanelConfig, index: number) => {
    const isLastPanel = index === currentLayout!.panels.length - 1
    const panelState = usePanelsStore.getState().panelStates[panel.id]
    const isVisible = panelState?.isVisible ?? true
    
    if (!isVisible) return null
    
    return (
      <React.Fragment key={panel.id}>
        <Panel
          id={panel.id}
          defaultSize={panel.defaultSize ?? 50}
          minSize={panel.minSize ?? 20}
          maxSize={panel.maxSize ?? 80}
          onResize={(size) => handlePanelResize(panel.id, size)}
        >
          <div className="h-full w-full bg-background">
            {renderPanelContent(panel)}
          </div>
        </Panel>
        {!isLastPanel && (
          <PanelResizeHandle className={cn(
            'group transition-colors',
            currentLayout!.direction === 'horizontal'
              ? 'w-2 hover:bg-gray-300'
              : 'h-2 hover:bg-gray-300'
          )}>
            <div className={cn(
              'transition-all',
              currentLayout!.direction === 'horizontal'
                ? 'h-8 w-0.5 bg-gray-300 group-hover:w-1 group-hover:bg-gray-400'
                : 'w-8 h-0.5 bg-gray-300 group-hover:h-1 group-hover:bg-gray-400',
              'mx-auto my-auto'
            )} />
          </PanelResizeHandle>
        )}
      </React.Fragment>
    )
  }, [currentLayout, handlePanelResize, renderPanelContent])

  const panels = useMemo(() => {
    if (!currentLayout) return null
    return currentLayout.panels.map((panel, index) => renderPanel(panel, index))
  }, [currentLayout, renderPanel])

  if (!currentLayout) {
    return (
      <div className={cn('flex h-full items-center justify-center', className)}>
        <p className="text-muted-foreground">No layout selected</p>
      </div>
    )
  }

  return (
    <PanelGroup
      direction={currentLayout.direction}
      id={currentLayout.id}
      className={cn('h-full w-full', className)}
    >
      {panels}
    </PanelGroup>
  )
}

export default PanelContainer