import React from 'react'
import GraphViewer from '@/features/GraphViewer'
import PanelWrapper from './PanelWrapper'
import { Network } from 'lucide-react'
import type { PanelConfig } from '@/types/panel'

interface KnowledgeGraphPanelProps {
  panel: PanelConfig
}

export function KnowledgeGraphPanel({ panel }: KnowledgeGraphPanelProps) {
  return (
    <PanelWrapper
      panel={panel}
      icon={Network}
      accentColor="violet"
    >
      <GraphViewer />
    </PanelWrapper>
  )
}

export default KnowledgeGraphPanel