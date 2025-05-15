import React from 'react'
import RetrievalTesting from '@/features/RetrievalTesting'
import PanelWrapper from './PanelWrapper'
import { MessageSquare } from 'lucide-react'
import type { PanelConfig } from '@/types/panel'

interface QAPanelProps {
  panel: PanelConfig
}

export function QAPanel({ panel }: QAPanelProps) {
  return (
    <PanelWrapper
      panel={panel}
      icon={MessageSquare}
      accentColor="cyan"
    >
      <RetrievalTesting />
    </PanelWrapper>
  )
}

export default QAPanel