import React from 'react'
import DocumentManager from '@/features/DocumentManager'
import PanelWrapper from './PanelWrapper'
import { FileText } from 'lucide-react'
import type { PanelConfig } from '@/types/panel'

interface SourceDocumentPanelProps {
  panel: PanelConfig
}

export function SourceDocumentPanel({ panel }: SourceDocumentPanelProps) {
  return (
    <PanelWrapper
      panel={panel}
      icon={FileText}
      accentColor="orange"
    >
      <DocumentManager />
    </PanelWrapper>
  )
}

export default SourceDocumentPanel