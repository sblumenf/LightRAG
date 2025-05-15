import React from 'react'
import PanelWrapper from './PanelWrapper'
import { NotebookPen } from 'lucide-react'
import type { PanelConfig } from '@/types/panel'

interface NotesPanelProps {
  panel: PanelConfig
}

export function NotesPanel({ panel }: NotesPanelProps) {
  return (
    <PanelWrapper
      panel={panel}
      icon={NotebookPen}
      accentColor="yellow"
    >
      <div className="p-4">
        <p className="text-muted-foreground">Notes panel - Coming soon</p>
      </div>
    </PanelWrapper>
  )
}

export default NotesPanel