import React from 'react'
import PanelWrapper from './PanelWrapper'
import { GraduationCap } from 'lucide-react'
import type { PanelConfig } from '@/types/panel'

interface PracticeExamPanelProps {
  panel: PanelConfig
}

export function PracticeExamPanel({ panel }: PracticeExamPanelProps) {
  return (
    <PanelWrapper
      panel={panel}
      icon={GraduationCap}
      accentColor="purple"
    >
      <div className="p-4">
        <p className="text-muted-foreground">Practice Exam panel - Coming soon</p>
      </div>
    </PanelWrapper>
  )
}

export default PracticeExamPanel