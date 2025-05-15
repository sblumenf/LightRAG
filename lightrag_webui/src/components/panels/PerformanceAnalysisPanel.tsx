import React from 'react'
import PanelWrapper from './PanelWrapper'
import { BarChart3 } from 'lucide-react'
import type { PanelConfig } from '@/types/panel'

interface PerformanceAnalysisPanelProps {
  panel: PanelConfig
}

export function PerformanceAnalysisPanel({ panel }: PerformanceAnalysisPanelProps) {
  return (
    <PanelWrapper
      panel={panel}
      icon={BarChart3}
      accentColor="green"
    >
      <div className="p-4">
        <p className="text-muted-foreground">Performance Analysis panel - Coming soon</p>
      </div>
    </PanelWrapper>
  )
}

export default PerformanceAnalysisPanel