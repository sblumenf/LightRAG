import React from 'react'
import { PanelContainer } from '@/components/panels'
import LayoutSelector from './LayoutSelector'
import usePanelsStore from '@/stores/panels'

export function PanelBasedWorkspace() {
  const currentLayout = usePanelsStore((state) => state.currentLayout)
  const setCurrentLayout = usePanelsStore((state) => state.setCurrentLayout)
  const savedLayouts = usePanelsStore((state) => state.savedLayouts)

  return (
    <div className="flex flex-col h-full w-full">
      {/* Layout Selector */}
      <div className="px-4 py-2 border-b border-gray-200 bg-gray-50">
        <LayoutSelector
          currentLayout={currentLayout}
          layouts={savedLayouts}
          onLayoutChange={setCurrentLayout}
        />
      </div>
      
      {/* Panel Container */}
      <div className="flex-1 p-4">
        <PanelContainer />
      </div>
    </div>
  )
}

export default PanelBasedWorkspace