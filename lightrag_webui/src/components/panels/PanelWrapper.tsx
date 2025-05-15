import React from 'react'
import { X, Maximize2, Minimize2, Pin, PinOff } from 'lucide-react'
import { cn } from '@/lib/utils'
import usePanelsStore from '@/stores/panels'
import Button from '@/components/ui/Button'
import type { PanelConfig } from '@/types/panel'

interface PanelWrapperProps {
  panel: PanelConfig
  children: React.ReactNode
  className?: string
  accentColor?: string
  icon?: React.ComponentType<{ className?: string }>
  actions?: React.ReactNode
}

export function PanelWrapper({
  panel,
  children,
  className,
  accentColor = 'blue',
  icon: Icon,
  actions,
}: PanelWrapperProps) {
  const { togglePanelVisibility, pinPanel, savePanelState } = usePanelsStore()
  const panelState = usePanelsStore((state) => state.panelStates[panel.id])
  
  const { isCollapsed = false, isPinned = false } = panelState || {}

  const handleClose = () => {
    togglePanelVisibility(panel.id)
  }

  const handlePin = () => {
    pinPanel(panel.id)
  }

  const handleToggleCollapse = () => {
    savePanelState(panel.id, { isCollapsed: !isCollapsed })
  }

  const accentColorClass = `text-${accentColor}-500`

  return (
    <div className={cn('h-full flex flex-col bg-white rounded-lg shadow-sm border border-gray-200', className)}>
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-100 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          {Icon && <Icon className={cn('w-5 h-5', accentColorClass)} />}
          <h3 className="font-medium text-gray-900">{panel.title}</h3>
        </div>
        <div className="flex items-center gap-1">
          {actions}
          <Button
            variant="ghost"
            size="sm"
            onClick={handlePin}
            className="h-8 w-8 p-0"
            title={isPinned ? 'Unpin panel' : 'Pin panel'}
          >
            {isPinned ? (
              <PinOff className="h-4 w-4" />
            ) : (
              <Pin className="h-4 w-4" />
            )}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleToggleCollapse}
            className="h-8 w-8 p-0"
            title={isCollapsed ? 'Expand panel' : 'Collapse panel'}
          >
            {isCollapsed ? (
              <Maximize2 className="h-4 w-4" />
            ) : (
              <Minimize2 className="h-4 w-4" />
            )}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleClose}
            className="h-8 w-8 p-0"
            title='Close panel'
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>
      
      {/* Content */}
      {!isCollapsed && (
        <div className="flex-1 overflow-auto">
          {children}
        </div>
      )}
    </div>
  )
}

export default PanelWrapper