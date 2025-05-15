import React from 'react'
import { Plus, Layout } from 'lucide-react'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select'
import Button from '@/components/ui/Button'
import type { WorkspaceLayout } from '@/types/panel'

interface LayoutSelectorProps {
  currentLayout: WorkspaceLayout | null
  layouts: WorkspaceLayout[]
  onLayoutChange: (layout: WorkspaceLayout) => void
}

export function LayoutSelector({ currentLayout, layouts, onLayoutChange }: LayoutSelectorProps) {
  const handleLayoutChange = (layoutId: string) => {
    const layout = layouts.find(l => l.id === layoutId)
    if (layout) {
      onLayoutChange(layout)
    }
  }

  return (
    <div className="flex items-center gap-4">
      <div className="flex items-center gap-2">
        <Layout className="w-4 h-4 text-gray-500" />
        <span className="text-sm font-medium text-gray-700">Layout:</span>
      </div>
      <Select 
        value={currentLayout?.id} 
        onValueChange={handleLayoutChange}
      >
        <SelectTrigger className="w-[200px]">
          <SelectValue placeholder="Select a layout" />
        </SelectTrigger>
        <SelectContent>
          {layouts.map((layout) => (
            <SelectItem key={layout.id} value={layout.id}>
              <div>
                <div className="font-medium">{layout.name}</div>
                {layout.description && (
                  <div className="text-xs text-gray-500">{layout.description}</div>
                )}
              </div>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Button
        variant="outline"
        size="sm"
        className="gap-1"
        onClick={() => {
          // TODO: Implement custom layout creation
          console.log('Create custom layout')
        }}
      >
        <Plus className="w-4 h-4" />
        Custom Layout
      </Button>
    </div>
  )
}

export default LayoutSelector