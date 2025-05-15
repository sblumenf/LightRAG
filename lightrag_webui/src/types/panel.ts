export interface PanelConfig {
  id: string;
  type: 'knowledge-graph' | 'rag-qa' | 'source-document' | 'performance-analysis' | 'practice-exam' | 'quiz' | 'srs-review' | 'notes' | 'syllabus';
  title: string;
  minSize?: number;
  defaultSize?: number;
  maxSize?: number;
  order?: number;
}

export interface WorkspaceLayout {
  id: string;
  name: string;
  description: string;
  panels: PanelConfig[];
  direction: 'horizontal' | 'vertical';
  studyMode?: 'concept-explorer' | 'deep-study' | 'review' | 'time-optimized' | 'custom';
  timeOptimization?: {
    duration: number; // in minutes
    priority: 'review' | 'learning' | 'practice';
  };
}

export interface PanelState {
  isVisible: boolean;
  size: number;
  isCollapsed: boolean;
  isPinned: boolean;
  groupId?: string;
}

export interface PanelContextData {
  selectedNode?: string;
  selectedDocument?: string;
  currentQuery?: string;
  studyFocus?: string;
}