# Panel System Setup Guide

## Overview
The LightRAG UI now includes a panel-based workspace system built with react-resizable-panels. This allows for flexible layout arrangements and better multi-context learning experiences.

## Setup Instructions

### 1. Install Dependencies
```bash
cd lightrag_webui
bun install
```

### 2. Configure Environment
Create a `.env.development` file from the example:
```bash
cp .env.development.example .env.development
```

The default configuration should work for local development:
```
VITE_API_PROXY=true
VITE_API_ENDPOINTS=/api,/auth-status,/login,/docs,/openapi.json
VITE_BACKEND_URL=http://localhost:9621
```

### 3. Start the Development Server
```bash
bun run dev
```

### 4. Access the Application
1. Open your browser to: `http://localhost:5173/webui/`
2. Login with any credentials (authentication is disabled in development)
3. Navigate to the "Panels" tab to see the new panel-based workspace

## Features

### Panel Types
- **Knowledge Graph Panel**: Interactive graph visualization
- **Q&A Panel**: Chat interface for questions and answers
- **Source Document Panel**: Document viewer
- **Performance Analysis Panel**: Performance metrics (placeholder)
- **Practice Exam Panel**: Practice questions (placeholder)
- **Notes Panel**: Note-taking (placeholder)

### Layout Presets
- **Concept Explorer**: Focus on knowledge graph with Q&A support
- **Deep Study**: Q&A, source documents, and notes
- **Review**: Practice exams and performance analysis
- **15-Minute Review**: Spaced repetition with concept map

### Panel Controls
- **Resize**: Drag panel dividers to resize
- **Pin**: Keep panel in place
- **Collapse/Expand**: Minimize panel to save space
- **Close**: Hide panel (can be restored by switching layouts)

## Troubleshooting

### Login Issues
If you can't login:
1. Ensure the backend is running at `http://localhost:9621`
2. Check that authentication is disabled (guest mode)
3. Clear browser cache and cookies
4. Check browser console for errors

### Connection Issues
If the frontend can't connect to the backend:
1. Verify the backend is running
2. Check the `.env.development` file is properly configured
3. Restart the development server after environment changes
4. Check for CORS errors in browser console

### Panel Issues
If panels aren't working correctly:
1. Clear localStorage to reset panel states
2. Refresh the page
3. Try a different browser or incognito mode