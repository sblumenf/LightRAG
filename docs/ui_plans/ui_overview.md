# Detailed UI Implementation Plan: GraphRAG Tutor (with Specific Tech Choices)

**Overarching Goal:** Develop a flexible, user-customizable frontend application using **React**, enabling users to interact with the GraphRAG Tutor backend via configurable multi-panel workspaces, accessed through a flippable/switchable navigation system.

---

## Phase 0: Prerequisites & Frontend Foundation (Setup)

**Objective:** Establish the project environment, select and integrate core technologies, and ensure the backend API is ready.

* **Task 0.1: Backend API Readiness Verification**
    * **Description:** Confirm that critical backend enhancements outlined in `pre_ui_enhancement_plan.txt` are complete. This ensures the frontend has stable, functional endpoints to interact with.
    * **Action:** Verify completion of Phase 0 cleanup and critical Phase 1 tasks (Error Handling, File Upload Ingestion `/ingest`, Decoupled Generation `/generate`, Pipeline State Refactor) in the backend codebase. Consult API documentation (`API_README.md`, `api_plan.md`).
    * **Output:** Confirmed readiness of backend API endpoints required for subsequent frontend phases. Documented API endpoints available.

* **Task 0.2: Frontend Technology Stack Selection & Setup**
    * **Decision:** Use **React** as the primary framework and **Tailwind CSS** for styling.
    * **Description:** Set up the foundational tools for building the React application.
    * **Action:** Initialize the project using `Vite` (`npm create vite@latest my-graphrag-tutor --template react-ts`). Install `tailwindcss` and its peer dependencies, configure `tailwind.config.js` and `postcss.config.js` per Tailwind's framework guide for Vite/React. Choose and configure a package manager (`npm` or `yarn`).
    * **Output:** A functional skeleton React project with TypeScript and Tailwind CSS configured.

* **Task 0.3: Frontend Project Tooling Setup**
    * **Description:** Configure essential development tools for code quality, consistency, and version control.
    * **Action:** Install and configure `ESLint` with recommended React/TypeScript plugins (e.g., `eslint-plugin-react`, `@typescript-eslint/eslint-plugin`). Install and configure `Prettier` with Tailwind CSS plugin (`prettier-plugin-tailwindcss`). Initialize a `Git` repository and establish a basic branching strategy (e.g., main, develop, feature branches).
    * **Output:** Project configured with linting, formatting, and version control.

* **Task 0.4: UI Component Library / Design System Selection & Setup**
    * **Decision:** Use **Shadcn/ui**.
    * **Description:** Integrate a set of accessible, unstyled base components that can be customized with Tailwind CSS to match the desired professional aesthetic.
    * **Action:** Follow the Shadcn/ui installation guide using the `npx shadcn-ui@latest init` command. Configure `components.json`. Add a few initial base components (`button`, `input`, `card`, `dialog`, `tabs`) using the `npx shadcn-ui@latest add [component]` command to verify setup.
    * **Output:** Shadcn/ui integrated, providing accessible base components ready for use and styling.

* **Task 0.5: Layout Management Library/Strategy Selection & Setup**
    * **Decision:** Initially use **React-Grid-Layout** for grid positioning and **react-split** (wrapper for Split.js) for resizing panes.
    * **Description:** Integrate libraries to handle the dynamic arrangement and resizing of UI panels, crucial for the customizable workspace vision.
    * **Action:** Install `react-grid-layout` (`npm install react-grid-layout`) and `react-split` (`npm install react-split`). Create simple proof-of-concept components demonstrating basic grid layout configuration with RGL and a two-pane resizable split using react-split to ensure integration.
    * **Note:** GoldenLayout remains a consideration for future phases if advanced docking/stacking is needed.
    * **Output:** Layout management libraries installed and basic functionality verified.

* **Task 0.6: State Management Solution Selection & Setup**
    * **Decision:** Use **Zustand**.
    * **Description:** Integrate a state management library to handle application-wide state, particularly the shared context between interacting panels.
    * **Action:** Install `zustand` (`npm install zustand`). Create an initial Zustand store (`src/store/store.js` or similar) with a simple example state slice (e.g., managing UI theme or a mock user setting) and demonstrate accessing/updating it from a basic React component.
    * **Output:** Zustand integrated and basic store structure established.

---

## Phase 1: Core Panels & Fixed Layouts (Foundation)

**Objective:** Build the initial versions of essential UI panels, connect them to the backend for basic data flow, and arrange them in predefined, switchable layouts.

* **Task 1.1: Implement Core Functional Panel Components (Static UI)**
    * **Description:** Create the reusable React components for the main functional areas, focusing on the UI structure using Shadcn/ui components. Data will be mocked or static initially.
    * **Action:**
        * Develop `RAGQAPanel.tsx`: Use Card, Input, Button, ScrollArea components for layout. Implement display logic for mock chat messages.
        * Develop `KGVisualizationPanel.tsx`: Integrate `react-flow` (`npm install reactflow`). Set up basic React Flow provider and component structure. Display placeholder text or a hardcoded simple graph initially.
        * Develop `SourceDocumentViewerPanel.tsx`: Integrate `react-pdf` (`npm install react-pdf`). Implement basic UI to load and display a sample static PDF or placeholder text.
        * Develop `DocumentManagementPanel.tsx`: Use Table, Button components for UI. Display mock file list. Implement basic file input element (styling might be needed).
    * **Output:** Structurally complete, reusable React components for core panels using Shadcn/ui and integrated specialized libraries (`react-flow`, `react-pdf`).

* **Task 1.2: Implement Default Workspace Layouts (Fixed Arrangement)**
    * **Description:** Define and implement the structure for the initial, non-customizable workspace layouts using the chosen layout libraries.
    * **Action:** Create React components representing each default workspace (e.g., `BasicQALayout.tsx`, `ConceptExplorerLayout.tsx`). Within these components, use `react-grid-layout` to position the relevant panels (from Task 1.1) and `react-split` to create resizable divisions between them where appropriate (e.g., Q&A panel vs. Source Viewer panel). Configure initial sizes and positions.
    * **Output:** React components rendering the predefined arrangements of core panels.

* **Task 1.3: Implement Basic Workspace Switching Logic**
    * **Description:** Allow users to navigate between the different fixed workspace layouts created above.
    * **Action:** Implement a state slice in the Zustand store to track the `activeWorkspace`. Create a simple navigation component (e.g., using `Tabs` from Shadcn/ui) where each tab updates the `activeWorkspace` state in Zustand. Conditionally render the appropriate workspace layout component (from Task 1.2) based on the `activeWorkspace` state.
    * **Output:** Functional navigation allowing users to switch between the fixed `BasicQALayout` and `ConceptExplorerLayout` views.

* **Task 1.4: Basic API Integration (Connecting Panels to Backend)**
    * **Description:** Make the core panels functional by connecting them to the verified backend API endpoints. Manage API call state (loading, error) using Zustand.
    * **Action:**
        * `RAGQAPanel`: Implement function (e.g., triggered by Button click) to take input value, call the backend `/generate` endpoint (using `fetch` or `axios`), update Zustand state with the response (answer, citations), and handle loading/error states. Display results in the chat area.
        * `DocumentManagementPanel`: Implement file upload logic using the file input, sending the selected file(s) to the `/ingest` endpoint. Implement a function to fetch the list of available documents from a new backend endpoint (needs definition) and display it in the Table. Handle loading/error states.
        * `KGVisualizationPanel`: Implement a function (e.g., triggered on load or by a topic change) to fetch graph data (nodes, edges) from a relevant backend endpoint (needs definition, potentially accepting query parameters). Pass fetched data to `react-flow` for rendering. Handle loading/error states.
        * `SourceDocumentViewerPanel`: Implement logic to fetch document content (or specific chunks based on citations) from a backend endpoint (needs definition) when requested. Load the fetched content into `react-pdf` or display text. Handle loading/error states.
    * **Output:** Core panels are now interactive, fetching and displaying data from the backend API, with basic loading/error handling managed via Zustand.

---

## Phase 2: Interactivity & Enhanced Navigation (Core UX)

**Objective:** Enhance the user experience by making panels interactive with each other, implementing the final navigation design, enabling resizing, and integrating the remaining core functional panels.

* **Task 2.1: Implement Panel Resizing Persistence**
    * **Description:** Allow users to resize panels within the fixed layouts and have those sizes remembered during their session.
    * **Action:** Configure `react-split` event handlers (`onDragEnd`) to capture the new sizes when a user finishes dragging a splitter. Store these sizes in the Zustand store (e.g., associated with the active workspace layout). Ensure the `react-split` component reads its initial sizes from the Zustand store when the workspace loads.
    * **Output:** Panels within default layouts are resizable, and sizes are persisted within the current session via Zustand.

* **Task 2.2: Implement Basic Inter-Panel Communication (Shared Context via Zustand)**
    * **Description:** Make panels aware of actions happening in other panels within the same workspace by using the shared Zustand store.
    * **Action:** Define specific state slices in Zustand for shared context (e.g., `currentTopic`, `selectedKGNodeId`, `highlightedSourceChunkId`).
        * Modify `RAGQAPanel`: When a citation is clicked, update the `highlightedSourceChunkId` state in Zustand.
        * Modify `SourceDocumentViewerPanel`: Subscribe to `highlightedSourceChunkId` from Zustand. When it changes, fetch the relevant chunk (if needed) and implement highlighting logic.
        * Modify `KGVisualizationPanel`: When a node is clicked/selected in `react-flow`, update the `selectedKGNodeId` and potentially `currentTopic` state in Zustand.
        * Modify `RAGQAPanel`: Subscribe to `selectedKGNodeId`/`currentTopic` state. Optionally, pre-fill the query input or display context information when this state changes.
    * **Output:** Core panels now interact contextually based on user actions in sibling panels within the same workspace, mediated by Zustand.

* **Task 2.3: Implement Chosen Workspace Switcher UI**
    * **Description:** Implement the final visual design for navigating between workspaces, replacing the simple tabs/buttons from Phase 1.
    * **Action:** Based on the chosen design (e.g., Carousel, enhanced Tabs, Dropdown):
        * Build the necessary React components using Shadcn/ui primitives.
        * Integrate the logic to update the `activeWorkspace` state in Zustand upon user interaction (click, swipe).
        * Ensure smooth visual transitions between workspace views (if applicable, e.g., for a carousel).
        * Provide clear visual indication of the currently active workspace.
    * **Output:** The polished, final navigation mechanism for switching between workspaces is implemented.

* **Task 2.4: Implement Remaining Core Panels & API Integration**
    * **Description:** Build the React components and integrate the backend API connections for the remaining functional panels.
    * **Action:** Develop and integrate:
        * `QuizPanel.tsx`: UI using Shadcn/ui (RadioGroup, Button, Card). Connect to backend endpoints (needs definition) for fetching quiz questions based on topic/syllabus, submitting answers, and receiving results. Manage quiz state (current question, score) in Zustand.
        * `SRSReviewPanel.tsx`: UI using Card, Button. Connect to backend endpoints (needs definition) for fetching due flashcards and posting review results (e.g., "easy," "hard"). Manage review session state in Zustand.
        * `SyllabusBrowserPanel.tsx`: UI potentially using a tree view component (find or adapt one, or use nested Cards/Accordions from Shadcn/ui). Fetch hierarchical syllabus data from a backend endpoint (needs definition). Implement logic to potentially update `currentTopic` in Zustand when an item is selected.
        * `PerformanceTrackerPanel.tsx`: UI using a charting library (e.g., `recharts`, `nivo`) integrated with React. Fetch performance summary data (quiz scores, SRS stats) from backend endpoints (needs definition) and render charts.
        * `NotesPanel.tsx`: Integrate a simple rich-text or Markdown editor component (e.g., `react-markdown`, `react-quill`). Implement basic note saving (initially maybe to Zustand/Local Storage).
    * **Output:** All core functional panels defined in the vision are implemented as React components and connected to relevant (potentially new) backend APIs via Zustand.

* **Task 2.5: Refine API Integrations & State Management**
    * **Description:** Improve the robustness and user experience of data handling and state updates.
    * **Action:** Review all API integration points. Implement consistent loading indicators (e.g., Skeleton components from Shadcn/ui, spinners) visible to the user during API calls. Implement user-friendly error handling (e.g., using Alert or Toast components from Shadcn/ui) for failed API calls. Review Zustand store structure for clarity, ensuring efficient updates and preventing unnecessary re-renders (using selectors).
    * **Output:** More polished, robust, and user-friendly interactions with the backend and state management.

---

## Phase 3: Full Customization (The Vision)

**Objective:** Implement the advanced features allowing users to create, modify, arrange, and save their own custom workspace layouts. This is the most complex phase.

* **Task 3.1: Design "Workspace Editor" UI/UX**
    * **Description:** Create the detailed visual and interaction design for the interface that empowers users to build their own workspaces.
    * **Action:** Develop high-fidelity mockups (e.g., in Figma) and potentially interactive prototypes demonstrating:
        * How users enter/exit "Edit Mode."
        * A palette/menu displaying available panels (`RAGQAPanel`, `KGVisualizationPanel`, etc.).
        * The mechanism for adding selected panels to the layout canvas.
        * The mechanism for removing panels.
        * How users resize panels (dragging splitters).
        * How users reposition panels (e.g., drag-and-drop using `react-grid-layout` handles).
        * The interface for saving the layout (naming, save button).
        * Get feedback on the design's intuitiveness before implementation.
    * **Output:** A detailed, user-tested, and approved UI/UX design specification for the workspace editor.

* **Task 3.2: Implement Layout Persistence (Local Storage)**
    * **Description:** Store the definitions of user-created workspace layouts so they persist between browser sessions.
    * **Action:** Define a clear JSON structure to represent a workspace layout (e.g., `{ id: 'uuid', name: 'My Layout', panels: [{ id: 'panel-1', type: 'RAGQAPanel', gridPos: { x: 0, y: 0, w: 6, h: 10 } }, { id: 'panel-2', type: 'KGVisualizationPanel', gridPos: { ... } }], splitSizes: [...] }`). Use Zustand middleware (like `persist` from `zustand/middleware`) or custom logic to automatically save relevant state slices (containing user layouts and default layouts) to the browser's Local Storage whenever they change. Implement logic to load these layouts from Local Storage into the Zustand store when the application initializes.
    * **Output:** User-defined workspace layouts are saved locally and reloaded when the user returns to the application.

* **Task 3.3: Implement Dynamic Panel Rendering & Layout**
    * **Description:** Modify the application's core rendering logic to display workspaces based on the dynamic layout definitions loaded from the Zustand store, rather than hardcoded fixed layouts.
    * **Action:** Refactor the main layout component. It should:
        * Read the definition of the `activeWorkspace` from the Zustand store.
        * Iterate over the `panels` array in the layout definition.
        * For each panel definition, dynamically render the corresponding React component (e.g., using a switch statement or mapping based on `panel.type`).
        * Pass the `gridPos` data from the layout definition to the `react-grid-layout` component's `layout` prop.
        * Pass the `splitSizes` data to the `react-split` component(s) used within the layout.
    * **Output:** The UI dynamically renders the correct panels in the correct positions and sizes based on the currently selected (and potentially user-defined) workspace layout stored in Zustand.

* **Task 3.4: Implement Panel Arrangement Logic (Editor Mode)**
    * **Description:** Build the interactive functionality of the "Workspace Editor" mode, allowing users to manipulate the layout visually.
    * **Action:**
        * Create the "Edit Mode" UI components (panel palette, save button, etc.) based on Task 3.1 design.
        * Implement logic to add a new panel definition to the current workspace layout state in Zustand when a user selects it from the palette.
        * Implement logic to remove a panel definition from the Zustand state when a user clicks a 'close' or 'remove' button on a panel (only visible in Edit Mode).
        * Configure `react-grid-layout`'s `onLayoutChange` prop. When the user drags or resizes a panel within the grid, this callback receives the updated layout array. Use this data to update the `gridPos` information in the corresponding panel definitions within the Zustand store.
        * Configure `react-split`'s `onDragEnd` prop to update the `splitSizes` array in the Zustand store when a splitter is moved.
    * **Output:** A functional "Workspace Editor" mode where users can visually add, remove, reposition, and resize panels, with changes reflected in the Zustand state.

* **Task 3.5: Implement Saving/Loading/Managing Custom Workspaces**
    * **Description:** Provide the UI and logic for users to fully manage their collection of custom workspaces.
    * **Action:**
        * Implement UI actions (e.g., "New Workspace" button, "Save As..." option) that create new layout definitions in the Zustand store or update existing ones. Use Shadcn/ui Dialogs for prompts like naming a new workspace.
        * Implement a "Delete Workspace" action that removes the selected layout definition from the Zustand store.
        * Modify the Workspace Switcher component (Task 2.3) to dynamically populate its options (Tabs, Dropdown items) based on the list of available workspace definitions (both default and user-defined) stored in Zustand. Ensure selecting an option updates the `activeWorkspace` state correctly.
    * **Output:** Users can create, name, save, load, and delete their custom workspace layouts through the UI.

* **Task 3.6: Advanced State Management for Custom Layouts**
    * **Description:** Ensure the inter-panel communication logic and shared context handling remain robust and efficient, regardless of the user's custom panel arrangements.
    * **Action:** Review and potentially refactor the Zustand store structure and component subscriptions. Ensure that context updates (like `selectedKGNodeId`) correctly affect *only* the relevant panels currently visible in the active workspace. Optimize selectors to prevent unnecessary re-renders in panels when unrelated parts of the state change. Consider strategies for lazy-loading panel components if performance becomes an issue with many potential panels.
    * **Output:** Robust and performant state management and inter-panel communication that adapts correctly to fully user-defined layouts.

---

## Phase 4: Testing & Refinement (Continuous)

**Objective:** Ensure the application is high-quality, reliable, performant, accessible, and meets user needs through rigorous testing and iterative refinement. This phase occurs throughout development but includes dedicated final checks.

* **Task 4.1: Component Testing**
    * **Description:** Verify individual React components function correctly in isolation.
    * **Action:** Write unit tests for complex components (especially panels), custom hooks, and utility functions using **Vitest** (or Jest) and **React Testing Library**. Test different props, states, and user interactions within the component.
    * **Output:** High code coverage for critical components and logic, ensuring individual parts work as expected.

* **Task 4.2: Integration Testing (Frontend-Backend)**
    * **Description:** Test the communication layer between the frontend application and the backend API.
    * **Action:** Use **Mock Service Worker (MSW)** to intercept API calls during tests. Write tests that simulate user actions triggering API calls and assert that the correct requests are made and that the UI updates correctly based on mock responses (success and error cases).
    * **Output:** Verified logic for API request/response handling in the frontend, independent of a live backend.

* **Task 4.3: End-to-End (E2E) Testing**
    * **Description:** Simulate real user workflows from start to finish in a browser environment.
    * **Action:** Use **Cypress** or **Playwright** to write scripts that automate scenarios like:
        * Uploading a document, asking a question, verifying the answer and citation.
        * Exploring the KG, selecting a node, asking a related question.
        * Creating a custom workspace layout, saving it, reloading the app, and verifying the layout loads correctly.
        * Completing a quiz or an SRS review session.
        * Run these tests against a staging environment with a live backend.
    * **Output:** Automated regression tests covering critical user journeys, ensuring major features work together correctly.

* **Task 4.4: Usability Testing**
    * **Description:** Gather qualitative feedback from target users interacting with the application.
    * **Action:** Recruit representative users (CFA/PMP candidates). Prepare test scenarios focusing on key tasks, especially workspace customization. Observe users interacting with the application (live or recorded). Collect feedback via think-aloud protocol, interviews, or surveys. Analyze feedback to identify pain points and areas for improvement.
    * **Output:** Actionable insights into user experience, leading to UI/UX refinements.

* **Task 4.5: Performance Testing & Optimization**
    * **Description:** Ensure the application is fast and responsive, even with complex layouts or data.
    * **Action:** Use **React DevTools Profiler** to identify slow-rendering components. Use browser performance tools (Lighthouse, Performance tab) to measure load times, interaction latency, and memory usage. Optimize React component rendering (e.g., `React.memo`, `useMemo`, `useCallback`). Optimize Zustand state updates and selectors. Analyze and optimize the performance of layout libraries and visualization components (e.g., `react-flow` with large graphs).
    * **Output:** Identified performance bottlenecks addressed, resulting in a smooth and responsive user experience.

* **Task 4.6: Accessibility Audit**
    * **Description:** Ensure the application is usable by people with disabilities, adhering to WCAG standards.
    * **Action:** Use automated accessibility testing tools (e.g., **Axe DevTools** browser extension). Perform manual testing for keyboard-only navigation across all interactive elements (buttons, inputs, tabs, graph nodes, splitters). Test using screen reader software (e.g., NVDA, VoiceOver). Verify sufficient color contrast ratios. Leverage accessibility features built into **Shadcn/ui**.
    * **Output:** An application compliant with accessibility standards, usable by a wider range of users.

---