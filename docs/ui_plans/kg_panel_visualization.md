#   Detailed Implementation Plan: KGVisualizationPanel Enhancements (Commercial Readiness Revision)

**Overarching Goal:** Develop a highly interactive, pedagogically rich, robust, scalable, and commercially viable knowledge graph visualization panel, incorporating core exploration, RAG-based features, advanced learning support, and user-centric design.

**Inspiration & Context:** The vision for this dynamic and interactive visualization component draws inspiration from tools like Infranodus, particularly in its approach to revealing connections and providing analytical insights through graph representation. However, while inspired by such tools, this plan details a component specifically tailored for the educational context of exam preparation. Its features focus on interacting with a pre-defined, structured knowledge graph derived from syllabus content, integrating unique pedagogical tools (like prerequisite paths, formula interaction, confidence mapping, curated resources) and RAG capabilities distinct from general text analysis or discovery tools.

**General Notes for Commercial Readiness:**
* **Robust Error Handling:** Implement user-friendly error messages, graceful degradation on failure, and potential recovery/retry mechanisms throughout.
* **Scalability:** Proactively consider performance implications of large graphs in both backend query design and frontend rendering strategies.
* **Security:** Ensure sanitization of externally sourced data (especially LLM output) and secure handling of user-specific state.
* **Maintainability:** Emphasize modular code structure, clear comments, and potentially component documentation (e.g., Storybook) for this complex component.
* **Accessibility:** Go beyond basic compliance; ensure specific graph interactions are accessible via keyboard and screen readers.

---

##   Phase 1: Interactive Drill-Down & Foundation

* **Task 1.1: Backend API Enhancement (Graph Data with Details)**
    * Description: Modify the backend API to provide detailed information about nodes (including properties, curated video links if available) and edges when requested. *Ensure queries are optimized for potential graph scale.*
    * Target File: `graphrag_tutor/api_server.py`, `graphrag_tutor/knowledge_graph/neo4j_kg_service.py` (and related files for data retrieval).
    * Micro-processes:
        1.  API Endpoint Modification: Create/Modify endpoints (e.g., `/graph/node/{node_id}`, `/graph/edge/{edge_id}`) for details, including video links list. Modify base graph endpoint for unique IDs.
        2.  Knowledge Graph Service Logic: Implement logic to fetch details from Neo4j. *Prioritize indexed lookups and efficient Cypher queries*. Consider caching.
        3.  Data Transformation: Ensure structured JSON output. Define clear data structures.
    * Verification: Test endpoints (Swagger/Postman). Verify correct data/format (incl. video links). Ensure base endpoint includes IDs. *Verify query performance under simulated load (if possible).*

* **Task 1.2: Frontend Implementation (Node/Edge Interaction & Basic Detail Display)**
    * Description: Implement basic interactive drill-down functionality, handling initial states and basic errors gracefully.
    * Target File: `src/components/KGVisualizationPanel.tsx`
    * Micro-processes:
        1.  Event Handling: Add `react-flow` event handlers for node/edge clicks.
        2.  Data Fetching: On click, retrieve ID, call backend API. Implement user-friendly loading states (e.g., skeletons) and clear error messages (e.g., toasts/alerts) for API failures or missing data.
        3.  UI Display (Basic): Use modal/tooltip/section (Shadcn/ui) for basic details (name, description). *Ensure sanitization of displayed text data*. Handle empty/null states gracefully (e.g., "Details not available"). Defer video link display.
        4.  State Management: Use Zustand for selected element state and fetched details.
    * Verification: Test clicks. Verify details fetched/displayed. Test loading states. *Verify user-friendly error handling and graceful display for empty/error states*. Ensure UI responsiveness. *Confirm displayed data is sanitized.*

* **Task 1.3: Node and Edge Styling**
    * Description: Define a clear, consistent, and accessible visual style.
    * Target File: `src/components/KGVisualizationPanel.tsx`
    * Micro-processes: (Color Palette, Node Shapes/Icons, Edge Styles, Typography as before) *Ensure high color contrast ratios meeting WCAG AA standards.*
    * Verification: Visualization clear, informative, accessible. Styles consistent. *Verify color contrast.*

* **Task 1.4: Node Interaction Animations**
    * Description: Add subtle, performant animations for node interaction feedback.
    * Target File: `src/components/KGVisualizationPanel.tsx`
    * Micro-processes: (Hover Effect, Click Effect, Node Expansion as before) *Ensure animations are implemented efficiently (e.g., using CSS transitions) and do not degrade performance on larger graphs.*
    * Verification: Animations smooth, responsive, clear feedback, not distracting. *Verify no performance impact.*

* **Task 1.5: Node Progress Indicators (Placeholder)**
    * Description: Add UI-only placeholders for student progress, ensuring accessibility.
    * Target File: `src/components/KGVisualizationPanel.tsx`
    * Micro-processes: (Node Modification, Visual Styles, Placeholder Data, Accessibility as before)
    * Verification: Nodes display placeholder correctly. Visual is intuitive and accessible.

##   Phase 2: Core Interactive Features & Connections

* **Task 2.1: Frontend Implementation (Lasso Selection)**
    * Description: Implement lasso selection functionality.
    * Target File: `src/components/KGVisualizationPanel.tsx`
    * Micro-processes: (Lasso Library Integration, Event Handling, Node Selection Logic, "Create Lesson" Button, State Management as before) *Consider accessibility alternatives if lasso is purely mouse-driven.*
    * Verification: Test lasso selection. Verify correct nodes selected/highlighted. Button enables correctly. *Evaluate accessibility.*

* **Task 2.2: Backend API Enhancement (Basic Lesson Generation)**
    * Description: Implement backend API/logic for generating a basic lesson from selected nodes. *Focus on quality and robustness.*
    * Target File: `graphrag_tutor/api_server.py`, `graphrag_tutor/generation/lesson_generator.py`
    * Micro-processes:
        1.  API Endpoint Creation: Create endpoint (`/lessons/generate`) accepting node IDs. Implement robust input validation.
        2.  Lesson Generation Logic: Implement `lesson_generator.py`: Fetch details, Use LLM Generator for basic content. *Implement error handling for LLM API failures or unexpected responses.* Structure content clearly.
        3.  Data Transformation: Ensure structured JSON. Define schema clearly.
    * Verification: Test endpoint with valid/invalid node IDs. Verify structured content returned. Evaluate quality/relevance. *Test LLM error handling.*

* **Task 2.3: Frontend Implementation (Basic Lesson Display)**
    * Description: Implement UI to display the generated lesson, handling potential errors and sanitizing content.
    * Target File: `src/components/KGVisualizationPanel.tsx`
    * Micro-processes:
        1.  API Call: Call `/lessons/generate`. Implement user-friendly loading/error states (e.g., "Lesson generation failed, please try again").
        2.  Lesson Display UI: Use modal/panel section (Shadcn/ui) for content. *Sanitize LLM-generated HTML/text before rendering.* Handle empty response state.
        3.  State Management: Use Zustand for lesson state.
    * Verification: Test generation/display. Verify display correct. *Verify robust error handling and content sanitization.*

* **Task 2.4: Layout and General Animation**
    * Description: Implement pleasing layout and general animation effects, considering performance.
    * Target File: `src/components/KGVisualizationPanel.tsx`
    * Micro-processes: (Graph Layout Algorithm, Animation Effects, Zoom/Pan, Visual Hierarchy as before) *Evaluate performance impact of chosen layout algorithms/animations early.* *Consider initial implementation of graph virtualization or LOD if large graphs are expected.*
    * Verification: Layout clear. Animations smooth/usable. Controls intuitive. Hierarchy guides attention. *Initial performance check acceptable.*

* **Task 2.5: Edge Interaction Animations**
    * Description: Animate edge highlighting and information display efficiently.
    * Target File: `src/components/KGVisualizationPanel.tsx`
    * Micro-processes: (Hover Effect, Click Effect, Related Node Emphasis as before) *Ensure efficiency.*
    * Verification: Edge effects clear/informative. Animations highlight effectively. *Verify no performance impact.*

* **Task 2.6: Lasso Selection Animations**
    * Description: Provide performant animated feedback during lasso selection.
    * Target File: `src/components/KGVisualizationPanel.tsx`
    * Micro-processes: (Lasso Drawing, Node Selection Indication, Button Animation as before) *Ensure efficiency.*
    * Verification: Lasso drawing clear. Node selection indicated. Button animation noticeable but not distracting. *Verify no performance impact.*

* **Task 2.7: Progress Interaction (Placeholder)**
    * Description: Add basic interaction feedback to the progress indicator placeholder.
    * Target File: `src/components/KGVisualizationPanel.tsx`
    * Micro-processes: (Hover Effect, Click Effect as before)
    * Verification: Hover effect displays value. Optional click effect provides feedback.

* **Task 2.8: Backend API Enhancement (Prerequisite Path)**
    * Description: Implement backend logic to determine prerequisite paths efficiently.
    * Target File: `graphrag_tutor/api_server.py`, `graphrag_tutor/knowledge_graph/neo4j_kg_service.py`
    * Micro-processes:
        1.  API Endpoint Creation: Create endpoint (`/graph/node/{node_id}/prerequisites`).
        2.  KG Service Logic: Implement *optimized* Cypher query/traversal logic for prerequisites. Handle cycles gracefully. Return path elements.
        3.  Data Transformation: Return structured JSON.
    * Verification: Test endpoint. Verify correct paths. Handle nodes w/o prerequisites. *Verify query performance.* *Verify cycle handling.*

* **Task 2.9: Frontend Implementation (Prerequisite Path Highlighting)**
    * Description: Allow users to trigger and view prerequisite paths, handling empty states.
    * Target File: `src/components/KGVisualizationPanel.tsx`
    * Micro-processes:
        1.  UI Trigger: Add button/option "Show Prerequisites".
        2.  API Call: Call API. Handle loading/error states clearly.
        3.  State Management: Store path element IDs in Zustand.
        4.  Highlighting Logic: Apply distinct styles to path elements. Allow clearing highlights. *Ensure accessible indication of highlighted path (e.g., ARIA attributes).*
    * Verification: Test trigger. Verify paths fetched. Test visual highlighting. *Verify handling of "No prerequisites found" state.* *Verify accessibility of highlighted path.*

* **Task 2.10: Frontend Implementation (Comparative Concept Analysis - Selection & Trigger)**
    * Description: Allow users to select multiple nodes for comparison.
    * Target File: `src/components/KGVisualizationPanel.tsx`
    * Micro-processes: (Multi-Select UI, State Management, Trigger UI as before) *Ensure multi-select is keyboard accessible.*
    * Verification: Test multi-node selection. Verify state updated. Button enables/disables correctly. *Verify keyboard accessibility.*

* **Task 2.11: Backend API Enhancement (Comparative Analysis)**
    * Description: Implement backend logic for comparison, optimizing queries.
    * Target File: `graphrag_tutor/api_server.py`, `graphrag_tutor/knowledge_graph/neo4j_kg_service.py`, `graphrag_tutor/generation/llm_generator.py`
    * Micro-processes:
        1.  API Endpoint Creation: Create endpoint (`/graph/compare`) accepting node IDs.
        2.  KG Service Logic: Implement *optimized* logic for common/distinct neighbors.
        3.  LLM Integration (Optional): Fetch details, call LLM Generator with robust error handling for comparison summary.
    * Verification: Test endpoint. Verify correct neighbors. Test LLM summary generation/error handling. *Verify query performance.*

* **Task 2.12: Frontend Implementation (Comparative Analysis - Display)**
    * Description: Display comparison results, handling errors and sanitizing LLM content.
    * Target File: `src/components/KGVisualizationPanel.tsx`, potentially `RAGQAPanel.tsx`
    * Micro-processes:
        1.  API Call: Call `/graph/compare`. Handle loading/error states gracefully (e.g., show highlights even if summary fails).
        2.  Visual Highlighting: Apply distinct styles to common/distinct neighbors. *Ensure accessible indication.*
        3.  Summary Display: Display LLM summary (if available), *sanitizing content before rendering*. Handle empty/error states for summary.
    * Verification: Test trigger. Verify highlighting. Verify summary display/sanitization. *Verify graceful handling of partial failures.* *Verify accessibility.*

##   Phase 3: Advanced Features, Integration, & Commercial Polish

* **Task 3.1: Backend API Enhancement (Study Session from Path)**
    * Description: Enhance backend for structured study session generation from path, ensuring robustness.
    * Target File: `graphrag_tutor/api_server.py`, `graphrag_tutor/generation/lesson_generator.py`
    * Micro-processes:
        1.  API Endpoint Modification: Create/modify endpoint (`/lessons/generate_path_session`) accepting an *ordered* list of node IDs. Implement input validation.
        2.  Enhanced Lesson Logic: Update generator for sequential processing. Handle potential errors for individual nodes within the path. Optionally integrate quiz generation calls with error handling. Structure output clearly.
    * Verification: Test endpoint with valid/invalid paths. Verify sequential structure and relevant content. *Verify robust error handling for path processing.*

* **Task 3.2: Frontend Implementation (Path Selection & Session Generation)**
    * Description: Allow users to define paths and trigger session generation, handling errors and sanitizing output.
    * Target File: `src/components/KGVisualizationPanel.tsx`
    * Micro-processes:
        1.  Path Selection UI: Implement sequential selection UI. *Ensure keyboard accessibility.* Indicate path visually.
        2.  State Management: Store the ordered list of node IDs for the path in Zustand.
        3.  Trigger UI: Add "Generate Study Session" button, enabled when path has nodes.
        4.  API Call: Call API. Handle loading/error states clearly (e.g., "Session generation failed for step X").
        5.  Session Display: Display structured session. *Sanitize all generated content.*
    * Verification: Test path selection UI/accessibility. Test trigger/display. Ensure session follows path order. *Verify error handling and content sanitization.*

* **Task 3.3: Backend API Enhancement (Confidence/Mastery & Video Links)**
    * Description: Backend support for storing/retrieving user progress and curated video links securely and efficiently.
    * Target File: `graphrag_tutor/api_server.py`, User data storage (DB/KG), `neo4j_kg_service.py`
    * Micro-processes:
        1.  Data Storage: Define secure mechanism for user progress storage. Define storage for curated video links (e.g., node properties).
        2.  API Endpoints: Create secure endpoint (`PUT /user/progress/{node_id}`). Modify node detail endpoint to include progress/video links, ensuring efficient retrieval. Consider admin endpoint for video link management.
    * Verification: Test API security (authentication/authorization if applicable). Test update/retrieval. Verify efficiency.

* **Task 3.4: Frontend Implementation (Personalized Visualization & Confidence Input)**
    * Description: Adapt visualization based on user progress and allow secure input.
    * Target File: `src/components/KGVisualizationPanel.tsx`
    * Micro-processes:
        1.  Data Fetching: Fetch user progress securely.
        2.  State Management: Store progress in Zustand, handling potential sensitivity.
        3.  Dynamic Styling: Apply styles based on progress state.
        4.  Confidence Input UI: Add accessible input elements (rating scale/buttons).
        5.  API Call: Call `PUT /user/progress/{node_id}` securely on input change. Handle errors gracefully.
    * Verification: Test fetching/display. Verify styles adapt. Test accessible input UI and secure saving. Verify error handling.

* **Task 3.5: Frontend Implementation (Curated Video Link Display)**
    * Description: Display curated video links clearly and handle empty states.
    * Target File: `src/components/KGVisualizationPanel.tsx`
    * Micro-processes:
        1.  UI Display: Check for video links in fetched data.
        2.  Link Rendering: Render clickable links clearly (titles, potentially favicons). *Handle state where no videos are linked.*
    * Verification: Test with/without links. Verify display and click behavior. *Verify graceful handling of empty state.*

* **Task 3.6: Backend API Enhancement (Interactive Formula Calculation)**
    * Description: Backend endpoint for secure formula calculations.
    * Target File: `graphrag_tutor/api_server.py`, potentially `formula_calculator.py`
    * Micro-processes:
        1.  Formula Representation: Ensure accessible storage in KG.
        2.  API Endpoint Creation: Create endpoint (`/calculate/formula/{node_id}`) accepting node ID and variable values dict. Implement robust input validation.
        3.  Calculation Logic: Retrieve formula. *Use secure calculation method (e.g., math parser library, NOT `eval`)*. Handle calculation errors (div by zero, etc.). Return result.
    * Verification: Test endpoint with various formulas/inputs. Verify correct results. *Verify security (no arbitrary code execution).* *Verify robust error handling.*

* **Task 3.7: Frontend Implementation (Interactive Formula UI)**
    * Description: Allow users to input variables and see results, handling errors.
    * Target File: `src/components/KGVisualizationPanel.tsx`
    * Micro-processes:
        1.  Identify Formula Nodes: Determine if selected node is a formula node.
        2.  Input UI: Display accessible input fields for variables. Handle empty/initial state where calculation hasn't run.
        3.  API Call: Call calculation API. Handle loading/error states clearly (e.g., "Invalid input", "Calculation error").
        4.  Result Display: Display result clearly. Update on input change or button click.
    * Verification: Test input display logic. Test input handling/API call. Verify result display. *Verify user-friendly error handling for calculation/input issues.* *Verify empty state handling.*

* **Task 3.8: Comprehensive Testing (Integration & Edge Cases)**
    * Description: Test complete functionality, focusing on integration, graph-specific edge cases, and robustness.
    * Target Files: All relevant frontend/backend files, testing suites (`tests/`).
    * Micro-processes:
        1.  Write integration tests (e.g., MSW) covering all features and their interactions.
        2.  *Explicitly test graph edge cases:* Prerequisite loops, comparison of identical/unrelated nodes, path generation with disconnected nodes, formula errors (div zero, text input), behavior with empty graph, large graph simulation (performance).
        3.  *Test error recovery scenarios:* API failures, connection drops.
    * Verification: All integration tests pass. Features work seamlessly. *Edge cases handled correctly.* *Error recovery functions as expected.*

* **Task 3.9: UI/UX Refinement & Accessibility Deep Dive**
    * Description: Refine UI/UX based on comprehensive testing and feedback, ensuring high usability and accessibility, especially for graph interactions. *Emphasize maintainability in implementation.*
    * Target File: `src/components/KGVisualizationPanel.tsx`, testing results.
    * Micro-processes:
        1.  Conduct usability testing focusing on all features, especially complex interactions (path, compare, formula).
        2.  Gather feedback and iterate on UI/interactions.
        3.  *Implement subtle onboarding cues* or context-sensitive help for advanced features.
        4.  *Perform deep accessibility audit:* Test keyboard navigation (nodes, edges, controls), screen reader announcements for selections/highlights/paths, accessible alternatives for visual interactions (lasso, path drawing). Address any gaps found.
        5.  *Code Review for Maintainability:* Ensure component code is modular (e.g., custom hooks, sub-components), well-commented, and follows best practices. Consider adding Storybook stories for sub-components.
    * Verification: UI is intuitive, user-friendly. Features effective. *Onboarding cues helpful.* *Graph interactions fully accessible (keyboard, screen reader).* *Code structure promotes maintainability.*

* **Task 3.10: Performance Optimization (Proactive & Reactive)**
    * Description: Ensure high performance and responsiveness under expected commercial load.
    * Target Files: All relevant frontend/backend files.
    * Micro-processes:
        1.  *Implement chosen rendering strategy* (virtualization/LOD) if decided earlier (Task 2.4).
        2.  Profile frontend rendering (React DevTools) and optimize slow components/hooks.
        3.  Optimize backend query performance based on testing (Task 1.1, 2.8, etc.).
        4.  Optimize state management updates (Zustand selectors).
        5.  Benchmark performance with realistic large datasets/user loads.
    * Verification: Panel responsive under load. *Chosen rendering strategy effective.* Bottlenecks addressed.

* **Task 3.11: User-Customizable Styles**
    * Description: Allow users to customize visual appearance (optional, can be deferred post-MVP).
    * Target File: `src/components/KGVisualizationPanel.tsx`, `src/store/store.ts`
    * Micro-processes: (Style Settings UI, State Management, Dynamic Styling as before)
    * Verification: Users can customize styles. Preferences saved/applied correctly.

---
This final plan incorporates the requested context and the necessary considerations for building a commercially robust component.